/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// #include "fastertransformer/th_op/decoder.h"
#include <vector>
#include <iostream>
#include <cuda_fp16.h>

#include <torch/script.h>
#include <torch/custom_class.h>
#include "torch/csrc/cuda/Stream.h"

#include "fastertransformer/open_decoder.h"
#include "fastertransformer/th_op/th_traits.h"
#include "fastertransformer/th_op/utils.h"

#include <c10/util/intrusive_ptr.h>
#include <c10d/Types.hpp>
#include <c10d/NCCLUtils.hpp>
#include <c10d/ProcessGroupNCCL.hpp>
#include <c10d/ProcessGroup.hpp>
// #include <cuda_runtime.h>

namespace torch_ext {
using namespace fastertransformer;
using torch::Tensor;

class IFTDecoder {
public:
  virtual ~IFTDecoder() {}
  // virtual void forward_context(Tensor in_hidden_states, Tensor out_hidden_states, Tensor attention_mask) = 0;
  virtual void forward(Tensor in_hidden_states, Tensor out_hidden_states, Tensor attention_mask) = 0;
};

template <class T>
class FTDecoder : public IFTDecoder {
public:
  FTDecoder(const int head_num, const int size_per_head, const int candidate_num, 
          const float probability_threshold, const float temperature,  const int max_seq_len,
          const int tensor_para_size, const bool is_fuse_QKV, const int max_batch_size, const float repetition_penalty,
          const std::vector<Tensor>& weights_transformer, TensorParallelParam &t_parallel_param)
          :max_seq_len_(max_seq_len), size_per_head_(size_per_head), candidate_num_(candidate_num),
          probability_threshold_(probability_threshold), temperature_(temperature), is_fuse_QKV_(is_fuse_QKV),
          max_batch_size_(max_batch_size), repetition_penalty_(repetition_penalty), tensor_parallel_param_(t_parallel_param)
  {
    const int local_head_num = head_num / tensor_para_size; 
    const int global_head_num = head_num; 
    const int local_hidden_units = local_head_num * size_per_head;
    const int global_hidden_units = global_head_num * size_per_head;
    const int local_inner_size = local_hidden_units * 4;

    global_head_num_ = global_head_num;
    hidden_units_ = head_num * size_per_head;

    tensor_para_nccl_comm = t_parallel_param.nccl_comm;
    Tensor self_layernorm_gamma = weights_transformer[0];
    Tensor self_layernorm_beta = weights_transformer[1];
    Tensor self_kernel = weights_transformer[2];
    Tensor self_bias = weights_transformer[3];
    Tensor self_output_kernel = weights_transformer[4];
    Tensor self_output_bias = weights_transformer[5];
    Tensor ffn_layernorm_gamma = weights_transformer[6];
    Tensor ffn_layernorm_beta = weights_transformer[7];
    Tensor ffn_kernel1 = weights_transformer[8];
    Tensor ffn_kernel2 = weights_transformer[9];
    Tensor ffn_bias1 = weights_transformer[10];
    Tensor ffn_bias2 = weights_transformer[11];

    // init weighs **************************************************************
    T *self_Q_kernel, *self_K_kernel, *self_V_kernel;
    T *self_Q_bias, *self_K_bias, *self_V_bias;
    self_Q_kernel = get_ptr<T>(self_kernel);
    self_K_kernel = self_Q_kernel + global_hidden_units * local_hidden_units;
    self_V_kernel = self_K_kernel + global_hidden_units * local_hidden_units;
    self_Q_bias = get_ptr<T>(self_bias);
    self_K_bias = self_Q_bias + local_hidden_units;
    self_V_bias = self_K_bias + local_hidden_units;    
    decoder_params->self_layernorm.gamma = get_ptr<T>(self_layernorm_gamma);
    decoder_params->self_layernorm.beta = get_ptr<T>(self_layernorm_beta);
    decoder_params->self_attention.query_weight.kernel = self_Q_kernel;
    decoder_params->self_attention.key_weight.kernel = self_K_kernel;
    decoder_params->self_attention.value_weight.kernel = self_V_kernel;
    decoder_params->self_attention.attention_output_weight.kernel = get_ptr<T>(self_output_kernel);
    decoder_params->self_attention.query_weight.bias = self_Q_bias;
    decoder_params->self_attention.key_weight.bias = self_K_bias;
    decoder_params->self_attention.value_weight.bias = self_V_bias;
    decoder_params->self_attention.attention_output_weight.bias = get_ptr<T>(self_output_bias);
    decoder_params->ffn_layernorm.gamma = get_ptr<T>(ffn_layernorm_gamma);
    decoder_params->ffn_layernorm.beta = get_ptr<T>(ffn_layernorm_beta);
    decoder_params->ffn.intermediate_weight.bias = get_ptr<T>(ffn_bias1);
    decoder_params->ffn.output_weight.bias = get_ptr<T>(ffn_bias2);
    decoder_params->ffn.intermediate_weight.kernel = get_ptr<T>(ffn_kernel1);
    decoder_params->ffn.output_weight.kernel = get_ptr<T>(ffn_kernel2);
    // ****************************************************************************
    decoder_ = new OpenDecoder<THTraits<T>::OpType>(global_head_num_, size_per_head_, 0 /* memory_hidden_units*/, is_fuse_QKV);
    // decoder_buf_ ***************************************************************
    
    Allocator<AllocatorType::TH> allocator;
    size_t from_tensor_size_ = max_seq_len_ * hidden_units_;
    size_t decoder_normed_result_buffer_size = max_batch_size_ * hidden_units_;
    size_t decoder_workspace_size = (size_t)decoder_->getWorkspaceSize(); 
    size_t cache_size = max_seq_len_ * max_seq_len_ * hidden_units_ / tensor_para_size;
    size_t datatype_buf_size = decoder_workspace_size + cache_size * 2 + decoder_normed_result_buffer_size;   
    K_cache_ = new DataType_ *[1];
    V_cache_ = new DataType_ *[1];
    buf_ = reinterpret_cast<void *>(allocator.malloc(
            ((sizeof(DataType_) == sizeof(half)) ? CUBLAS_WORKSPACE_SIZE : 0) + sizeof(DataType_) * datatype_buf_size));
    
    if (sizeof(DataType_) == sizeof(half)){
          cublas_workspace_ = buf_;
          K_cache_[0] = (DataType_ *)((char*)cublas_workspace_ + CUBLAS_WORKSPACE_SIZE);
    }else{
          cublas_workspace_ = nullptr;
          K_cache_[0] = (DataType_ *)buf_;
    }
    
    V_cache_[0] = (DataType_ *)(K_cache_[0] + cache_size);
    decoder_buf_ = V_cache_[0] + cache_size;
    // ****************************************************************************
    // set decoder ***************************************************************
    decoder_->set_max_batch_size(max_batch_size);
    decoder_->set_tensor_parallel_param(t_parallel_param);
    // void initialize(DecoderInitParam<DataType_> param, DataType_ *buf, void *cublas_workapsce, bool set_local_batch = true)
    decoder_->initialize(*decoder_params, decoder_buf_, cublas_workspace_, false); // match the allocated buffer with the value name, can be called in each forward.
    // ****************************************************************************
    check_cuda_error(cublasLtCreate(&cublasLtHandle));
  }

  ~FTDecoder() override {
    ncclCommDestroy(tensor_para_nccl_comm);    
    cublasLtDestroy(cublasLtHandle);
    delete decoder_params;
    delete decoder_;
  }

  // void forward_context(DataType_* workspace,
  //                        DataType_ *decoder_output, 
  //                        DataType_ *key_cache_,  
  //                        DataType_ *value_cache_,
  //                        const DataType_ *from_tensor,
  //                        const DataType_ *d_attn_mask,
  //                        const int local_batch_size,
  //                        const int seq_len,
  //                        const int ite,
  //                        const int max_seq_len,
  //                        const bool is_final)
  
  // void forward_context(Tensor in_hidden_states, Tensor out_hidden_states, Tensor attention_mask) override
  // {
  //   DataType_* in = get_ptr<T>(in_hidden_states);
  //   DataType_* out = get_ptr<T>(out_hidden_states);
  //   DataType_* att_mk = get_ptr<T>(attention_mask);
  //   // from_tensor_[0] = in_hidden_states;
  //   // from_tensor_[1] = out_hidden_states;  
  //   decoder_->forward_context(decoder_workspace,
  //                             in,
  //                             K_cache_[0], // 暂不考虑增量推理
  //                             V_cache_[0],
  //                             out,
  //                             att_mk,
  //                             max_batch_size_,
  //                             10,
  //                             1,
  //                             max_seq_len_,
  //                             false);
  // }

  // void forward_v2(const DataType_ *from_tensor, const DataType_ *memory_tensor,
  //                   DataType_ *key_cache_, DataType_ *value_cache_,
  //                   DataType_ *key_mem_cache_, DataType_ *value_mem_cache_,
  //                   const int *memory_sequence_length, DataType_ *decoder_output, const int step,
  //                   const int decoder_max_seq_len, 
  //                   const bool is_cross_attention, 
  //                   const bool* finished = nullptr,
  //                   const int max_input_len = 0, const int *input_lengths = nullptr)

  void forward(Tensor in_hidden_states, Tensor out_hidden_states, Tensor attention_mask) override
  {
    DataType_* in = get_ptr<T>(in_hidden_states);
    DataType_* out = get_ptr<T>(out_hidden_states);
    DataType_* att_mk = get_ptr<T>(attention_mask);

    decoder_->forward_v2(in, nullptr, // memory_tensor should be nullptr
                          K_cache_[0], V_cache_[0],
                          nullptr, nullptr, // key_mem_cache_ and value_mem_cache_ should be nullptr
                          nullptr, // memory_sequence_length should be nullptr
                          out, 1, 128, false, nullptr, 0, nullptr);
  }

private:

  const int size_per_head_;  
  const int candidate_num_;  
  const float probability_threshold_;
  const float temperature_;
  const int max_seq_len_;
  int tensor_para_size_;
  const int is_fuse_QKV_;
  const int max_batch_size_;
  const float repetition_penalty_;
  int global_head_num_;
  int hidden_units_;
  size_t from_tensor_size_;
  cublasLtHandle_t cublasLtHandle;

  TensorParallelParam tensor_parallel_param_;
  ncclComm_t tensor_para_nccl_comm;
  OpenDecoder<THTraits<T>::OpType> *decoder_;
  typedef DecoderTransformerTraits<THTraits<T>::OpType> Traits_;
  typedef typename Traits_::DataType DataType_;

  void *buf_;
  DataType_ **K_cache_;
  DataType_ **V_cache_;
  DataType_ *from_tensor_[2];  // input ? input and output through pass in  
  DecoderInitParam<T>* decoder_params;
  void *cublas_workspace_ = nullptr;
  DataType_ *decoder_buf_;
};


// template <typename>
class FusedGPTLayer : public torch::jit::CustomClassHolder {
public:
  FusedGPTLayer(
      const int64_t head_num,
      const int64_t size_per_head,
      const int64_t max_seq_len,
      const int64_t max_batch_size,
      const int64_t tensor_para_size,
      const int64_t local_rank,
      const bool is_fuse_QKV,
      const int64_t candidate_num, // not use
      const double probability_threshold, // not use
      const double temperature, // not use
      const double repetition_penalty, // not use
      Tensor ncclid,
      Tensor self_layernorm_gamma,
      Tensor self_layernorm_beta,
      Tensor self_kernel,
      Tensor self_bias,
      Tensor self_output_kernel,
      Tensor self_output_bias,
      Tensor ffn_layernorm_gamma,
      Tensor ffn_layernorm_beta,
      Tensor ffn_kernel1,
      Tensor ffn_kernel2,
      Tensor ffn_bias1,
      Tensor ffn_bias2
    ):st_(self_layernorm_gamma.scalar_type()),
      weights_transformer{self_layernorm_gamma, self_layernorm_beta, self_kernel, self_bias,
              self_output_kernel, self_output_bias, ffn_layernorm_gamma, ffn_layernorm_beta,
              ffn_kernel1, ffn_kernel2, ffn_bias1, ffn_bias2}  
  {
        CHECK_INPUT(self_layernorm_gamma, st_); 
        CHECK_INPUT(self_layernorm_beta, st_);
        CHECK_INPUT(self_kernel, st_); 
        CHECK_INPUT(self_bias, st_);
        CHECK_INPUT(self_output_kernel, st_); 
        CHECK_INPUT(self_output_bias, st_);
        CHECK_INPUT(ffn_layernorm_gamma, st_); 
        CHECK_INPUT(ffn_layernorm_beta, st_);
        CHECK_INPUT(ffn_kernel1, st_); 
        CHECK_INPUT(ffn_kernel2, st_);
        CHECK_INPUT(ffn_bias1, st_); 
        CHECK_INPUT(ffn_bias2, st_);        

        ncclUniqueId tensor_para_nccl_uid;
        char* temp = get_ptr<char>(ncclid);

        for(int i = 0; i<NCCL_UNIQUE_ID_BYTES; i++){
          tensor_para_nccl_uid.internal[i] = temp[i];
          // std::cout<<tensor_para_nccl_uid.internal[i]-48;
        }

        ncclComm_t tensor_para_nccl_comm;
        NCCLCHECK(ncclCommInitRank(&tensor_para_nccl_comm, tensor_para_size, tensor_para_nccl_uid, local_rank));
        
        int local_head_num = head_num / tensor_para_size;
        int local_hidden_units = local_head_num * size_per_head;

        t_parallel_param_.rank = local_rank;
        t_parallel_param_.world_size = tensor_para_size;
        t_parallel_param_.nccl_comm = tensor_para_nccl_comm;
        t_parallel_param_.local_head_num_ = local_head_num;
        t_parallel_param_.local_hidden_units_ = local_hidden_units; 

        switch (st_) {
          case at::ScalarType::Float:
          ftdecoder = new FTDecoder<float>(head_num, size_per_head, candidate_num, probability_threshold, temperature, max_seq_len, 
                                        tensor_para_size, is_fuse_QKV, max_batch_size, repetition_penalty, 
                                        weights_transformer, t_parallel_param_);
          break;
          case at::ScalarType::Half:
          ftdecoder = new FTDecoder<half>(head_num, size_per_head, candidate_num, probability_threshold, temperature, max_seq_len, 
                                      tensor_para_size, is_fuse_QKV, max_batch_size, repetition_penalty,
                                      weights_transformer, t_parallel_param_);
          break;
          default:
          throw std::runtime_error("Wrong Tensor type.");
        }
  }

  ~FusedGPTLayer() {
    delete ftdecoder;
  }

  Tensor forward(Tensor in_hidden_states, Tensor attention_mask){
    CHECK_INPUT(in_hidden_states, st_);
    CHECK_INPUT(attention_mask, st_);
    
    auto out_hidden_states = torch::empty_like(in_hidden_states);

    ftdecoder->forward(in_hidden_states, out_hidden_states, attention_mask);
    return out_hidden_states;
  } 

  // Tensor forward_context(Tensor in_hidden_states, Tensor attention_mask){
  //   CHECK_INPUT(in_hidden_states, _st);
  //   CHECK_INPUT(out_hidden_states, _st);
  //   CHECK_INPUT(attention_mask, _st);
    
  //   auto out_hidden_states torch::empty_like(in_hidden_states);

  //   ftdecoder->forward_context(in_hidden_states, out_hidden_states, attention_mask);

  //   return out_hidden_states;
  // }

  // void InitModel(int64_t tensor_para_size, int64_t local_rank, at::){
  // }

  // }

  // std::vector<Tensor> get_pickle_info() const;

private:
  const at::ScalarType st_;
  torch_ext::IFTDecoder* ftdecoder;
  std::vector<Tensor> weights_transformer;
  TensorParallelParam t_parallel_param_;
  int64_t head_num;
  int64_t size_per_head;
  int64_t max_seq_len;
  int max_batch_size;
  int64_t tensor_para_size;
  bool is_fuse_QKV;
  int64_t candidate_num; // not use
  double probability_threshold; // not use
  double temperature; // not use
  double repetition_penalty;
  
};
} // namespace torch_ext

using torch::Tensor;

static auto fasterTransformerDecoderTHS = 
    torch::jit::class_<torch_ext::FusedGPTLayer>("Giant","FusedGPTLayer")
    .def(torch::jit::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool, int64_t, double, double, double,
                        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>())
    .def("forward", &torch_ext::FusedGPTLayer::forward);

    // .def("forward_context", &torch_ext::FusedGPTLayer::forward_context)    
    // .def("merge", &torch_ext::FusedGPTLayer::merge)
    // .def("InitModel", &torch_ext::FusedGPTLayer::InitModel)