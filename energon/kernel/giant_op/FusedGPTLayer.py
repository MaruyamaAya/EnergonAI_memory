from __future__ import print_function

import os
import torch
import torch.nn as nn
import numpy as np
import importlib
from energon.core import global_context as gpc
from energon.context import ParallelMode


# hidden_dim
class DecoderWeights(object):
    def __init__(self, head_num, size_per_head, layer_num, max_seq_len, tensor_para_size):

        self.head_num = head_num
        self.size_per_head = size_per_head
        self.layer_num = layer_num
        self.max_seq_len = max_seq_len
        self.tensor_para_size = tensor_para_size

        local_head_num = head_num // tensor_para_size
        global_head_number = head_num
        local_hidden_units = local_head_num * size_per_head
        global_hidden_units = global_head_number * size_per_head
        local_inner_size = local_hidden_units * 4

        self.local_head_num = head_num // tensor_para_size
        self.global_head_number = head_num
        self.local_hidden_units = local_head_num * size_per_head
        self.global_hidden_units = global_head_number * size_per_head
        self.local_inner_size = local_hidden_units * 4

        self.w = [[] for _ in range(layer_num)]
        for layer_weights in self.w:
            self.w.append(torch.zeros(global_hidden_units))   # self_layernorm_gamma
            self.w.append(torch.zeros(global_hidden_units))   # self_layernorm_beta
            self.w.append(torch.zeros(global_hidden_units, local_hidden_units * 3))   # self_kernel
            self.w.append(torch.zeros(local_hidden_units * 3))   # self_bias
            self.w.append(torch.zeros(local_hidden_units, global_hidden_units))   # self_output_kernel
            self.w.append(torch.zeros(global_hidden_units))   # self_output_bias
            self.w.append(torch.zeros(global_hidden_units))   # ffn_layernorm_gamma
            self.w.append(torch.zeros(global_hidden_units))   # ffn_layernorm_beta
            self.w.append(torch.zeros(global_hidden_units, local_inner_size))   # ffn_kernel1
            self.w.append(torch.zeros(local_inner_size, global_hidden_units))   # ffn_kernel2
            self.w.append(torch.zeros(local_inner_size))   # ffn_bias1
            self.w.append(torch.zeros(global_hidden_units))   # ffn_bias2
            
            for i in range(len(layer_weights)):
                torch.nn.init.uniform_(layer_weights[i], -1, 1)
    
    # def __getitem__(self, idx):
    #     return self.w[idx]

    # def __setitem__(self, idx, val):
    #     self.w[idx] = val

    # def __len__(self):    
    #     return len(self.w)

    # def to_cuda(self):
    #     for i in range(self.layer_num):
    #         for j in range(len(self.w[i])):
    #             self.w[i][j] = self.w[i][j].cuda()

    # def to_half(self):
    #     for i in range(self.layer_num):
    #         for j in range(len(self.w[i])):
    #             self.w[i][j] = self.w[i][j].half()




class FusedGPTLayer(nn.Module):
    def __init__(self, head_num, size_per_head, max_seq_len, max_batch_size, is_fuse_QKV, candidate_num, probability_threshold, temperature, repetition_penalty,
                weights, lib_path="/home/lcdjs/ColossalAI-Inference/energon/kernel/giant_op/csrc/build/libFTdecoder.so"):
        super().__init__

        assert torch.cuda.is_available(), "CUDA is required for this model."        
        nccl_func = importlib.import_module("energon_nccl")

        local_rank = gpc.get_local_rank(ParallelMode.PARALLEL_1D)
        tensor_para_size = gpc.get_world_size(ParallelMode.PARALLEL_1D)
        assert head_num % self.tensor_para_size == 0, "head_num must be a multiple of tensor_para_size."
        
        ncclid = None
        if(gpc.is_initialized(ParallelMode.PARALLEL_1D)):
            ncclid = nccl_func.init_nccl(tensor_para_size, local_rank, gpc.get_group(ParallelMode.PARALLEL_1D))    
        
        torch.classes.load_library(os.path.abspath(lib_path))
        self.model = torch.classes.Giant.FusedGPTLayer(head_num, size_per_head, max_seq_len, max_batch_size, tensor_para_size, tensor_para_rank, 
                                                                   is_fuse_QKV, candidate_num, probability_threshold, temperature, repetition_penalty, ncclid,
                                                                   *weights)
    
                                                        

    # def load(self, ckpt_path):
    #     self.weights.load(ckpt_path, tensor_para_rank=self.tensor_para_rank, layer_para_rank=self.layer_para_rank)

    # def half(self):
    #     self.weights._map(lambda w : w.half())
    #     if self.model is not None:
    #         self.cuda()

    # def cuda(self):
    #     self.weights._map(lambda w : w.cuda(self.device))
    #     self.model = torch.classes.FasterTransformer.GPT(self.head_num, self.size_per_head, self.vocab_size,
    #                                                     self.start_id, self.end_id, self.layer_num, self.top_k, self.top_p, self.temperature, self.max_seq_len,
    #                                                     self.tensor_para_size, self.layer_para_size, self.layer_para_batch_size, 
    #                                                     True, self.max_batch_size, 1.0, *self.weights.w)

    # def forward(self, start_ids, start_lengths, attn_mask, batch_first=True):
    #     batch_size = start_ids.size(0)
    #     assert batch_size <= self.max_batch_size, "batch_size must not exceed max_batch_size."
    #     assert batch_size >= self.layer_para_batch_size, "batch_size must be equal to or larger than layer_para_batch_size."
    #     assert batch_size % self.layer_para_batch_size == 0, "batch_size must be a multiple of layer_para_batch_size."

    #     input_len = min(start_lengths)
    #     assert input_len > 0, "input_len must be larger than zero. For an unconditional case, use start_id as the first token."
    #     assert input_len + self.output_len <= self.max_seq_len, "input_len + output_len must not exceed max_seq_len."

    #     # Inputs to device
    #     start_ids = start_ids.cuda(self.device)
    #     start_lengths = start_lengths.cuda(self.device)
    #     attn_mask = attn_mask.cuda(self.device)

    #     assert self.model is not None, "The model must be copied to the device(s) through cuda()."

    #     output_ids, = self.model.forward(start_ids, start_lengths, attn_mask, self.output_len)
    #     if batch_first:
    #         output_ids = output_ids.T

    #     if self.rank == 0:
    #         return output_ids





























            # layer_weights.append(torch.zeros(global_hidden_units))   # self_layernorm_gamma
            # layer_weights.append(torch.zeros(global_hidden_units))   # self_layernorm_beta
            # layer_weights.append(torch.zeros(global_hidden_units, local_hidden_units))   # self_kernel_q
            # layer_weights.append(torch.zeros(global_hidden_units, local_hidden_units))   # self_kernel_k
            # layer_weights.append(torch.zeros(global_hidden_units, local_hidden_units))   # self_kernel_v
            # layer_weights.append(torch.zeros(local_hidden_units))   # self_bias_q
            # layer_weights.append(torch.zeros(local_hidden_units))   # self_bias_k
            # layer_weights.append(torch.zeros(local_hidden_units))   # self_bias_v
            # layer_weights.append(torch.zeros(local_hidden_units, global_hidden_units))   # self_output_kernel
            # layer_weights.append(torch.zeros(global_hidden_units))   # self_output_bias
            # layer_weights.append(torch.zeros(global_hidden_units))   # cross_layernorm_gamma
            # layer_weights.append(torch.zeros(global_hidden_units))   # cross_layernorm_beta


            # layer_weights.append(torch.zeros(hidden_dim, hidden_dim))   # cross_kernel_q
            # layer_weights.append(torch.zeros(hidden_dim, hidden_dim))   # cross_kernel_k
            # layer_weights.append(torch.zeros(hidden_dim, hidden_dim))   # cross_kernel_v

            # layer_weights.append(torch.zeros(hidden_dim))   # cross_bias_q
            # layer_weights.append(torch.zeros(hidden_dim))   # cross_bias_k
            # layer_weights.append(torch.zeros(hidden_dim))   # cross_bias_v
            # layer_weights.append(torch.zeros(hidden_dim, hidden_dim))   # cross_output_kernel
            # layer_weights.append(torch.zeros(hidden_dim))   # cross_output_bias
            # layer_weights.append(torch.zeros(hidden_dim))   # ffn_layernorm_gamma
            # layer_weights.append(torch.zeros(hidden_dim))   # ffn_layernorm_beta
            # layer_weights.append(torch.zeros(hidden_dim, 4 * hidden_dim))   # inter_kernel
            # layer_weights.append(torch.zeros(4 * hidden_dim))   # inter_bias
            # layer_weights.append(torch.zeros(4 * hidden_dim, hidden_dim))   # output_kernel
            # layer_weights.append(torch.zeros(hidden_dim))   # output_bias