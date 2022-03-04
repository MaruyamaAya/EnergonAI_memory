#include <c10/util/intrusive_ptr.h>
#include <c10d/Types.hpp>
#include <c10d/NCCLUtils.hpp>
#include <c10d/ProcessGroupNCCL.hpp>

class TorchProcessGroup: c10d::ProcessGroupNCCL
{
    public:

        c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCL::sendNcclUniqueId(ncclUniqueId ncclId, int dstRank)
        {
            auto ret = send();
            return send;
        }
        

        c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupNCCL::send(std::vector<at::Tensor>& tensors, int dstRank, int /* unused */) {
            check_gpu_tensors_different_devices(tensors);
            auto ret = pointToPoint(tensors, 
                                    [&](at::Tensor& input,
                                    ncclComm_t comm, 
                                    at::cuda::CUDAStream& stream,
                                    int dst) {
                                        torch::cuda::nccl::send(input, comm, stream, dst);
                                        return ncclSuccess;
                                    },
                                    dstRank,
                                    OpType::SEND,
                                    "nccl:send");
            return ret;
        }
};