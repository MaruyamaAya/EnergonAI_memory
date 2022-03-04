import torch
from energon.core import global_context as gpc
from energon.context import ParallelMode
from energon.initialize import launch_from_torch
from FusedGPTLayer import DecoderWeights, FusedGPTLayer
import os


pp = 1  
tp = 2

# int64_t world_size, int64_t rank, int64_t tensor_para_size, int64_t local_rank, const c10::intrusive_ptr<c10d::ProcessGroup>& pg

launch_from_torch(tp_size =tp, pp_size = pp)

tensor_para_size = gpc.get_world_size(ParallelMode.PARALLEL_1D)

weights = DecoderWeights(12, 64, 1, 128, tp)

decoderLayer = FusedGPTLayer(12, 64, 128, 4, True, 1,  1, 0.0, 1.0, 1.0, weights.w[0])