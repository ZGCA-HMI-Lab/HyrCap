#pragma once

#ifdef WITH_CUDA
#include "cuda/ms_deform_attn_cuda.h"
#endif

at::Tensor
ms_deform_attn_forward(
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int seq2col_step)
{
    if (value.is_cuda())
    {
#ifdef WITH_CUDA
        return ms_deform_attn_cuda_forward(
            value, spatial_shapes, level_start_index, sampling_loc, attn_weight, seq2col_step);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}
