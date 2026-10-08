#include "cuda/ms_deform_im2col_cuda.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

at::Tensor ms_deform_attn_cuda_forward(
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int column_step)
{
    AT_ASSERTM(value.is_contiguous(), "value tensor must be contiguous");
    AT_ASSERTM(spatial_shapes.is_contiguous(), "shape tensor must be contiguous");
    AT_ASSERTM(level_start_index.is_contiguous(), "index tensor must be contiguous");
    AT_ASSERTM(sampling_loc.is_contiguous(), "locations must be contiguous");
    AT_ASSERTM(attn_weight.is_contiguous(), "weights must be contiguous");
    AT_ASSERTM(value.is_cuda(), "value must be an accelerator tensor");
    AT_ASSERTM(spatial_shapes.is_cuda(), "shape tensor must be on the accelerator");
    AT_ASSERTM(level_start_index.is_cuda(), "index tensor must be on the accelerator");
    AT_ASSERTM(sampling_loc.is_cuda(), "locations must be on the accelerator");
    AT_ASSERTM(attn_weight.is_cuda(), "weights must be on the accelerator");

    const int batch = value.size(0);
    const int spatial_size = value.size(1);
    const int heads = value.size(2);
    const int channels = value.size(3);
    const int levels = spatial_shapes.size(0);
    const int queries = sampling_loc.size(1);
    const int points = sampling_loc.size(4);
    const int step = std::min(batch, column_step);
    AT_ASSERTM(batch % step == 0, "batch must be divisible by column step");

    auto output = at::zeros({batch, queries, heads, channels}, value.options());
    auto grouped = output.view({batch / step, step, queries, heads, channels});
    const auto value_size = spatial_size * heads * channels;
    const auto location_size = queries * heads * levels * points;
    const auto weight_size = queries * heads * levels * points;

    for (int group = 0; group < batch / step; ++group)
    {
        auto columns = grouped.select(0, group);
        AT_DISPATCH_FLOATING_TYPES(
            value.scalar_type(), "ms_deform_attn_forward_cuda", ([&] {
                ms_deformable_im2col_cuda(
                    at::cuda::getCurrentCUDAStream(),
                    value.data_ptr<scalar_t>() + group * step * value_size,
                    spatial_shapes.data_ptr<int64_t>(),
                    level_start_index.data_ptr<int64_t>(),
                    sampling_loc.data_ptr<scalar_t>() + group * step * location_size,
                    attn_weight.data_ptr<scalar_t>() + group * step * weight_size,
                    step,
                    spatial_size,
                    heads,
                    channels,
                    levels,
                    queries,
                    points,
                    columns.data_ptr<scalar_t>());
            }));
    }
    return output.view({batch, queries, heads * channels});
}
