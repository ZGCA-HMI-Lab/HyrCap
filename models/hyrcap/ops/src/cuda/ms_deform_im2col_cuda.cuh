#include <cstdio>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#define CUDA_KERNEL_LOOP(index, count)                                      \
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < count; \
         index += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;

inline int get_blocks(const int count, const int threads)
{
    return (count + threads - 1) / threads;
}

template <typename scalar_t>
__device__ scalar_t linear_sample(
    const scalar_t *&values,
    const int width,
    const int heads,
    const int channels,
    const scalar_t position,
    const int head,
    const int channel)
{
    const int lower = floor(position);
    const int upper = lower + 1;
    const scalar_t upper_weight = position - lower;
    const scalar_t lower_weight = 1 - upper_weight;
    const int stride = heads * channels;
    const int base = head * channels + channel;
    scalar_t lower_value = 0;
    scalar_t upper_value = 0;
    if (lower >= 0)
        lower_value = values[lower * stride + base];
    if (upper <= width - 1)
        upper_value = values[upper * stride + base];
    return lower_weight * lower_value + upper_weight * upper_value;
}

template <typename scalar_t>
__global__ void temporal_sample_kernel(
    const int count,
    const scalar_t *values,
    const int64_t *shapes,
    const int64_t *level_start_index,
    const scalar_t *sampling_locations,
    const scalar_t *attention_weights,
    const int batch_size,
    const int spatial_size,
    const int heads,
    const int channels,
    const int levels,
    const int queries,
    const int points,
    scalar_t *columns)
{
    CUDA_KERNEL_LOOP(index, count)
    {
        int cursor = index;
        const int channel = cursor % channels;
        cursor /= channels;
        const int sampling_index = cursor;
        const int head = cursor % heads;
        cursor /= heads;
        cursor /= queries;
        const int batch = cursor;

        int weight_index = sampling_index * levels * points;
        int location_index = weight_index;
        const int value_stride = heads * channels;
        const int batch_offset = batch * spatial_size * value_stride;
        scalar_t result = 0;

        for (int level = 0; level < levels; ++level)
        {
            const int start = level_start_index[level];
            const int width = shapes[level];
            const scalar_t *level_values = values + batch_offset + start * value_stride;
            for (int point = 0; point < points; ++point)
            {
                const scalar_t position = sampling_locations[location_index] * width - 0.5;
                if (position > -1 && position < width)
                    result += linear_sample(
                        level_values,
                        width,
                        heads,
                        channels,
                        position,
                        head,
                        channel) * attention_weights[weight_index];
                ++weight_index;
                ++location_index;
            }
        }
        columns[index] = result;
    }
}

template <typename scalar_t>
void ms_deformable_im2col_cuda(
    cudaStream_t stream,
    const scalar_t *values,
    const int64_t *shapes,
    const int64_t *level_start_index,
    const scalar_t *sampling_locations,
    const scalar_t *attention_weights,
    const int batch_size,
    const int spatial_size,
    const int heads,
    const int channels,
    const int levels,
    const int queries,
    const int points,
    scalar_t *columns)
{
    const int count = batch_size * queries * heads * channels;
    temporal_sample_kernel<scalar_t>
        <<<get_blocks(count, CUDA_NUM_THREADS), CUDA_NUM_THREADS, 0, stream>>>(
            count,
            values,
            shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            batch_size,
            spatial_size,
            heads,
            channels,
            levels,
            queries,
            points,
            columns);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        printf("temporal attention kernel failed: %s\n", cudaGetErrorString(error));
}
