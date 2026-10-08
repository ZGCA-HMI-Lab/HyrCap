#include <ATen/ATen.h>
#include <torch/extension.h>

using namespace at;

#define CHECK_CPU_INPUT(x)                                                   \
    TORCH_CHECK(!x.device().is_cuda(), #x " must be a CPU tensor");        \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

Tensor nms_1d_cpu(Tensor segments, Tensor scores, float iou_threshold)
{
    if (segments.numel() == 0)
        return at::empty({0}, segments.options().dtype(at::kLong));

    auto starts_tensor = segments.select(1, 0).contiguous();
    auto ends_tensor = segments.select(1, 1).contiguous();
    auto areas_tensor = ends_tensor - starts_tensor + 1e-6;
    auto order_tensor = std::get<1>(scores.sort(0, true));
    auto selected_tensor = at::ones(
        {segments.size(0)}, segments.options().dtype(at::kBool));

    auto selected = selected_tensor.data_ptr<bool>();
    auto order = order_tensor.data_ptr<int64_t>();
    auto starts = starts_tensor.data_ptr<float>();
    auto ends = ends_tensor.data_ptr<float>();
    auto areas = areas_tensor.data_ptr<float>();

    for (int64_t first = 0; first < segments.size(0); first++)
    {
        if (!selected[first])
            continue;
        auto current = order[first];
        for (int64_t second = first + 1; second < segments.size(0); second++)
        {
            if (!selected[second])
                continue;
            auto candidate = order[second];
            auto intersection = std::max(
                0.f,
                std::min(ends[current], ends[candidate])
                    - std::max(starts[current], starts[candidate]));
            auto overlap = intersection /
                (areas[current] + areas[candidate] - intersection);
            if (overlap >= iou_threshold)
                selected[second] = false;
        }
    }
    return order_tensor.masked_select(selected_tensor);
}

Tensor nms_1d(Tensor segments, Tensor scores, float iou_threshold)
{
    CHECK_CPU_INPUT(segments);
    CHECK_CPU_INPUT(scores);
    return nms_1d_cpu(segments, scores, iou_threshold);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)
{
    module.def(
        "nms",
        &nms_1d,
        "one-dimensional non-maximum suppression",
        py::arg("segments"),
        py::arg("scores"),
        py::arg("iou_threshold"));
}
