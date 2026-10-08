"""Autograd wrapper for the native temporal attention operator."""

from torch.autograd import Function

try:
    from .. import MultiScaleTemporalDeformableAttention as MSDA
except ImportError:
    import MultiScaleTemporalDeformableAttention as MSDA


class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(
        context,
        value,
        value_shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
        column_step,
    ):
        del context
        return MSDA.ms_deform_attn_forward(
            value,
            value_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            column_step,
        )
