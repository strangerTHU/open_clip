from functools import partial

import torch
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

_save_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
    torch.ops.aten.convolution.default,
}


def _custom_policy(ctx, func, *args, **kwargs):
    # print(func)
    to_save = func in _save_list
    return CheckpointPolicy.MUST_SAVE if to_save else CheckpointPolicy.PREFER_RECOMPUTE


def selective_checkpointing_context_fn():
    return partial(create_selective_checkpoint_contexts, _custom_policy)
