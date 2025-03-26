import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

__all__ = ["init_modules", "zero_last_gamma"]


def init_modules(model: nn.Module | list[nn.Module], init_type="trunc_normal") -> None:
    _DEFAULT_INIT_PARAM = {"trunc_normal": 0.02}

    if isinstance(model, list):
        for sub_module in model:
            init_modules(sub_module, init_type)
    else:
        init_params = init_type.split("@")
        init_params = float(init_params[1]) if len(init_params) > 1 else None

        if init_type.startswith("trunc_normal"):
            init_func = lambda param: nn.init.trunc_normal_(
                param, std=(_DEFAULT_INIT_PARAM["trunc_normal"] if init_params is None else init_params)
            )
        else:
            raise NotImplementedError

        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                init_func(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                init_func(m.weight)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.MultiheadAttention):
                init_func(m.in_proj_weight)
                if m.in_proj_bias is not None:
                    m.in_proj_bias.data.zero_()
                init_func(m.out_proj.weight)
                if m.out_proj.bias is not None:
                    m.out_proj.bias.data.zero_()
            elif isinstance(m, nn.Parameter):
                init_func(m)
            else:
                pass