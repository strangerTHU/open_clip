import copy

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
from typing import Any, Optional

def torch_random(generator: Optional[torch.Generator] = None) -> float:
    """uniform distribution on the interval [0, 1)"""
    return float(torch.rand(1, generator=generator))

def torch_uniform(low: float, high: float, generator: Optional[torch.Generator] = None) -> float:
    """uniform distribution on the interval [low, high)"""
    rand_val = torch_random(generator)
    return (high - low) * rand_val + low


def torch_random_choices(
    src_list: list[Any],
    generator: Optional[torch.Generator] = None,
    k=1,
    weight_list: Optional[list[float]] = None,
) -> Any | list:
    if weight_list is None:
        rand_idx = torch.randint(low=0, high=len(src_list), generator=generator, size=(k,))
        out_list = [src_list[i] for i in rand_idx]
    else:
        assert len(weight_list) == len(src_list)
        accumulate_weight_list = np.cumsum(weight_list)

        out_list = []
        for _ in range(k):
            val = torch_uniform(0, accumulate_weight_list[-1], generator)
            active_id = 0
            for i, weight_val in enumerate(accumulate_weight_list):
                active_id = i
                if weight_val > val:
                    break
            out_list.append(src_list[active_id])

    return out_list[0] if k == 1 else out_list

__all__ = [
    "RRSController",
    "get_interpolate",
    "MyRandomResizedCrop",
]


class RRSController:
    ACTIVE_SIZE = (224, 224)
    IMAGE_SIZE_LIST = [(224, 224)]

    CHOICE_LIST = None

    @staticmethod
    def get_candidates() -> list[tuple[int, int]]:
        return copy.deepcopy(RRSController.IMAGE_SIZE_LIST)

    @staticmethod
    def sample_resolution(batch_id: int) -> None:
        RRSController.ACTIVE_SIZE = RRSController.CHOICE_LIST[batch_id]

    @staticmethod
    def set_epoch(epoch: int, batch_per_epoch: int) -> None:
        g = torch.Generator()
        g.manual_seed(epoch)
        RRSController.CHOICE_LIST = torch_random_choices(
            RRSController.get_candidates(),
            g,
            batch_per_epoch,
        )


def get_interpolate(name: str) -> F.InterpolationMode:
    mapping = {
        "nearest": F.InterpolationMode.NEAREST,
        "bilinear": F.InterpolationMode.BILINEAR,
        "bicubic": F.InterpolationMode.BICUBIC,
        "box": F.InterpolationMode.BOX,
        "hamming": F.InterpolationMode.HAMMING,
        "lanczos": F.InterpolationMode.LANCZOS,
    }
    if name in mapping:
        return mapping[name]
    elif name == "random":
        return torch_random_choices(
            [
                F.InterpolationMode.NEAREST,
                F.InterpolationMode.BILINEAR,
                F.InterpolationMode.BICUBIC,
                F.InterpolationMode.BOX,
                F.InterpolationMode.HAMMING,
                F.InterpolationMode.LANCZOS,
            ],
        )
    else:
        raise NotImplementedError


class MyRandomResizedCrop(transforms.RandomResizedCrop):
    def __init__(
        self,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation: str = "random",
    ):
        super(MyRandomResizedCrop, self).__init__(224, scale, ratio)
        self.interpolation = interpolation

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        i, j, h, w = self.get_params(img, list(self.scale), list(self.ratio))
        target_size = RRSController.ACTIVE_SIZE
        return F.resized_crop(img, i, j, h, w, list(target_size), get_interpolate(self.interpolation))

    def __repr__(self) -> str:
        format_string = self.__class__.__name__
        format_string += f"(\n\tsize={RRSController.get_candidates()},\n"
        format_string += f"\tscale={tuple(round(s, 4) for s in self.scale)},\n"
        format_string += f"\tratio={tuple(round(r, 4) for r in self.ratio)},\n"
        format_string += f"\tinterpolation={self.interpolation})"
        return format_string
