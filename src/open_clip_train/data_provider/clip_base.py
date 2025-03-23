import numpy as np
import torchvision.transforms as transforms

from dataclasses import dataclass
from typing import Optional, Any, Callable, Union, Tuple
from torch.utils.data import DataLoader, Dataset, Subset, Sampler
from .sampler import DistributedRangedSampler
from .dc_base import BaseDataProvider, BaseDataProviderConfig

__all__ = ["CLIPCoreDataProviderConfig", "CLIPCoreDataProvider"]


@dataclass
class CLIPCoreDataProviderConfig(BaseDataProviderConfig):
    resolution: Any = 256 # int | tuple[int]
    size_transform: str = "ResizeCentorCrop"
    mean: Any = (0.5, 0.5, 0.5) # float | tuple[float, float, float]
    std: Any = (0.5, 0.5, 0.5) # float | tuple[float, float, float]
    
    shuffle_chunk_size: Optional[int] = None


class CLIPCoreDataProvider(BaseDataProvider):
    def __init__(self, cfg: CLIPCoreDataProviderConfig):
        super().__init__(cfg)
        self.cfg : CLIPCoreDataProviderConfig

    def build_transform(self) -> Tuple[Callable, Callable]:
        if self.cfg.size_transform == "ResizeCentorCrop":
            size_transform = transforms.Compose([
                transforms.Resize(self.cfg.resolution),
                transforms.CenterCrop(self.cfg.resolution)
            ])
        else:
            raise ValueError(f"size transform {self.cfg.size_transform} is not supported")
        transforms_list = [
            transforms.ToTensor(),
            transforms.Normalize(self.cfg.mean, self.cfg.std),
        ]
        return size_transform, transforms.Compose(transforms_list)

    def build_sampler(self) -> Sampler:
        sampler = DistributedRangedSampler(
            dataset=self.dataset,
            num_replicas=self.dist_size,
            rank=self.rank,
            shuffle=self.cfg.shuffle,
            shuffle_chunk_size=self.cfg.shuffle_chunk_size,
            seed=self.cfg.seed
        )
        return sampler

    def build_filtered_dataset(self, complete_dataset: Dataset, mask: bool | np.ndarray) -> Dataset:
        if mask == True:
            return complete_dataset
        else:
            raise NotImplementedError