import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Tuple, Callable, Optional, Any
from ..utils.image import convert_image_to_rgb
from .web_dataset.wids import WebDataset
from .clip_base import CLIPCoreDataProvider, CLIPCoreDataProviderConfig


@dataclass
class CoyoDataProviderConfig(CLIPCoreDataProviderConfig):
    name: str = "Coyo"
    data_dir: str = "~/dataset/coyo"
    wds_meta_path: Optional[str] = None
    metadata_path: Optional[str] = None


@dataclass
class CoyoTrainDataProviderConfig(CoyoDataProviderConfig):
    name: str = "CoyoTrain"
    shuffle_chunk_size: int = 1000
    drop_last: bool = True

class CoyoDataset(WebDataset):
    def __init__(
        self,
        data_dir: str,
        meta_path: str,
        size_transform: Callable,
        transform: Callable,
        convert_to_rgb: bool = True,
    ):
        super().__init__(data_dir, meta_path)
        self._size_transform = size_transform 
        self._transform = transform
        self.convert_to_rgb = convert_to_rgb

    def __getitem__(self, index: int | Tuple[int, int] | Tuple[int, int, int]) -> Tuple[torch.Tensor, Any]:
        sample = self.dataset[index]
        image = sample[".jpg"]
        if self.convert_to_rgb:
            image = convert_image_to_rgb(image)
        image = self._size_transform(image)
        image = self._transform(image)
        label = {"index": index, "caption": sample[".json"]["caption"]}
        return image, label


class CoyoDataProvider(CLIPCoreDataProvider):
    def __init__(self, cfg: CoyoDataProviderConfig):
        super().__init__(cfg)
        self.cfg: CoyoDataProviderConfig

    def build_complete_dataset(self) -> Dataset:
        size_transform, transform = self.build_transform()
        dataset = CoyoDataset(self.cfg.data_dir, self.cfg.wds_meta_path, size_transform, transform)
        return dataset


def debug():
    from omegaconf import OmegaConf
    import torch
    import numpy as np
    from tqdm import tqdm
    from torchvision.utils import save_image
    import json
    import ipdb

    from ..utils.dist import dist_init, get_dist_local_rank, dist_barrier, is_dist_initialized

    dist_init()
    if is_dist_initialized():
        torch.cuda.set_device(get_dist_local_rank())
    # ipdb.set_trace()
    cfg: CoyoTrainDataProviderConfig = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(CoyoTrainDataProviderConfig), OmegaConf.from_cli()))
    data_provider = CoyoDataProvider(cfg)

    image, label = data_provider.dataset[0]
    print(f"image {image.shape}, label {label}")
    # ipdb.set_trace()
    for i, (images, labels) in enumerate(data_provider.data_loader):
        dist_barrier()
        print(f"step {i}, rank {data_provider.rank}, shape {images.shape}", flush=True)
        save_image(images*0.5+0.5, f"exp_dongyun/clip/tmp/tmp_{i}_{data_provider.rank}.jpg", nrow=int(np.sqrt(cfg.batch_size)))
        if i == 5:
            break


if __name__ == "__main__":
    debug()

"""
python -m open_clip_train.data_provider.coyo data_dir=/dataset/coyo-700m_full_webdata/part_00000 wds_meta_path=coyo_part_00000.json

torchrun --nnodes=1 --nproc_per_node=8 -m efficientvit.clipcore.data_provider.coyo data_dir=/dataset/coyo-700m_full_webdata/part_00000 wds_meta_path=assets/data/meta/coyo_part_00000.json
"""