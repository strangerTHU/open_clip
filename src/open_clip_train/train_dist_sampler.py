from typing import Iterator, Optional, TypeVar

import numpy as np
import torch.utils.data
from torch.distributed import DeviceMesh
from torch.utils.data import DistributedSampler

__all__ = ["TrainDistSampler"]

T_co = TypeVar("T_co", covariant=True)


class TrainDistSampler(DistributedSampler):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
        device_mesh: Optional[DeviceMesh] = None,
    ) -> None:
        super().__init__(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

        self.epoch = 0
        self.iteration = 0

        self.sp_degree = device_mesh["sp"].size() if device_mesh is not None else 1

        # Consider sequence parallelism
        if self.sp_degree > 1:
            self.dp_rank = device_mesh.get_local_rank(mesh_dim="dp")
            self.dp_num_replicas = num_replicas // self.sp_degree
            self.corresponding_ranks = list(range(self.dp_rank * self.sp_degree, (self.dp_rank + 1) * self.sp_degree))
        else:
            self.dp_rank = rank
            self.dp_num_replicas = num_replicas

        # indices dtype
        if len(self.dataset) - 1 <= np.iinfo(np.uint16).max:
            self.idx_dtype = np.uint16
        elif len(self.dataset) - 1 <= np.iinfo(np.uint32).max:
            self.idx_dtype = np.uint32
        else:
            raise NotImplementedError(f"dataset size ({len(dataset)}) exceeds the maximum index range")

    def __len__(self) -> int:
        return self.sp_degree * (self.num_samples - self.iteration)

    def set_epoch(self, epoch: int, iteration: Optional[int] = None) -> None:
        self.epoch = epoch
        self.iteration = iteration if iteration is not None else 0

    def __iter__(self) -> Iterator[T_co]:
        indices = np.arange(len(self.dataset), dtype=self.idx_dtype)
        if self.shuffle:
            # Deterministically shuffle based on epoch and seed
            # Torch built-in randomness is not very random, so we use numpy.
            rng = np.random.Generator(np.random.PCG64(seed=self.seed + self.epoch))
            rng.shuffle(indices)

        if not self.drop_last:
            # Add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            arrays_to_concatenate = [indices]
            while padding_size > 0:
                array_to_concatenate = indices[: min(padding_size, len(indices))]
                arrays_to_concatenate.append(array_to_concatenate)
                padding_size -= len(array_to_concatenate)
                del array_to_concatenate
            indices = np.concatenate(arrays_to_concatenate)
        else:
            # Remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # Start at the specified index.
        if self.iteration > 0:
            indices = indices[self.iteration * self.num_replicas :]

        # Sequence Parallelism is enabled, to ensure the same behavior as data parallelism
        if self.sp_degree > 1:
            indices = indices.reshape(-1, self.dp_num_replicas * self.sp_degree)

            # Select the indices for this group
            start_idx = self.dp_rank * self.sp_degree
            end_idx = start_idx + self.sp_degree
            indices = indices[:, start_idx:end_idx].flatten()
        else:
            # Slice indices by rank to avoid duplicates.
            indices = indices[self.rank : len(indices) : self.num_replicas]
        assert len(indices) == len(self)

        return iter(indices)
