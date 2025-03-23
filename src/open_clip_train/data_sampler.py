from typing import Iterator, Optional, TypeVar

import numpy as np
import torch.utils.data
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
        return self.num_samples - self.iteration

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

        # Slice indices by rank to avoid duplicates.
        indices = indices[self.rank : len(indices) : self.num_replicas]
        assert len(indices) == len(self)

        return iter(indices)
