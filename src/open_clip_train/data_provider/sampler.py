from typing import Optional

import torch
from torch.utils.data import Dataset, Sampler

__all__ = ["DistributedRangedSampler"]


class DistributedRangedSampler(Sampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        num_samples: Optional[int] = None,
        shuffle_chunk_size: Optional[int] = None,
    ):
        assert rank >= 0 and rank < num_replicas
        self.num_samples = num_samples if num_samples is not None else len(dataset)
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        if drop_last:
            self.num_samples_per_rank = self.num_samples // num_replicas
        else:
            self.num_samples_per_rank = (self.num_samples - 1) // num_replicas + 1
        self.shuffle_chunk_size = shuffle_chunk_size
        if shuffle_chunk_size is not None:
            start = self.rank * self.num_samples_per_rank
            end = (self.rank + 1) * self.num_samples_per_rank
            self.ranges = [(i, min(i + shuffle_chunk_size, end)) for i in range(start, end, shuffle_chunk_size)]
        self.epoch = 0
        self.iter_index = 0

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.iter_index = 0

    def set_iter_index(self, iter_index):
        self.iter_index = iter_index

    def __len__(self):
        return self.num_samples_per_rank

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.num_samples, generator=g).tolist()
            if not self.drop_last:
                total_size = self.num_replicas * self.num_samples_per_rank
                padding_size = total_size - len(indices)
                indices += (indices * ((padding_size - 1) // len(indices) + 1))[:padding_size]
            indices = indices[self.rank * self.num_samples_per_rank : (self.rank + 1) * self.num_samples_per_rank]
            assert len(indices) == self.num_samples_per_rank
            yield from indices[self.iter_index :]
        elif self.shuffle_chunk_size is not None:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            shard_indices = torch.randperm(len(self.ranges), generator=g).tolist()
            cnt = 0
            for shard_index in shard_indices:
                shard_start, shard_end = self.ranges[shard_index]
                num_shard_samples = shard_end - shard_start
                sample_indices = (
                    (torch.randperm(num_shard_samples, generator=g) + shard_start) % self.num_samples
                ).tolist()
                if cnt + num_shard_samples <= self.iter_index:
                    cnt += num_shard_samples
                    continue
                yield from sample_indices[max(self.iter_index - cnt, 0) :]
                cnt += num_shard_samples
        else:
            start = self.rank * self.num_samples_per_rank + self.iter_index
            end = (self.rank + 1) * self.num_samples_per_rank
            indices = (torch.arange(self.num_replicas * self.num_samples_per_rank) % self.num_samples).tolist()
            yield from indices[start:end]


class MultiResolutionDistributedRangedSampler(DistributedRangedSampler):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        resolution: int | list[int],
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        num_samples: Optional[int] = None,
        shuffle_chunk_size: Optional[int] = None,
    ):
        super().__init__(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
            num_samples=num_samples,
            shuffle_chunk_size=shuffle_chunk_size,
        )
        self.batch_size = batch_size
        resolution_list = [resolution] if isinstance(resolution, int) else resolution
        self.resolution_list = torch.tensor(resolution_list, dtype=int)

    def __iter__(self):
        indices = list(super().__iter__())
        g = torch.Generator()
        g.manual_seed(self.epoch)
        resolution_indices = torch.randint(
            0, len(self.resolution_list), size=((self.num_samples_per_rank - 1) // self.batch_size + 1,), generator=g
        ).repeat_interleave(self.batch_size)[: self.num_samples_per_rank]
        resolutions = self.resolution_list[resolution_indices].tolist()
        seeds = torch.randint(0, 2**63 - 1, (self.num_samples_per_rank,), generator=g).tolist()
        return iter(zip(indices, resolutions[self.iter_index :], seeds[self.iter_index :]))
