import os
from datetime import timedelta
from typing import Optional, TypeVar

T = TypeVar("T")

import torch
import torch.distributed

__all__ = [
    "dist_init",
    "is_dist_initialized",
    "get_dist_rank",
    "get_dist_size",
    "is_master",
    "dist_barrier",
    "get_dist_local_rank",
    "sync_tensor",
]


def dist_init(timeout: Optional[timedelta] = None) -> None:
    if is_dist_initialized():
        return
    try:
        torch.distributed.init_process_group(
            backend="nccl", timeout=timeout, device_id=torch.device(f"cuda:{get_dist_local_rank()}")
        )
        assert torch.distributed.is_initialized()
    except Exception:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        print("warning: dist not init")


def is_dist_initialized() -> bool:
    return torch.distributed.is_initialized()


def get_dist_rank() -> int:
    return int(os.environ["RANK"])


def get_dist_size() -> int:
    return int(os.environ["WORLD_SIZE"])


def is_master() -> bool:
    return get_dist_rank() == 0


def dist_barrier() -> None:
    if is_dist_initialized():
        torch.distributed.barrier()


def get_dist_local_rank() -> int:
    return int(os.environ["LOCAL_RANK"])

def sync_object(obj: T) -> list[T]:
    if not is_dist_initialized():
        return [obj]
    obj_list = [None for _ in range(get_dist_size())]
    torch.distributed.all_gather_object(obj_list, obj)
    return obj_list


def broadcast_object(obj: T, src: int = 0) -> T:
    if not is_dist_initialized():
        return obj
    obj_list = [obj]
    torch.distributed.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def gather_object(obj: T, dst: int = 0) -> Optional[list[T]]:
    if not is_dist_initialized():
        return [obj]
    obj_list = [None for _ in range(get_dist_size())] if get_dist_rank() == dst else None
    torch.distributed.gather_object(obj, obj_list, dst=dst)
    return obj_list


def destroy_process_group():
    if is_dist_initialized():
        torch.distributed.destroy_process_group()
