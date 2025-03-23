import getpass
import io
import json
import os
import random
import tarfile
import time
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple

from PIL import Image
from tqdm import tqdm


def generate_tar(data_format: Dict[str, str], data_list: List[Tuple[str, Dict[str, Any]]], tar_path: str) -> None:
    """
    data_format: `key` is suffix like ".png", `value` should be str from ["file_path", "data"], "file_path" means `data[key]` is a file path, "data" means `data[key]` is raw data.
    data_list: List of (prefix, data). For each data, `key` should be in `data_format`, `data[key]` should be either a file path or raw data.
    """
    tar_path = os.path.abspath(tar_path)
    tar = tarfile.open(tar_path, "w")
    for prefix, data in tqdm(data_list, desc=f"generate {tar_path}", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"):
        for key in data_format:
            if data_format[key] == "file_path":
                tar.add(data[key], arcname=prefix + key)
            elif data_format[key] == "data":
                if key == ".json":
                    fileobj = io.BytesIO(json.dumps(data[key]).encode("utf-8"))
                elif key == ".jpg":
                    if isinstance(data[key], Image.Image):
                        fileobj = io.BytesIO()
                        data[key].save(fileobj, format="JPEG")
                        fileobj.seek(0)
                    elif isinstance(data[key], io.BytesIO):
                        fileobj = data[key]
                    else:
                        raise ValueError(f"type {type(data[key])} is not supported for .jpg")
                else:
                    raise ValueError(f"{key} is not supported as raw data")
                tar_info = tarfile.TarInfo(name=prefix + key)
                tar_info.size = fileobj.getbuffer().nbytes
                tar_info.mtime = time.time()
                tar_info.uname = getpass.getuser()
                tar_info.gname = "dip"
                tar.addfile(tar_info, fileobj)
    tar.close()


def generate_all_tar(
    data_format: Dict[str, str],
    data_list: List[Tuple[str, Dict[str, Any]]],
    tar_dir: str,
    num_samples_per_tar: int,
    shuffle: bool,
    num_digits_in_tar_name: int,
    processes: int,
    seed: int = 0,
) -> None:
    """
    data_format: `key` is suffix like ".png", `value` should be str from ["file_path", "data"], "file_path" means `data[key]` is a file path, "data" means `data[key]` is raw data.
    data_list: List of (prefix, data). For each data, `key` should be in `data_format`, `data[key]` should be either a file path or raw data.
    """
    if shuffle:
        random.seed(seed)
        random.shuffle(data_list)
    os.makedirs(tar_dir, exist_ok=True)
    num_samples = len(data_list)

    if processes == 1:
        for tar_idx, start in enumerate(range(0, num_samples, num_samples_per_tar)):
            generate_tar(
                data_format,
                data_list[start : min(start + num_samples_per_tar, num_samples)],
                os.path.join(tar_dir, f"{tar_idx:0{num_digits_in_tar_name}d}.tar"),
            )
    else:
        pool = Pool(processes=processes)
        for tar_idx, start in enumerate(range(0, num_samples, num_samples_per_tar)):
            pool.apply_async(
                generate_tar,
                args=(
                    data_format,
                    data_list[start : min(start + num_samples_per_tar, num_samples)],
                    os.path.join(tar_dir, f"{tar_idx:0{num_digits_in_tar_name}d}.tar"),
                ),
            )
        pool.close()
        pool.join()


def debug():
    import json

    root_data_dir = "assets/data/extract_mj_v6/images/00/00"
    all_meta_path = "assets/data/extract_mj_v6/results/500000_000000.json"
    with open(all_meta_path, "r") as f:
        all_meta = json.load(f)

    data_format: Dict[str] = {".png": "file_path", ".json": "data"}
    data_list: List[Tuple[str, Dict[str, Any]]] = []
    for dir_path, dir_names, file_names in os.walk(root_data_dir):
        for file_name in file_names:
            if file_name.endswith(".png"):
                prefix = file_name.split(".")[0]
                data_list.append(
                    (
                        prefix,
                        {
                            ".png": os.path.join(dir_path, file_name),
                            ".json": all_meta[prefix],
                        },
                    )
                )

    generate_all_tar(data_format, data_list, "tmp", 1000, True, 8, 3)


if __name__ == "__main__":
    debug()

"""
python -m efficientvit.apps.utils.tar
"""
