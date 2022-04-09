from os import environ
from pathlib import Path
from random import seed
from subprocess import run
from typing import Optional

import numpy as np
import torch
from torch import cuda
from torch.backends import cudnn

FTType = torch.FloatTensor
LTType = torch.LongTensor
BatchType = tuple[FTType, FTType]


def sort_gpu() -> list[int]:
    output: str = run('nvidia-smi -q -d Memory | grep -A5 GPU | grep Free', capture_output=True,
                      shell=True, encoding='utf8').stdout

    device_index: list[int] = torch.argsort(torch.LongTensor(
        [int(line.strip().split()[2]) for line in output.strip().split('\n')]
    ), descending=True).tolist()

    environ.setdefault('CUDA_VISIBLE_DEVICES', ','.join(map(str, device_index)))
    return device_index


def fix_seed(seed_num: int, gpu: bool) -> None:
    seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    cuda.manual_seed(seed_num)

    if gpu:
        cudnn.deterministic = True


def init_device(seed_num: int | None = None, gpu: bool = True) -> list[torch.device]:
    gpu &= cuda.is_available()
    if seed_num:
        fix_seed(seed_num, gpu)

    if gpu:
        device_index: list[int] = sort_gpu()
        return [torch.device(f'cuda:{i}') for i in device_index]
    else:
        return [torch.device('cpu')]


def get_path(root_path: Path, sub_dir: str, name: Optional[str] = None) -> Path:
    result: Path = root_path / sub_dir
    if name is not None:
        result = result / name.split('-')[0]

    result.mkdir(parents=True, exist_ok=True)
    return result
