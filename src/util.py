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


def sort_gpu() -> None:
    output: str = run('nvidia-smi -q -d Memory | grep -A5 GPU | grep Free', capture_output=True,
                      shell=True, encoding='utf8').stdout

    environ.setdefault('CUDA_VISIBLE_DEVICES', ','.join(map(str, torch.argsort(torch.LongTensor(
        [int(line.strip().split()[2]) for line in output.strip().split('\n')]
    ), descending=True).tolist())))


def init_device(gpu: bool) -> torch.device:
    sort_gpu()
    return torch.device('cuda' if gpu and cuda.is_available() else 'cpu')


def fix_seed(fix_seed: int) -> None:
    seed(fix_seed)
    np.random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    cuda.manual_seed(fix_seed)
    cudnn.deterministic = True


def get_path(root_path: Path, sub_dir: str, name: Optional[str] = None) -> Path:
    result: Path = root_path / sub_dir
    if name is not None:
        result = result / name.split('-')[0]

    result.mkdir(parents=True, exist_ok=True)
    return result
