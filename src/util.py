from os import environ
from pathlib import Path
from random import seed
from subprocess import run
from typing import Any, Optional

import numpy as np
import torch
from torch import cuda
from torch.backends import cudnn
from yaml import safe_load

FTType = torch.FloatTensor
LTType = torch.LongTensor
BatchType = tuple[FTType, FTType]


def sort_gpu() -> None:
    output: str = run('nvidia-smi -q -d Memory | grep -A5 GPU | grep Free', capture_output=True,
                      shell=True, encoding='utf8').stdout

    device_index: list[int] = torch.argsort(torch.LongTensor(
        [int(line.strip().split()[2]) for line in output.strip().split('\n')]
    ), descending=True).tolist()

    environ.setdefault('CUDA_VISIBLE_DEVICES', ','.join(map(str, device_index)))


def fix_seed(seed_num: int, gpu: bool) -> None:
    seed(seed_num)
    environ['PYTHONHASHSEED'] = str(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)

    if gpu:
        cuda.manual_seed_all(seed_num)
        cudnn.benchmark = False
        cudnn.deterministic = True


def init_device(seed_num: int | None = None, gpu: bool = True) -> torch.device:
    gpu &= cuda.is_available()
    if seed_num:
        fix_seed(seed_num, gpu)

    if gpu:
        sort_gpu()
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def get_path(root_path: Path, sub_dir: str, name: Optional[str] = None) -> Path:
    result: Path = root_path / sub_dir
    if name is not None:
        result = result / name.split('-')[0]

    result.mkdir(parents=True, exist_ok=True)
    return result


def get_config(root_path: Path, name: str) -> dict[str, Any]:
    with (get_path(root_path, 'config') / f'{name}.yaml').open('r', encoding='utf8') as f:
        return safe_load(f)


def check_inf_nan(x: FTType) -> tuple[bool, bool, bool]:
    return x.isposinf().any().item(), x.isneginf().any().item(), x.isnan().any().item()
