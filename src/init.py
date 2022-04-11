from os import environ
from random import seed
from subprocess import run
from typing import Optional

import numpy as np
import torch
from torch import cuda
from torch.backends import cudnn


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


def init_device(seed_num: Optional[int] = None, gpu: bool = True) -> torch.device:
    gpu &= cuda.is_available()
    if seed_num:
        fix_seed(seed_num, gpu)

    if gpu:
        sort_gpu()
        return torch.device('cuda')
    else:
        return torch.device('cpu')
