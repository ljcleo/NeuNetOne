from pathlib import Path
from typing import Any, Optional

import torch
from yaml import safe_load

FTType = torch.FloatTensor
LTType = torch.LongTensor
BatchType = tuple[FTType, FTType]


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
