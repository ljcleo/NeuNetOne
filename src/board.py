
from pathlib import Path

from torch.utils.tensorboard.writer import SummaryWriter

from src.util import get_path


def cleanup(path: Path) -> None:
    if path.exists():
        for file in path.iterdir():
            if file.is_dir():
                cleanup(file)
                file.rmdir()
            else:
                file.unlink()


def make_tensorboard(name: str, root_path: Path) -> SummaryWriter:
    writer_path: Path = get_path(root_path, 'log', name) / name
    cleanup(writer_path)
    return SummaryWriter(writer_path)
