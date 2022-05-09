from logging import FileHandler, Formatter, Logger, StreamHandler, getLogger
from pathlib import Path

from src.util import get_path


def make_logger(name: str, root_path: Path, console: bool, direct: bool = False) -> Logger:
    logger: Logger = getLogger(name)
    logger.setLevel('INFO')
    formatter: Formatter = Formatter('[%(asctime)s %(name)s] %(levelname)s: %(message)s')

    if console:
        console_handler: StreamHandler = StreamHandler()
        console_handler.setLevel('INFO')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    file_handler: FileHandler = FileHandler(
        get_path(root_path, 'log', None if direct else name) / f'{name}.log', 'w', encoding='utf8'
    )

    file_handler.setLevel('INFO')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
