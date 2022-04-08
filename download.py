from hashlib import new
from logging import Logger
from pathlib import Path
from shutil import move
from tarfile import open as taropen

from requests import get

from src.logger import make_logger
from src.util import get_path


def download(url: str, path: Path) -> bool:
    try:
        with get(url, stream=True) as response, taropen(fileobj=response.raw, mode='r|gz') as tar:
            tar.extractall(path)

        subpath: Path = path / 'cifar-100-python'
        move(subpath / 'train', path)
        move(subpath / 'test', path)
        move(subpath / 'meta', path)

        for subfile in subpath.iterdir():
            subfile.unlink()

        subpath.rmdir()
        return True
    except Exception:
        return False


def check_md5(path: Path, md5: str) -> bool:
    try:
        with path.open('rb') as f:
            return new('md5', f.read()).hexdigest() == md5
    except Exception:
        return False


def download_and_check(url: str, path: Path, md5: dict[str, str]) -> bool:
    if path.exists() and all([check_md5(path / name, file_md5) for name, file_md5 in md5.items()]):
        return True
    return download(url, path) and all([check_md5(path / name, file_md5) for name,
                                        file_md5 in md5.items()])


if __name__ == '__main__':
    root_path: Path = Path('.')
    data_path: Path = get_path(root_path, 'data')
    cifar_url: str = 'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'

    md5: dict[str, str] = {
        'train': '16019d7e3df5f24257cddd939b257f8d',
        'test': 'f0ef6b0ae62326f3e7ffdfab6717acfc',
        'meta': '7973b15100ade9c7d40fb424638fde48'
    }

    logger: Logger = make_logger('download', root_path, True, direct=True)
    logger.info('Start downloading CIFAR-100 dataset ...')

    for i in range(3):
        logger.info(f'{"Retry downloading" if i > 0 else "Downloading"} from {cifar_url}')

        if download_and_check(cifar_url, data_path, md5):
            logger.info(f'Finished writing data to {data_path}.')
            break
    else:
        logger.error(f'Failed to download from {cifar_url} to {data_path}!')
        raise RuntimeError(f'cannot download from {cifar_url} to {data_path}')

    logger.info('Finished downloading CIFAR-100 dataset.')
