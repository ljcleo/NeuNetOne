from torchinfo import summary

from src.model import ResNeXt

if __name__ == '__main__':
    batch_size: int = 256
    model: ResNeXt = ResNeXt(3, 100, [(3, 8, 64, 256), (3, 8, 128, 512), (3, 8, 256, 1024)])
    summary(model, (batch_size, 3, 32, 32), device='cpu')
    model: ResNeXt = ResNeXt(3, 100, [(3, 32, 4, 256), (3, 32, 8, 512), (3, 32, 16, 1024)])
    summary(model, (batch_size, 3, 32, 32), device='cpu')
