import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from src.model import ResNeXt
from src.util import FTType


def evaluate_loss_acc(model: ResNeXt, loader: DataLoader) -> tuple[float, float]:
    with torch.no_grad():
        model.eval()
        all_pred: list[FTType] = []
        all_label: list[FTType] = []
        images: FTType
        labels: FTType

        for images, labels in loader:
            raw_pred: FTType = model.forward(images.flatten(end_dim=-4))
            all_pred.append(raw_pred.reshape(-1, labels.size(0), labels.size(1)).mean(dim=0))
            all_label.append(labels)

        cat_pred: FTType = torch.cat(all_pred)
        cat_label: FTType = torch.cat(all_label)
        loss: float = prob_ce(cat_pred, cat_label).item()
        acc: float = torch.sum(cat_pred.argmax(dim=1) ==
                               cat_label.argmax(dim=1)).item() / cat_label.size(0)

    return loss, acc


def prob_ce(pred: FTType, labels: FTType) -> FTType:
    return -torch.sum(F.log_softmax(pred, dim=-1) * labels, dim=-1).mean()
