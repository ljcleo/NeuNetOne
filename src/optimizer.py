import torch
from torch.optim import Optimizer
from torch.optim.sgd import SGD

from src.util import FTType


class ASAM(Optimizer):
    def __init__(self, params, lr: float, rho: float = 1, eta: float = 0.01,
                 weight_decay: float = 0.0005, **kwargs) -> None:
        super().__init__(params, dict(rho=rho, eta=eta, **kwargs))
        self.base_optimizer: SGD = SGD(self.param_groups, lr=lr, weight_decay=weight_decay,
                                       **kwargs)
        self.param_groups: list[dict] = self.base_optimizer.param_groups

    @torch.no_grad()
    def step_1(self, zero_grad: bool = False) -> None:
        norm: FTType = self._t_grad_norm()

        for group in self.param_groups:
            scale: FTType = group['rho'] / (norm + 1e-8)
            param: FTType

            for param in group['params']:
                if param.grad is None:
                    continue

                eps: FTType = self._apply_t2(param, param.grad) * scale.to(param)
                param.add_(eps)
                self.state[param]['eps'] = eps

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step_2(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            param: FTType

            for param in group['params']:
                if param.grad is not None:
                    param.sub_(self.state[param]['eps'])

        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _t_grad_norm(self) -> FTType:
        device: torch.device = self.param_groups[0]['params'][0].device

        return torch.norm(torch.stack([
            self._apply_t(param, param.grad).norm(p=2).to(device) for group in self.param_groups
            for param in group['params'] if param.grad is not None
        ]), p=2)

    def _apply_t(self, param: FTType, x: FTType) -> FTType:
        return (torch.abs(param) + self.param_groups[0]['eta']) * x

    def _apply_t2(self, param: FTType, x: FTType) -> FTType:
        op: FTType = torch.abs(param) + self.param_groups[0]['eta']
        return op * op * x
