import torch
from torch_optimizer import Lamb


class LambWithGradientClipping(Lamb):
    """ A version of LAMB that clips gradients based on their norm. """
    def __init__(self, *args, max_grad_norm: float, **kwargs):
        self.max_grad_norm = max_grad_norm
        super().__init__(*args, **kwargs)

    def step(self, *args, **kwargs):
        iter_params = (param for group in self.param_groups for param in group['params'])
        torch.nn.utils.clip_grad_norm_(iter_params, self.max_grad_norm)
        return super().step(*args, **kwargs)
