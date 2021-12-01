"""A catch-all module for the dirty hacks required to make HF Trainer work with collaborative training"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers.trainer import Trainer
from hivemind import CollaborativeOptimizer
from hivemind.optim import HivemindGradScaler
from hivemind.utils.logging import get_logger, use_hivemind_log_handler

use_hivemind_log_handler("in_root_logger")
logger = get_logger()
LRSchedulerBase = getattr(torch.optim.lr_scheduler, '_LRScheduler', None)


class CollaborativeHFTrainer(Trainer):
    """
    A version of HuggingFace trainer that shuffles the dataset using a separate random seed.
    Used to ensure that peers don't process batches in the same order.
    """

    def __init__(self, *, data_seed: int, collaborative_optimizer: CollaborativeOptimizer, **kwargs):
        self.data_seed = data_seed
        self.collaborative_optimizer = collaborative_optimizer
        super().__init__(optimizers=(collaborative_optimizer, NoOpScheduler(collaborative_optimizer)), **kwargs)

        if self.fp16_backend is not None:
            assert self.use_amp
            self.scaler = HivemindGradScaler()

    def get_train_dataloader(self) -> DataLoader:
        """Shuffle data independently for each peer to avoid duplicating batches [important for quality]"""
        torch.manual_seed(self.data_seed)
        return super().get_train_dataloader()

    def _wrap_model(self, model, training=True):
        # if reuse_grad_buffers is True, we should accumulate gradients in .grad without zeroing them after each step
        return IgnoreGradManipulations(super()._wrap_model(model, training=training),
                                       override_zero_grad=self.collaborative_optimizer.grad_averager.reuse_grad_buffers)


class NoOpScheduler(LRSchedulerBase):
    """Dummy scheduler for transformers.Trainer. The real scheduler is defined in CollaborativeOptimizer.scheduler"""

    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

    def print_lr(self, *args, **kwargs):
        if self.optimizer.scheduler:
            return self.optimizer.scheduler.print_lr(*args, **kwargs)

    def step(self):
        logger.debug("Called NoOpScheduler.step")
        self._last_lr = self.get_lr()

    def state_dict(self):
        return {}

    def load_state_dict(self, *args, **kwargs):
        logger.debug("Called NoOpScheduler.load_state_dict")


class IgnoreGradManipulations(nn.Module):
    """ Wrapper for model that blocks gradient manipulations in huggingface Trainer (e.g. zero_grad, clip_grad) """
    def __init__(self, module, override_clipping: bool = True, override_zero_grad: bool = True):
        super().__init__()
        self.module = module
        self.override_clipping = override_clipping
        self.override_zero_grad = override_zero_grad

    def forward(self, *args, **kwargs):
        return self.module.forward(*args, **kwargs)

    def zero_grad(self, set_to_none: bool = False) -> None:
        if self.override_zero_grad and \
                all(param.grad.isfinite().all() for param in self.parameters() if param.requires_grad):
            logger.debug("Successfully bypassed zero_grad")
        else:
            self.module.zero_grad(set_to_none=set_to_none)

    def clip_grad_norm_(self, max_norm: float, norm_type: int = 2):
        """ ignore clip_grad_norm on each step, clip in optimizer instead """
        if self.override_clipping:
            logger.debug("Successfully bypassed clip_grad_norm_")
        else:
            return torch.nn.utils.clip_grad_norm_(self.module.parameters(), max_norm, norm_type=norm_type)
