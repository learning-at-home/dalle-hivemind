import os.path
from typing import Any

import hivemind
import torch
import transformers
from transformers import TrainingArguments

from arguments import TrainingPeerArguments
from task import TrainingTask
from utils import LocalMetrics, logger


class CollaborativeCallback(transformers.TrainerCallback):
    """
    This callback monitors and reports collaborative training progress,
    In case of a catastrophic failure, it can also revert training to a backup
    """

    def __init__(self, task: TrainingTask, args: TrainingPeerArguments):
        super().__init__()
        self.task = task
        self.dht, self.collaborative_optimizer = task.dht, task.collaborative_optimizer
        self.statistics_expiration = args.statistics_expiration
        self.last_reported_collaboration_step = -1
        self.samples = 0
        self.steps = 0
        self.loss = 0
        self.total_samples_processed = 0
        self.backup_every_steps = args.backup_every_steps
        self.state_path = args.state_path

    def on_train_begin(
        self, args: TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs
    ):
        if os.path.isfile(self.state_path):
            self.restore_from_backup(self.state_path)
            logger.info("Loaded state")

        logger.info("Loading state from peers")
        self.collaborative_optimizer.load_state_from_peers()

        if os.path.isfile(self.state_path):
            self.restore_from_backup(self.state_path, check_step=True)

    def on_step_end(
        self, args: TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs
    ):
        control.should_log = True
        if not self.params_are_finite():
            if not os.path.exists(self.state_path):
                raise RuntimeError("Encountered broken parameters, but there is no backup to fall back to.")
            logger.warning("Parameters are invalid, reloading model from earlier state")
            self.restore_from_backup(self.state_path)
            return control

        if state.log_history:
            self.loss += state.log_history[-1]["loss"]
            self.steps += 1
            if self.collaborative_optimizer.local_step != self.last_reported_collaboration_step:
                self.last_reported_collaboration_step = self.collaborative_optimizer.local_step
                self.total_samples_processed += self.samples
                samples_per_second = self.collaborative_optimizer.performance_ema.samples_per_second
                statistics = LocalMetrics(
                    step=self.collaborative_optimizer.local_step,
                    samples_per_second=samples_per_second,
                    samples_accumulated=self.samples,
                    loss=self.loss,
                    mini_steps=self.steps,
                )
                logger.info(f"Step {self.collaborative_optimizer.local_step}")
                logger.info(f"Your current contribution: {self.total_samples_processed} samples")
                logger.info(f"Performance: {samples_per_second} samples per second.")
                if self.steps:
                    logger.info(f"Local loss: {self.loss / self.steps}")

                self.loss = 0
                self.steps = 0
                if self.collaborative_optimizer.is_synchronized:
                    self.dht.store(
                        key=self.collaborative_optimizer.prefix + "_metrics",
                        subkey=self.task.local_public_key,
                        value=statistics.dict(),
                        expiration_time=hivemind.get_dht_time() + self.statistics_expiration,
                        return_future=True,
                    )
                if self.backup_every_steps is not None and \
                        self.collaborative_optimizer.local_step % self.backup_every_steps == 0:
                    self.backup_state()

        self.samples = self.collaborative_optimizer.local_samples_accumulated

        return control

    @torch.no_grad()
    def params_are_finite(self):
        for param in self.task.model.parameters():
            if not torch.all(torch.isfinite(param)):
                return False
        return True

    @torch.no_grad()
    def backup_state(self) -> Any:
        logger.info("Saving backup")
        return torch.save(
            {
                "model": self.task.model.state_dict(),
                "training": self.collaborative_optimizer.state_dict(),
                "scheduler": self.collaborative_optimizer.scheduler.state_dict(),
                "local_step": self.collaborative_optimizer.local_step,
            },
            self.state_path,
        )

    @torch.no_grad()
    def restore_from_backup(self, path, check_step=False):
        state = torch.load(path)
        current_step = self.collaborative_optimizer.local_step
        backup_step = state['training']['state'][0]['step'] #TODO FIX THIS, use state['local_step']
        if not check_step or backup_step >= current_step:
            if (
                "albert.encoder.albert_layer_groups.0.albert_layers.0.attention.attention_core.rotary_emb.cos"
                in state["model"]
            ):
                del state["model"][
                    "albert.encoder.albert_layer_groups.0.albert_layers.0.attention.attention_core.rotary_emb.cos"
                ]
                del state["model"][
                    "albert.encoder.albert_layer_groups.0.albert_layers.0.attention.attention_core.rotary_emb.sin"
                ]
            if "scheduler" in state:
                self.collaborative_optimizer.scheduler.load_state_dict(state["scheduler"])
            self.collaborative_optimizer.load_state_dict(state["training"])
            self.collaborative_optimizer.averager.local_step = backup_step
            self.task.model.load_state_dict(state["model"], strict=False)
            logger.info("Restored from a backup")
        else:
            logger.info("Bypassed restoring state from local backup: backup state is too old.")
