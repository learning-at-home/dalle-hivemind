#!/usr/bin/env python3
import time

import torch
import wandb
import transformers
from transformers import HfArgumentParser
from huggingface_hub import HfFolder, Repository
from hivemind.utils.logging import get_logger, use_hivemind_log_handler

import utils
from arguments import AuxiliaryPeerArguments, CollaborativeArguments, HFTrainerArguments
from task import TrainingTask


transformers.utils.logging.set_verbosity_warning()
use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)


class CheckpointHandler:
    def __init__(self, task: TrainingTask, peer_args: AuxiliaryPeerArguments):
        self.task, self.peer_args = task, peer_args
        self.save_checkpoint_step_interval = peer_args.save_checkpoint_step_interval
        self.prefix = peer_args.experiment_prefix
        self.local_path = peer_args.local_path
        self.upload_interval = peer_args.upload_interval
        if self.upload_interval is not None:
            assert task.authorizer is not None, 'Model uploading needs Hugging Face auth to be enabled'
            self.repo = Repository(
                local_dir=self.local_path,
                clone_from=peer_args.repo_url,
                use_auth_token=task.authorizer.hf_user_access_token,
            )
            self.last_upload_time = None
        self.previous_step = -1

    def should_save_state(self, cur_step):
        if self.save_checkpoint_step_interval is None:
            return False
        elif cur_step - self.previous_step >= self.save_checkpoint_step_interval:
            return True
        else:
            return False

    def save_state(self, cur_step):
        logger.info("Saving state from peers")
        self.task.collaborative_optimizer.load_state_from_peers()
        self.previous_step = cur_step

    def is_time_to_upload(self):
        if self.upload_interval is None:
            return False
        elif self.last_upload_time is None or time.time() - self.last_upload_time >= self.upload_interval:
            return True
        else:
            return False

    def upload_checkpoint(self, current_loss):
        self.last_upload_time = time.time()

        logger.info("Saving model")
        torch.save(self.task.model.state_dict(), f"{self.local_path}/model_state.pt")
        logger.info("Saving optimizer")
        torch.save(self.task.collaborative_optimizer.state_dict(), f"{self.local_path}/optimizer_state.pt")
        logger.info("Started uploading to Model Hub")
        try:
            # We start by pulling the remote changes (for example a change in the readme file)
            self.repo.git_pull()

            # Then we add / commmit and push the changes
            self.repo.push_to_hub(commit_message=f"Epoch {self.task.collaborative_optimizer.local_epoch}, loss {current_loss:.3f}")
            logger.info("Finished uploading to Model Hub")
        except Exception:
            logger.exception("Uploading the checkpoint to HF Model Hub failed:")
            logger.warning("Ensure that your access token is valid and has WRITE permissions")


def assist_averaging_in_background(task: TrainingTask, peer_args: AuxiliaryPeerArguments):
    while True:
        time.sleep(peer_args.assist_refresh)
        task.collaborative_optimizer.step()


if __name__ == "__main__":
    parser = HfArgumentParser((AuxiliaryPeerArguments, HFTrainerArguments, CollaborativeArguments))
    peer_args, trainer_args, collab_args = parser.parse_args_into_dataclasses()

    task = TrainingTask(peer_args, trainer_args, collab_args)
    dht, collaborative_optimizer = task.dht, task.collaborative_optimizer

    if peer_args.wandb_project is not None:
        wandb.init(project=peer_args.wandb_project)

    current_step = 0
    if peer_args.store_checkpoints:
        checkpoint_handler = CheckpointHandler(task, peer_args)

    if peer_args.assist_in_averaging:
        # assert not peer_args.client_mode, "client-mode peers cannot assist in averaging"
        # averaging_thread = threading.Thread(
        #     name="AveragingAuxThread", target=assist_averaging_in_background, args=[task, peer_args], daemon=True)
        # averaging_thread.start()
        raise NotImplementedError('aux peers with hivemind.optim.experimental are not supported yet')

    while True:
        metrics_entry = dht.get(peer_args.experiment_prefix + "_metrics", latest=True)
        if metrics_entry is not None and len(metrics_entry.value) > 0:
            metrics_dict = metrics_entry.value
            metrics = [utils.LocalMetrics.parse_obj(metrics_dict[peer].value) for peer in metrics_dict]
            latest_step = max(item.step for item in metrics)

            if latest_step != current_step:
                logger.debug(f"Got metrics from {len(metrics)} peers")

                for i, metrics_for_peer in enumerate(metrics):
                    logger.debug(f"{i} peer {metrics_for_peer}")

                current_step = latest_step
                alive_peers = 0
                sum_loss = 0
                num_samples = 0
                sum_perf = 0
                sum_mini_steps = 0

                for item in metrics:
                    sum_loss += item.loss
                    alive_peers += 1
                    sum_perf += item.samples_per_second
                    num_samples += item.samples_accumulated
                    sum_mini_steps += item.mini_steps
                current_loss = sum_loss / sum_mini_steps
                logger.info(f"Epoch #{current_step}\tloss = {current_loss:.5f}")

                if peer_args.wandb_project is not None:
                    wandb.log(
                        {
                            "loss": current_loss,
                            "alive peers": alive_peers,
                            "samples": num_samples,
                            "performance": sum_perf,
                            "step": latest_step,
                        }
                    )

                if peer_args.store_checkpoints:
                    if checkpoint_handler.should_save_state(current_step):
                        checkpoint_handler.save_state(current_step)
                        if checkpoint_handler.is_time_to_upload():
                            checkpoint_handler.upload_checkpoint(current_loss)
        logger.debug("Peer is still alive...")
        time.sleep(peer_args.refresh_period)
