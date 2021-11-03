#!/usr/bin/env python

import os
from pathlib import Path

import transformers
from transformers import HfArgumentParser
from hivemind.utils.logging import get_logger, use_hivemind_log_handler

from lib.training.hf_trainer import CollaborativeHFTrainer

import callback
import utils
from arguments import TrainingPeerArguments, HFTrainerArguments, CollaborativeArguments
from task import TrainingTask


transformers.utils.logging.set_verbosity_warning()
use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)


def main():
    parser = HfArgumentParser((TrainingPeerArguments, HFTrainerArguments, CollaborativeArguments))
    training_peer_args, trainer_args, collab_args = parser.parse_args_into_dataclasses()

    logger.info(f"Trying {len(training_peer_args.initial_peers)} initial peers: {training_peer_args.initial_peers}")
    # if len(training_peer_args.initial_peers) == 0:
    #     logger.warning("Please specify at least one network endpoint in initial peers.")

    utils.log_process_rank(trainer_args)
    task = TrainingTask(training_peer_args, trainer_args, collab_args)
    model = task.model.to(trainer_args.device)

    collaborative_callback = callback.CollaborativeCallback(task, training_peer_args)
    assert trainer_args.do_train and not trainer_args.do_eval

    # Note: the code below creates the trainer with dummy scheduler and removes some callbacks.
    # This is done because collaborative training has its own callbacks that take other peers into account.
    trainer = CollaborativeHFTrainer(
        model=model,
        args=trainer_args,
        tokenizer=task.tokenizer,
        data_collator=task.data_collator,
        data_seed=hash(task.local_public_key),
        train_dataset=task.training_dataset,
        eval_dataset=None,
        collaborative_optimizer=task.collaborative_optimizer,
        callbacks=[collaborative_callback],
    )
    trainer.remove_callback(transformers.trainer_callback.PrinterCallback)
    trainer.remove_callback(transformers.trainer_callback.ProgressCallback)

    latest_checkpoint_dir = max(Path(trainer_args.output_dir).glob("checkpoint*"), key=os.path.getctime, default=None)
    trainer.train(model_path=latest_checkpoint_dir)


if __name__ == "__main__":
    main()
