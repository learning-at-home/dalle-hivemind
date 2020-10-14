import os
from dataclasses import asdict
from pathlib import Path

import hivemind
import transformers
from hivemind import SizeAdaptiveCompression, Float16Compression, Uniform8BitQuantization
from transformers import AlbertTokenizerFast, get_linear_schedule_with_warmup, DataCollatorForLanguageModeling

import utils
from arguments import HFTrainerArguments, BasePeerArguments, CollaborativeArguments
from data import make_dataset
from huggingface_auth import authorize_with_huggingface
from lib import LeanAlbertConfig, LeanAlbertForPreTraining
from lib.staging.collaborative import CollaborativeOptimizer
from lib.training.clipped_lamb import LambWithGradientClipping
from lib.training.offload import OffloadOptimizer

hivemind.use_hivemind_log_handler("in_root_logger")
logger = hivemind.get_logger()


class TrainingTask:
    """A container that defines the training config, model, tokenizer, optimizer and other local training utilities"""
    _dht = _collaborative_optimizer = _training_dataset = None


    def __init__(
            self, peer_args: BasePeerArguments, trainer_args: HFTrainerArguments, collab_args: CollaborativeArguments):
        self.peer_args, self.trainer_args, self.collab_args = peer_args, trainer_args, collab_args
        self.validators, self.local_public_key = utils.make_validators(self.peer_args.experiment_prefix)
        transformers.set_seed(trainer_args.seed)  # seed used for initialization
        self.config = LeanAlbertConfig.from_pretrained(peer_args.model_config_path)
        self.tokenizer = AlbertTokenizerFast.from_pretrained(peer_args.tokenizer_path, cache_dir=peer_args.cache_dir)

        output_dir = Path(trainer_args.output_dir)
        logger.info(f'Checkpoint dir {output_dir}, contents {list(output_dir.glob("checkpoint*"))}')
        latest_checkpoint_dir = max(output_dir.glob("checkpoint*"), default=None, key=os.path.getctime)

        if latest_checkpoint_dir is None:
            logger.info(f"Creating model")
            self.model = LeanAlbertForPreTraining(self.config)
            self.model.resize_token_embeddings(len(self.tokenizer))
        else:
            logger.info(f"Loading model from {latest_checkpoint_dir}")
            self.model = LeanAlbertForPreTraining.from_pretrained(latest_checkpoint_dir)

    @property
    def dht(self):
        if self._dht is None:
            self._dht = hivemind.DHT(
                start=True,
                initial_peers=self.peer_args.initial_peers,
                client_mode=self.peer_args.client_mode,
                host_maddrs=self.peer_args.host_maddrs,
                announce_maddrs=self.peer_args.announce_maddrs,
                use_ipfs=self.peer_args.use_ipfs,
                record_validators=self.validators,
                identity_path=self.peer_args.identity_path,
                authorizer=authorize_with_huggingface() if self.peer_args.authorize else None,
            )
            if self.peer_args.client_mode:
                logger.info(f"Created client mode peer with peer_id={self._dht.peer_id}")
            else:
                utils.log_visible_maddrs(self._dht.get_visible_maddrs(), only_p2p=self.peer_args.use_ipfs)
        return self._dht

    @property
    def collaborative_optimizer(self):
        if self._collaborative_optimizer is None:
            opt, scheduler = self._get_local_optimizer_and_scheduler(self.trainer_args)
            averaging_compression = SizeAdaptiveCompression(
                threshold=2 ** 16 + 1, less=Float16Compression(), greater_equal=Uniform8BitQuantization())
            state_compression = hivemind.Float16Compression()
            self._collaborative_optimizer = CollaborativeOptimizer(
                dht=self.dht, opt=opt, scheduler=scheduler, prefix=self.peer_args.experiment_prefix,
                batch_size_per_step=self.trainer_args.batch_size_per_step,
                compression=averaging_compression, state_compression=state_compression,
                client_mode=self.peer_args.client_mode, verbose=True, start=True, **asdict(self.collab_args))
        return self._collaborative_optimizer

    def _get_local_optimizer_and_scheduler(self, training_args: HFTrainerArguments):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": training_args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        opt = OffloadOptimizer(
            optimizer_grouped_parameters,
            optim_cls=LambWithGradientClipping,
            lr=training_args.learning_rate,
            betas=(training_args.adam_beta1, training_args.adam_beta2),
            eps=training_args.adam_epsilon,
            weight_decay=training_args.weight_decay,
            max_grad_norm=training_args.max_grad_norm,
            clamp_value=training_args.clamp_value,
            debias=True,
        )

        scheduler = get_linear_schedule_with_warmup(
            opt, num_warmup_steps=training_args.warmup_steps, num_training_steps=training_args.total_steps
        )

        return opt, scheduler

    @property
    def training_dataset(self):
        if self._training_dataset is None:
            self._training_dataset = make_dataset(
                self.tokenizer, shuffle_seed=hash(self.local_public_key) % 2 ** 31,
                max_sequence_length=self.trainer_args.seq_length
            )
        return self._training_dataset

    @property
    def data_collator(self):
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, pad_to_multiple_of=self.trainer_args.pad_to_multiple_of
        )
