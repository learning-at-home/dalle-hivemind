from typing import Optional

import hivemind
import numpy as np
from datasets import load_dataset

logger = hivemind.get_logger(__name__)


def make_dataset(
    tokenizer,
    *,
    shuffle_buffer_size: int = 10 ** 4,
    shuffle_seed: Optional[int],
    preprocessing_batch_size: int = 256,
    max_sequence_length: int,
):
    ds = load_dataset('laion/laion_100m_vqgan_f8', split='train', streaming=True)
    ds = ds.shuffle(shuffle_buffer_size, seed=shuffle_seed)
    ds = ds.map(lambda item: dict(
        tokenizer(item['caption'], truncation=True, max_length=max_sequence_length),
        image=np.stack([np.frombuffer(encoded, np.int16).astype(np.int64) for encoded in item['code']]),
    ), batched=True, batch_size=preprocessing_batch_size)
    ds = ds.with_format('torch')
    return ds
