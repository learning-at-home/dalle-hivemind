import itertools
from typing import Optional

import hivemind
import numpy as np
from datasets import load_dataset

logger = hivemind.get_logger(__name__)


def preprocess_batch(batch, tokenizer, max_sequence_length: int):
    mask = [
        (
            caption is not None and len(caption) >= 3 and
            nsfw == 'UNLIKELY' and
            orig_width > 0 and orig_height > 0 and
            max(orig_height / orig_width, orig_width / orig_height) <= 2
        ) for caption, nsfw, orig_width, orig_height in
        zip(batch['caption'], batch['NSFW'], batch['original_width'], batch['original_height'])
    ]
    logger.debug(f'{np.mean(mask) * 100:.1f}% of examples left after filtering')

    if any(mask):
        result = tokenizer(list(itertools.compress(batch['caption'], mask)),
                           add_special_tokens=False, max_length=max_sequence_length, truncation=True)
    else:
        # This branch is necessary because tokenizer([]) raises IndexError
        result = {'input_ids': [], 'attention_mask': []}
    result['image'] = [np.frombuffer(encoded, np.int16).astype(np.int64)
                       for encoded in itertools.compress(batch['code'], mask)]
    return result


def make_dataset(
    tokenizer,
    *,
    shuffle_buffer_size: int = 8192,
    shuffle_seed: Optional[int],
    preprocessing_batch_size: int = 256,
    max_sequence_length: int,
):
    ds = load_dataset('laion/laion_100m_vqgan_f8', split='train', streaming=True)
    ds = ds.shuffle(shuffle_buffer_size, seed=shuffle_seed)
    ds = ds.map(lambda batch: preprocess_batch(batch, tokenizer, max_sequence_length),
                batched=True, batch_size=preprocessing_batch_size)
    ds = ds.with_format('torch')
    return ds
