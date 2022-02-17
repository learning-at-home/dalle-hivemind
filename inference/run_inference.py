#!/usr/bin/env python3

import argparse
import os
import pickle
from collections import OrderedDict
from datetime import datetime
from itertools import cycle, islice

import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from einops import rearrange
# Note: Use dalle_pytorch >= 1.4.2 for this script (newer than in the rest of the repo)
from dalle_pytorch import DALLE
from dalle_pytorch.vae import VQGanVAE
from transformers import T5TokenizerFast
from tqdm import tqdm

torch.set_grad_enabled(False)


class VQGanParams(VQGanVAE):
    def __init__(self, *, num_layers=3, image_size=256, num_tokens=8192, is_gumbel=True):
        nn.Module.__init__(self)

        self.num_layers = num_layers
        self.image_size = image_size
        self.num_tokens = num_tokens
        self.is_gumbel = is_gumbel


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, image):
        loss = self.model.forward(text=input_ids, image=image, mask=attention_mask, return_loss=True)
        return {'loss': loss}


def make_model():
    tokenizer = T5TokenizerFast.from_pretrained('t5-small')
    tokenizer.pad_token = tokenizer.eos_token

    depth = 64
    attn_types = list(islice(cycle(['axial_row', 'axial_col', 'axial_row', 'axial_row']), depth - 1))
    attn_types.append('conv_like')
    shared_layer_ids = list(islice(cycle(range(4)), depth - 1))
    shared_layer_ids.append('w_conv')

    dalle = DALLE(
        vae=VQGanParams(),
        num_text_tokens=tokenizer.vocab_size,
        text_seq_len=256,
        dim=1024,
        depth=depth,
        heads=16,
        dim_head=64,
        attn_types=attn_types,
        ff_dropout=0,
        attn_dropout=0,
        shared_attn_ids=shared_layer_ids,
        shared_ff_ids=shared_layer_ids,
        rotary_emb=True,
        reversible=True,
        share_input_output_emb=True,
        optimize_for_inference=True,
    )
    model = ModelWrapper(dalle)

    return tokenizer, model


def generate(query, *, tokenizer, model,
             batch_size=16, n_iters=1, temperature=0.5, filter_thres=0.5):
    input_ids = torch.tensor(tokenizer(query, add_special_tokens=False, max_length=256, truncation=True)['input_ids'])
    input_ids = F.pad(input_ids, (0, 256 - len(input_ids)), value=1)
    input_ids = input_ids.repeat(batch_size, 1)
    input_ids = input_ids.cuda()

    result = []
    for _ in tqdm(range(n_iters), desc=query, leave=False):
        output = model.model.generate_images(
            input_ids, temperature=temperature, filter_thres=filter_thres, use_cache=True)
        output = rearrange(output, 'b c h w -> b h w c').cpu().numpy()
        result.extend(output)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries', type=str, help='List of queries (*.txt, newline-separated)')
    parser.add_argument('--temperature', type=float, help='Sampling temperature')
    parser.add_argument('--model', type=str, help='DALL-E checkpoint (*.pt)')
    parser.add_argument('--vqgan', type=str, help='VQGAN checkpoint (*.ckpt)')
    parser.add_argument('--vqgan-config', type=str, help='VQGAN config (*.yaml)')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    args = parser.parse_args()

    with open(args.queries) as f:
        queries = [line.rstrip() for line in f]
        queries = [item for item in queries if len(item) > 0]
    print(f'[*] Loaded {len(queries)} queries')

    tokenizer, model = make_model()

    print(f'[*] Model modification time: {datetime.fromtimestamp(os.stat(args.model).st_mtime)}')
    state_dict = torch.load(args.model)
    # The model version optimized for inference requires some renaming in state_dict
    state_dict = OrderedDict([(key.replace('net.fn.fn', 'net.fn.fn.fn').replace('to_qkv', 'fn.to_qkv').replace('to_out', 'fn.to_out'), value)
                            for key, value in state_dict.items()])
    ok = model.load_state_dict(state_dict)
    print(f'[*] Loaded model: {ok}')

    gan = VQGanVAE(args.vqgan, args.vqgan_config).cuda()
    model.model.vae = gan
    model = model.cuda()

    clip_model, clip_preprocess = clip.load("ViT-L/14", device='cuda')

    os.makedirs(args.output_dir, exist_ok=True)
    print(f'[*] Saving results to `{args.output_dir}`')

    for query in tqdm(queries):
        images = generate(query, tokenizer=tokenizer, model=model, batch_size=16, n_iters=8, temperature=args.temperature)

        images_for_clip = torch.cat([clip_preprocess(Image.fromarray((img * 255).astype(np.uint8))).unsqueeze(0).cuda() for img in images])
        text = clip.tokenize([query]).cuda()
        _, logits_per_text = clip_model(images_for_clip, text)
        clip_scores = logits_per_text[0].softmax(dim=-1).cpu().numpy()

        with open(os.path.join(args.output_dir, f'{query}.pickle'), 'wb') as f:
            outputs = {'query': query, 'temperature': args.temperature, 'images': images, 'clip_scores': clip_scores}
            pickle.dump(outputs, f)


if __name__ == '__main__':
    main()
