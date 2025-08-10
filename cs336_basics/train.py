import time

import numpy as np
import torch

from cs336_basics import bpe_tokenizer, modules, utils
from bpeasy.tokenizer import BPEasyTokenizer


if __name__ == '__main__':
    d_model = 128
    num_heads = 4
    d_ff = 512
    rope_theta = 10000
    vocab_size = 512
    context_length = 1024
    num_layers = 8
    batch_size = 16
    device = 'cuda'

    part = "train"
    dataset = np.memmap(f'data/dataset_{part}.npy', dtype=np.uint16)

    # Check that dataset can be decoded correctly
    batch, targets = utils.run_get_batch(dataset, batch_size, context_length, device)
    tokenizer = BPEasyTokenizer.from_file(f"tokenizer/tokenizer-{part}.json")
    decoded = tokenizer.decode(batch[0].tolist())

    model = modules.Transformer(d_model, num_heads, d_ff, rope_theta,
                                vocab_size, context_length, num_layers, device=torch.device(device))
    print(f"Model size: {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")
    model = torch.compile(model)

    token_positions = torch.arange(batch.shape[-1])

    optimizer = utils.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-08)

    for it in range(1, 1000 + 1):
        s = time.monotonic()
        batch, targets = utils.run_get_batch(dataset, batch_size, context_length, device)
        decoded = tokenizer.decode(batch[0].tolist())

        optimizer.zero_grad()
        logits = model(batch, token_positions)

        loss = utils.run_cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        loss.backward()
        optimizer.step()

        e = time.monotonic()
        print(f"Iter {it} loss: {loss.item()}, time: {int((e-s)*1000)} ms")

        if it % 200 == 0:
            utils.run_save_checkpoint(model, optimizer, it, f'checkpoints/model_{it}_it.pt')
