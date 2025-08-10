import numpy as np
import torch

from cs336_basics import bpe_tokenizer, modules, utils
from bpeasy.tokenizer import BPEasyTokenizer


if __name__ == '__main__':
    prompt = 'Once upon a time'

    tokenizer = BPEasyTokenizer.from_file("tokenizer/tokenizer-train.json")
    input = tokenizer.encode(prompt)
    inputs = torch.tensor(input).unsqueeze(0)
    token_positions = torch.arange(inputs.shape[-1])
    end_of_text_ind = tokenizer.encode("<|endoftext|>", allowed_special={'<|endoftext|>'})

    d_model = 128
    num_heads = 4
    d_ff = 512
    rope_theta = 10000
    vocab_size = 512
    context_length = 1024
    num_layers = 8
    batch_size = 16
    device = 'cuda'

    model = modules.Transformer(d_model, num_heads, d_ff, rope_theta,
                                vocab_size, context_length, num_layers, device=torch.device(device))

    optimizer = utils.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-08)

    utils.run_load_checkpoint('checkpoints/model_1000_it.pt', model, optimizer)

    # print(inputs)
    # print(logits.shape)
    temp = 0.01
    top_p = 0.8

    max_tokens = 100
    for tok in range(max_tokens):
        logits = model(inputs, token_positions)

        next_logits = logits[:, -1, :]
        next_logits /= temp
        probs = modules.softmax(next_logits, -1)
        probs = probs.squeeze()

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Find cutoff
        cutoff = cumulative_probs > top_p
        if torch.any(cutoff):
            cutoff_index = torch.argmax(cutoff.float()).item()
            sorted_probs = sorted_probs[:cutoff_index + 1]
            sorted_indices = sorted_indices[:cutoff_index + 1]

        # Renormalize the filtered probabilities
        filtered_probs = sorted_probs / sorted_probs.sum()

        # Sample from the filtered distribution
        sampled_index = torch.multinomial(filtered_probs, 1).item()

        next_token = sorted_indices[sampled_index].item()
        # next_token = torch.multinomial(probs, 1).item()

        input.append(next_token)
        inputs = torch.tensor(input).unsqueeze(0)
        token_positions = torch.arange(inputs.shape[-1])

        print(tokenizer.decode([next_token]), end='')
        # print()
        # print(next_token)
        # print(end_of_text_ind)
        if next_token == end_of_text_ind[0]:
            break
