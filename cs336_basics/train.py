from cs336_basics import bpe_tokenizer, modules, utils


if __name__ == '__main__':
    d_model = 128
    num_heads = 4
    d_ff = 512
    rope_theta = 10000
    vocab_size = 512
    context_length = 1024
    num_layers = 8

    special_tokens = ['<|endoftext|>']
    vocab, merges = bpe_tokenizer.run_train_bpe('data/TinyStoriesV2-GPT4-train.txt', vocab_size, special_tokens)
    tokenizer = bpe_tokenizer.Tokenizer(vocab, merges, special_tokens)

    model = modules.Transformer(d_model, num_heads, d_ff, rope_theta,
                                vocab_size, context_length, num_layers)
