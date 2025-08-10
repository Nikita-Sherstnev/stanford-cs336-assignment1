from cs336_basics import bpe_tokenizer, modules, utils


if __name__ == '__main__':
    vocab_size = 512
    # special_tokens = ['<|endoftext|>']
    # vocab, merges = bpe_tokenizer.run_train_bpe('data/TinyStoriesV2-GPT4-valid.txt', vocab_size, special_tokens)

    # vocab_filepath = "./tokenizer/vocab"
    # merges_filepath = "./tokenizer/merges"
    #
    # with open(vocab_filepath, "w") as vf:
    #     vf.write(str(vocab))  # {int: bytes}
    #
    # with open(merges_filepath, "w") as mf:
    #     mf.write(str(merges))  # [(b'foo', b'bar'), ...]


    from bpeasy.tokenizer import BPEasyTokenizer

    part = "train"
    with open(f'data/TinyStoriesV2-GPT4-{part}.txt', "r") as f:
        data = f.read()

    iterator = (part for part in data.split('<|endoftext|>'))

    vocab_size = 512
    special_tokens = ['<|endoftext|>']
    max_token_length = 10
    regex_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    tokenizer = BPEasyTokenizer.train(
        iterator,
        vocab_size=vocab_size - len(special_tokens),
        max_token_length=max_token_length,
        regex_pattern=regex_pattern,
        special_tokens=special_tokens,
        fill_to_nearest_multiple_of_eight=True,
        name="bpeasy",
    )

    # tokenizer.export_to_huggingface_format("hf_tokenizer.json")
    tokenizer.save(f"tokenizer/tokenizer-{part}.json")
