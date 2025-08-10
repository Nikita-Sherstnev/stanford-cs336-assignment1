import numpy as np
from bpeasy.tokenizer import BPEasyTokenizer

part = 'train'
tokenizer = BPEasyTokenizer.from_file(f"tokenizer/tokenizer-{part}.json")

with open(f'data/TinyStoriesV2-GPT4-{part}.txt', "r") as f:
    data = f.read()

encoded = tokenizer.encode(data, allowed_special={'<|endoftext|>'})

print(type(encoded))
print(encoded[0:5])
np_array = np.array(encoded, dtype=np.uint16)
np.save(f'data/dataset_{part}.npy', np_array)
