import regex as re
import os
from typing import Any
from collections import defaultdict, Counter
from collections.abc import Iterable, Iterator


class Chunk:
    __slots__ = ('idx', 'val', 'prev', 'next', 'actual')

    def __init__(self, idx: int, val: int):
        self.idx = idx
        self.val = val
        self.prev: Chunk | None = None
        self.next: Chunk | None = None
        self.actual: bool = True


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, encoding='utf-8') as f:
        text = f.read()

    PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    text_splitted = (token for part in text.split("|".join(special_tokens)) 
                     for token in PAT.finditer(part))

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merge_list: list[tuple[bytes, bytes]] = []

    pair_freq = Counter()
    pairs_to_chunks = defaultdict(set)
    all_tokens = []

    def add_pair(a: Chunk, b: Chunk):
        pair = (a.val, b.val)
        pair_freq[pair] += 1
        pairs_to_chunks[pair].add(a)

    def remove_pair(a: Chunk, b: Chunk):
        pair = (a.val, b.val)
        if pair in pairs_to_chunks and a in pairs_to_chunks[pair]:
            pairs_to_chunks[pair].discard(a)
            pair_freq[pair] -= 1
            if pair_freq[pair] == 0:
                del pair_freq[pair]
                del pairs_to_chunks[pair]

    for text in text_splitted:
        word = list(text.group().encode('utf-8'))
        tokens = [Chunk(i, b) for i, b in enumerate(word)]
        for i in range(len(tokens) - 1):
            tokens[i].next = tokens[i + 1]
            tokens[i + 1].prev = tokens[i]
            add_pair(tokens[i], tokens[i + 1])
        all_tokens.append(tokens)

    del text_splitted

    next_idx = 256
    num_merges = vocab_size - next_idx - len(special_tokens)

    for _ in range(num_merges):
        if not pair_freq:
            break

        most_freq = max(pair_freq.items(), key=lambda x: (x[1], vocab[x[0][0]], vocab[x[0][1]]))[0]
        new_token_val = next_idx
        next_idx += 1
        vocab[new_token_val] = vocab[most_freq[0]] + vocab[most_freq[1]]
        merge_list.append((vocab[most_freq[0]], vocab[most_freq[1]]))

        affected = list(pairs_to_chunks[most_freq])
        for a in affected:
            b = a.next
            if not (a.actual and b and b.actual and (a.val, b.val) == most_freq):
                continue

            # Remove neighbors before merge
            if a.prev and a.prev.actual:
                remove_pair(a.prev, a)
            if b.next and b.next.actual:
                remove_pair(b, b.next)

            # Merge b into a
            a.val = new_token_val
            a.next = b.next
            if b.next:
                b.next.prev = a
            b.actual = False

            # Add new neighbors after merge
            if a.prev and a.prev.actual:
                add_pair(a.prev, a)
            if a.next and a.next.actual:
                add_pair(a, a.next)

        # Clean up used pair
        pairs_to_chunks.pop(most_freq, None)
        pair_freq.pop(most_freq, None)

    vocab = {k + 1: v for k, v in vocab.items()}
    vocab[0] = b'<|endoftext|>'

    return vocab, merge_list


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.id_to_token = vocab.copy()
        self.token_to_id = {v: k for k, v in vocab.items()}
        self.next_id = max(vocab) + 1

        self.merges = merges
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = special_tokens

        if special_tokens:
            for token in special_tokens:
                btoken = token.encode("utf-8")
                if btoken not in self.token_to_id:
                    self.token_to_id[btoken] = self.next_id
                    self.id_to_token[self.next_id] = btoken
                    self.next_id += 1

        self.PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ) -> 'Tokenizer':
        with open(vocab_filepath, "rb") as vf:
            vocab = eval(vf.read())  # {int: bytes}

        with open(merges_filepath, "rb") as mf:
            merges = eval(mf.read())  # [(b'foo', b'bar'), ...]

        return cls(vocab, merges, special_tokens)

    def bpe(self, word: bytes) -> list[bytes]:
        tokens = list(word)
        tokens = [bytes([c]) for c in tokens]

        while True:
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            ranked_pairs = [(self.merge_ranks.get(pair, float("inf")), pair, i)
                            for i, pair in enumerate(pairs)]
            ranked_pairs.sort()
            if not ranked_pairs or ranked_pairs[0][0] == float("inf"):
                break

            _, (a, b), i = ranked_pairs[0]
            tokens = tokens[:i] + [a + b] + tokens[i + 2:]

        return tokens

    def _extract_special_tokens(self, text: str) -> list[tuple[str, bool]]:
        """Split text into (segment, is_special_token) pairs, preserving overlapping special tokens."""
        if not hasattr(self, "special_tokens") or not self.special_tokens:
            return [(text, False)]

        pattern = "|".join(re.escape(tok) for tok in sorted(self.special_tokens, key=len, reverse=True))
        regex = re.compile(f"({pattern})")

        result = []
        last = 0
        for match in regex.finditer(text):
            start, end = match.span()
            if start > last:
                result.append((text[last:start], False))
            result.append((match.group(), True))
            last = end
        if last < len(text):
            result.append((text[last:], False))
        return result

    def encode(self, text: str) -> list[int]:
        ids = []

        segments = self._extract_special_tokens(text)

        for segment, is_special in segments:
            if is_special:
                btoken = segment.encode("utf-8")
                if btoken not in self.token_to_id:
                    self.token_to_id[btoken] = self.next_id
                    self.id_to_token[self.next_id] = btoken
                    self.next_id += 1
                ids.append(self.token_to_id[btoken])
            else:
                tokens = self.PAT.findall(segment)
                for tok in tokens:
                    bword = tok.encode("utf-8")
                    for token in self.bpe(bword):
                        if token not in self.token_to_id:
                            self.token_to_id[token] = self.next_id
                            self.id_to_token[self.next_id] = token
                            self.next_id += 1
                        ids.append(self.token_to_id[token])

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for line in iterable:
            yield from self.encode(line)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.id_to_token[i] for i in ids).decode("utf-8", errors="replace")



def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    return Tokenizer(vocab, merges, special_tokens)
