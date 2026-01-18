from collections import defaultdict


class BPETokenizer:

    def __init__(self, vocab_size=2500):
        self.vocab_size = vocab_size
        self.merges = {}
        self.id_to_pair = {}

    def _get_stats(self, token_ids: list[int]) -> dict[tuple, int]:
        counts = defaultdict(int)
        for pair in zip(token_ids, token_ids[1:]):
            counts[pair] += 1

        return counts

    def _find_most_common_pair(self, tokens: list[int]):
        cts = self._get_stats(tokens)

        return max(cts.items(), key=lambda x: x[1])

    def _merge_pair(self, token_ids: list[int], pair: tuple, new_id: int):
        result = []
        i = 0
        while i < len(token_ids):
            if (
                pair[0] == token_ids[i]
                and i < len(token_ids) - 1
                and pair[1] == token_ids[i + 1]
            ):
                result.append(new_id)
                i += 2
            else:
                result.append(token_ids[i])
                i += 1

        return result

    def train(self, text: str):
        tokens = list(map(int, text.encode("utf8")))

        while len(tokens) >= 2 and len(self.merges) + 256 < self.vocab_size:
            pair, ct = self._find_most_common_pair(tokens)
            if ct == 1:  # no further compression possible
                break
            new_id = 256 + len(self.merges)  # ids start at 0 so our first id is 256
            self.merges[pair] = new_id
            tokens = self._merge_pair(tokens, pair, new_id)

        self.id_to_pair = {v: k for k, v in self.merges.items()}

    def encode(self, text: str):
        token_ids = list(map(int, text.encode("utf8")))
        while len(token_ids) >= 2:
            pairs = set(zip(token_ids, token_ids[1:]))
            pair = min(pairs, key=lambda p: self.merges.get(p, self.vocab_size + 1))
            if pair not in self.merges:
                break

            new_id = self.merges[pair]
            token_ids = self._merge_pair(token_ids, pair, new_id)

        return token_ids

    def decode(self, token_ids: list[int]):
        result = []
        stack: list = list(reversed(token_ids))

        while stack:
            next_id = stack.pop()
            if next_id < 256:
                result.append(next_id)
            elif next_id in self.id_to_pair:
                id_1, id_2 = self.id_to_pair[next_id]
                stack.append(id_2)
                stack.append(id_1)
            else:
                result.extend(b"UNK")

        return bytes(result).decode("utf-8")
