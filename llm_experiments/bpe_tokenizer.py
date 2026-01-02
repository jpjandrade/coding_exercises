from collections import defaultdict


class BPETokenizer:
    _UNKNOWN = "<unk>"
    _END_OF_WORD = "</w>"

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.merges = {}  # (pair) -> merged_token
        self.vocab = {}  # token -> id
        self.id_to_token = {}  # id -> token

    def _get_stats(self, corpus: list[list[str]]) -> dict[tuple[str, str], int]:
        pairs = defaultdict(int)
        for word in corpus:
            for i in range(len(word) - 1):
                pairs[word[i], word[i + 1]] += 1

        return pairs

    def _merge_pair(self, corpus: list[list[str]], pair: tuple[str, str]) -> list[str]:
        new_corpus = []
        bigram = pair[0] + pair[1]
        print(f"Merging {bigram}...")

        for word in corpus:
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(bigram)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            print(f"Old word: {word}, new word: {new_word}")
            new_corpus.append(new_word)

        return new_corpus

    def train(self, texts: list[str]):
        corpus = []
        for text in texts:
            words = text.split()
            for word in words:
                corpus.append(list(word) + [self._END_OF_WORD])

        self.vocab = {chr(i): i for i in range(256)}

        for word in corpus:
            for char in word:
                if char not in self.vocab:
                    self.vocab[char] = len(self.vocab)

        num_merges = self.vocab_size - len(self.vocab)

        for i in range(num_merges):
            pairs = self._get_stats(corpus)
            if not pairs:
                break

            best_pair = max(pairs, key=lambda pair: pairs[pair])
            print(f"Best pair: {best_pair}.")
            corpus = self._merge_pair(corpus, best_pair)
            merged_token = best_pair[0] + best_pair[1]
            self.merges[best_pair] = merged_token
            self.vocab[merged_token] = len(self.vocab)

            if (i + 1) % 100 == 0:
                print(
                    f"Merge {i + 1}: {best_pair} -> {merged_token} (freq: {pairs[best_pair]})"
                )

        self.id_to_token = {v: k for k, v in self.vocab.items()}

    def _tokenize_word(self, word: str):
        """Apply learned pairwise token merges to a single word."""
        tokens = list(word) + [self._END_OF_WORD]

        for pair, merged in self.merges.items():
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    tokens = tokens[:i] + [merged] + tokens[i + 2 :]
                else:
                    i += 1

        return tokens

    def encode(self, text: str):
        tokens = []
        for word in text.split():
            tokens.extend(self._tokenize_word(word))

        return [self.vocab.get(t, self.vocab.get(self._UNKNOWN, 0)) for t in tokens]

    def decode(self, ids: list[int]) -> str:
        tokens = [self.id_to_token.get(i, self._UNKNOWN) for i in ids]
        text = "".join(tokens)
        return text.replace(self._END_OF_WORD, " ").strip()
