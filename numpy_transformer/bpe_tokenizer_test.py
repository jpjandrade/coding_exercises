import numpy as np
import pytest

from bpe_tokenizer import BPETokenizer


class TestBPETokenizer:
    """Tests for BPE tokenizer training, encoding, and decoding."""

    @pytest.fixture
    def tokenizer(self):
        return BPETokenizer()

    @pytest.fixture
    def trained_tokenizer(self):
        tokenizer = BPETokenizer()
        tokenizer.train("hello hello hello world world")
        return tokenizer

    # === Roundtrip tests ===

    def test_encode_decode_roundtrip_simple(self, trained_tokenizer):
        text = "hello world"
        encoded = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(encoded)
        assert decoded == text

    def test_encode_decode_roundtrip_unseen_text(self, trained_tokenizer):
        text = "hero"  # not in training, but shares bytes
        encoded = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(encoded)
        assert decoded == text

    def test_encode_decode_roundtrip_empty(self, trained_tokenizer):
        encoded = trained_tokenizer.encode("")
        decoded = trained_tokenizer.decode(encoded)
        assert decoded == ""

    # === Encoding tests ===

    def test_encode_returns_list(self, trained_tokenizer):
        encoded = trained_tokenizer.encode("hello")
        assert isinstance(encoded, list)

    def test_encode_reduces_length_after_training(self, trained_tokenizer):
        text = "hello"
        encoded = trained_tokenizer.encode(text)
        raw_bytes = list(text.encode("utf-8"))
        assert len(encoded) <= len(raw_bytes)

    def test_encode_without_training_returns_raw_bytes(self, tokenizer):
        text = "hello"
        encoded = tokenizer.encode(text)
        expected = list(text.encode("utf-8"))
        assert encoded == expected

    # === Decoding tests ===

    def test_decode_raw_bytes(self, tokenizer):
        raw = np.array([104, 101, 108, 108, 111])  # "hello" in utf-8
        assert tokenizer.decode(raw) == "hello"

    def test_decode_empty_list(self, tokenizer):
        assert tokenizer.decode([]) == ""

    # === Training tests ===

    def test_train_creates_merges(self, tokenizer):
        tokenizer.train("aaaa")  # should merge (97, 97) -> 256
        encoded = tokenizer.encode("aaaa")
        assert len(encoded) < 4

    def test_train_empty_string(self, tokenizer):
        tokenizer.train("")  # should not crash
        assert tokenizer.encode("test") is not None

    def test_train_single_char(self, tokenizer):
        tokenizer.train("a")
        encoded = tokenizer.encode("a")
        decoded = tokenizer.decode(encoded)
        assert decoded == "a"

    # === Unicode tests ===

    def test_unicode_roundtrip(self, tokenizer):
        tokenizer.train("über über über café café")
        for text in ["über", "café", "über café"]:
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded)
            assert decoded == text

    def test_emoji_roundtrip(self, tokenizer):
        tokenizer.train("🎉🎉🎉 party 🎉🎉🎉")
        text = "🎉 party"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text

    def test_multilingual_roundtrip(self, tokenizer):
        training = "こんにちは こんにちは hello hello"
        tokenizer.train(training)
        text = "こんにちは hello"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text

    # === Determinism tests ===

    def test_encode_deterministic(self, trained_tokenizer):
        text = "hello world"
        encoded1 = trained_tokenizer.encode(text)
        encoded2 = trained_tokenizer.encode(text)
        assert encoded1 == encoded2

    def test_training_deterministic(self):
        tok1 = BPETokenizer()
        tok2 = BPETokenizer()
        text = "the quick brown fox jumps over the lazy dog"
        tok1.train(text)
        tok2.train(text)

        test_text = "the fox"
        assert tok1.encode(test_text) == tok2.encode(test_text)

    # === Edge cases ===

    def test_repeated_single_byte(self, tokenizer):
        tokenizer.train("x" * 1000)
        encoded = tokenizer.encode("x" * 100)
        decoded = tokenizer.decode(encoded)
        assert decoded == "x" * 100

    def test_all_ascii_bytes(self, tokenizer):
        text = "".join(chr(i) for i in range(32, 127))
        tokenizer.train(text * 10)
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text

    # === Vocab size tests ===

    def test_vocab_size_limits_merges(self):
        tokenizer = BPETokenizer(vocab_size=258)  # 256 base + 2 merges max
        tokenizer.train("aaaa bbbb cccc dddd" * 100)
        # Should only learn 2 merges maximum
        assert len(tokenizer.merges) <= 2

    def test_vocab_size_default(self, tokenizer):
        assert tokenizer.vocab_size == 2500

    # === Whitespace tests ===

    def test_whitespace_roundtrip(self, tokenizer):
        text = "hello\tworld\nfoo  bar"
        tokenizer.train(text * 10)
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text

    def test_newlines_only(self, tokenizer):
        text = "\n\n\n"
        tokenizer.train(text * 10)
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text

    # === Token validity tests ===

    def test_encoded_tokens_are_valid(self, trained_tokenizer):
        text = "hello world hello"
        encoded = trained_tokenizer.encode(text)
        valid_ids = set(range(256)) | set(trained_tokenizer.merges.values())
        for token_id in encoded:
            assert token_id in valid_ids, f"Invalid token ID: {token_id}"

    def test_decode_unknown_token_returns_unk(self, tokenizer):
        # Token ID 9999 is not a base byte and not in merges
        result = tokenizer.decode([104, 9999, 105])  # h, UNK, i
        assert result == "hUNKi"

    # === Training behavior tests ===

    def test_train_stops_when_no_repeated_pairs(self, tokenizer):
        # Each pair appears only once, so no merges should be learned
        tokenizer.train("abcdefgh")
        assert len(tokenizer.merges) == 0

    def test_train_merges_most_frequent_pair_first(self, tokenizer):
        # "aa" appears 4 times, "bb" appears 2 times
        tokenizer.train("aa aa aa aa bb bb")
        # First merge should be (97, 97) -> 256 (the 'a' pair)
        assert (97, 97) in tokenizer.merges
        first_merge_id = tokenizer.merges[(97, 97)]
        assert first_merge_id == 256  # First learned merge

    # === Merge priority tests ===

    def test_encode_applies_merges_in_priority_order(self):
        tokenizer = BPETokenizer()
        # Train so that 'aa' is merged first, then 'aaa' (256, 97)
        tokenizer.train("aaa aaa aaa aaa")
        encoded = tokenizer.encode("aaa")
        # Should be length 2 or less if nested merges work
        assert len(encoded) <= 2

    # === Longer text tests ===

    def test_longer_training_corpus(self, tokenizer):
        corpus = "the quick brown fox jumps over the lazy dog " * 100
        tokenizer.train(corpus)
        text = "the quick fox"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text
        # Should have compression
        assert len(encoded) < len(text)

    # === Binary-like edge cases ===

    def test_high_bytes_roundtrip(self, tokenizer):
        # Characters that produce high byte values
        text = "ÿþýüûú"  # Latin chars with byte values > 200
        tokenizer.train(text * 20)
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text

    # === Partial overlap tests ===

    def test_encode_partial_overlap_with_training(self, tokenizer):
        tokenizer.train("ab ab ab ab")
        # "abc" has the "ab" pair but "c" is separate
        encoded = tokenizer.encode("abc")
        decoded = tokenizer.decode(encoded)
        assert decoded == "abc"

    def test_encode_text_longer_than_training(self, tokenizer):
        tokenizer.train("hi")
        # Encode much longer text with same chars
        text = "hi" * 100
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text
