import pytest

from bpe_tokenizer import BPETokenizer

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def simple_corpus():
    """Basic corpus for simple tests."""
    return [
        "the cat sat on the mat",
        "the dog sat on the log",
    ]


@pytest.fixture
def repetitive_corpus():
    """Corpus with highly repetitive patterns for predictable merges."""
    return ["aa aa aa", "bb bb bb", "ab ab ab"]


@pytest.fixture
def morphological_corpus():
    """Corpus to test morpheme-like merges."""
    return [
        "lower lowest low lowering lowered",
        "higher highest high",
        "walking walked walker walks",
    ]


@pytest.fixture
def trained_tokenizer(simple_corpus):
    """Pre-trained tokenizer for encode/decode tests."""
    tokenizer = BPETokenizer(vocab_size=300)
    tokenizer.train(simple_corpus)
    return tokenizer


@pytest.fixture
def minimal_tokenizer(repetitive_corpus):
    """Tokenizer with minimal vocab for predictable behavior."""
    tokenizer = BPETokenizer(vocab_size=270)
    tokenizer.train(repetitive_corpus)
    return tokenizer


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestInitialization:
    """Tests for tokenizer initialization."""

    def test_default_vocab_size(self):
        tokenizer = BPETokenizer()
        assert tokenizer.vocab_size == 1000

    def test_custom_vocab_size(self):
        tokenizer = BPETokenizer(vocab_size=500)
        assert tokenizer.vocab_size == 500

    def test_initial_state_empty(self):
        tokenizer = BPETokenizer()
        assert tokenizer.merges == {}
        assert tokenizer.vocab == {}

    def test_small_vocab_size(self):
        tokenizer = BPETokenizer(vocab_size=256)
        assert tokenizer.vocab_size == 256

    def test_large_vocab_size(self):
        tokenizer = BPETokenizer(vocab_size=50000)
        assert tokenizer.vocab_size == 50000


# =============================================================================
# INTERNAL METHOD TESTS
# =============================================================================


class TestGetStats:
    """Tests for the _get_stats method."""

    def test_simple_pair_counting(self):
        tokenizer = BPETokenizer()
        corpus = [["a", "b", "c"], ["a", "b", "d"]]
        stats = tokenizer._get_stats(corpus)
        assert stats[("a", "b")] == 2
        assert stats[("b", "c")] == 1
        assert stats[("b", "d")] == 1

    def test_empty_corpus(self):
        tokenizer = BPETokenizer()
        stats = tokenizer._get_stats([])
        assert stats == {}

    def test_single_char_words(self):
        tokenizer = BPETokenizer()
        corpus = [["a"], ["b"], ["c"]]
        stats = tokenizer._get_stats(corpus)
        assert stats == {}

    def test_repeated_pairs(self):
        tokenizer = BPETokenizer()
        corpus = [["a", "a", "a", "a"]]
        stats = tokenizer._get_stats(corpus)
        assert stats[("a", "a")] == 3

    def test_no_pairs_in_empty_words(self):
        tokenizer = BPETokenizer()
        corpus = [[], [], []]
        stats = tokenizer._get_stats(corpus)
        assert stats == {}


class TestMergePair:
    """Tests for the _merge_pair method."""

    def test_simple_merge(self):
        tokenizer = BPETokenizer()
        corpus = [["a", "b", "c"]]
        result = tokenizer._merge_pair(corpus, ("a", "b"))
        assert result == [["ab", "c"]]

    def test_multiple_occurrences(self):
        tokenizer = BPETokenizer()
        corpus = [["a", "b", "a", "b"]]
        result = tokenizer._merge_pair(corpus, ("a", "b"))
        assert result == [["ab", "ab"]]

    def test_no_match(self):
        tokenizer = BPETokenizer()
        corpus = [["x", "y", "z"]]
        result = tokenizer._merge_pair(corpus, ("a", "b"))
        assert result == [["x", "y", "z"]]

    def test_merge_across_words(self):
        tokenizer = BPETokenizer()
        corpus = [["a", "b"], ["a", "b"], ["c", "d"]]
        result = tokenizer._merge_pair(corpus, ("a", "b"))
        assert result == [["ab"], ["ab"], ["c", "d"]]

    def test_adjacent_pairs(self):
        tokenizer = BPETokenizer()
        corpus = [["a", "b", "b", "c"]]
        result = tokenizer._merge_pair(corpus, ("b", "b"))
        assert result == [["a", "bb", "c"]]

    def test_overlapping_potential_merges(self):
        tokenizer = BPETokenizer()
        corpus = [["a", "a", "a"]]
        result = tokenizer._merge_pair(corpus, ("a", "a"))
        # Should merge first two, leaving third alone
        assert result == [["aa", "a"]]

    def test_empty_corpus_merge(self):
        tokenizer = BPETokenizer()
        corpus = []
        result = tokenizer._merge_pair(corpus, ("a", "b"))
        assert result == []


# =============================================================================
# TRAINING TESTS
# =============================================================================


class TestTraining:
    """Tests for the train method."""

    def test_vocab_includes_base_bytes(self, simple_corpus):
        tokenizer = BPETokenizer(vocab_size=300)
        tokenizer.train(simple_corpus)
        # Should have all 256 byte values
        for i in range(256):
            assert chr(i) in tokenizer.vocab

    def test_vocab_includes_end_marker(self, simple_corpus):
        tokenizer = BPETokenizer(vocab_size=300)
        tokenizer.train(simple_corpus)
        assert "</w>" in tokenizer.vocab

    def test_merges_created(self, simple_corpus):
        tokenizer = BPETokenizer(vocab_size=300)
        tokenizer.train(simple_corpus)
        assert len(tokenizer.merges) > 0

    def test_vocab_size_respected(self, simple_corpus):
        vocab_size = 280
        tokenizer = BPETokenizer(vocab_size=vocab_size)
        tokenizer.train(simple_corpus)
        assert len(tokenizer.vocab) <= vocab_size

    def test_frequent_pairs_merged_first(self, repetitive_corpus):
        tokenizer = BPETokenizer(vocab_size=260)
        tokenizer.train(repetitive_corpus)
        # First merge should be one of the most frequent pairs
        first_merge = list(tokenizer.merges.keys())[0]
        # 'b</w>' should be the first merge due to frequency
        assert len(tokenizer.merges) > 0
        assert first_merge == ("b", "</w>")

    def test_empty_corpus_training(self):
        tokenizer = BPETokenizer(vocab_size=300)
        tokenizer.train([])
        # Should still have base vocab
        assert len(tokenizer.vocab) >= 256
        assert len(tokenizer.merges) == 0

    def test_single_word_training(self):
        tokenizer = BPETokenizer(vocab_size=300)
        tokenizer.train(["hello"])
        assert "</w>" in tokenizer.vocab
        assert "h" in tokenizer.vocab

    def test_unique_vocab_ids(self, simple_corpus):
        tokenizer = BPETokenizer(vocab_size=300)
        tokenizer.train(simple_corpus)
        ids = list(tokenizer.vocab.values())
        assert len(ids) == len(set(ids))  # All unique

    def test_training_idempotent_merges(self, simple_corpus):
        """Training twice should produce same merges."""
        tokenizer1 = BPETokenizer(vocab_size=280)
        tokenizer1.train(simple_corpus)

        tokenizer2 = BPETokenizer(vocab_size=280)
        tokenizer2.train(simple_corpus)

        assert tokenizer1.merges == tokenizer2.merges


# =============================================================================
# TOKENIZE WORD TESTS
# =============================================================================


class TestTokenizeWord:
    """Tests for the _tokenize_word method."""

    def test_applies_merges(self, trained_tokenizer):
        tokens = trained_tokenizer._tokenize_word("the")
        # Should have </w> at end
        assert tokens[-1] == "</w>" or tokens[-1].endswith("</w>")

    def test_unknown_word_chars_preserved(self, trained_tokenizer):
        tokens = trained_tokenizer._tokenize_word("xyz")
        # All characters should be present in some form
        result = "".join(tokens).replace("</w>", "")
        assert result == "xyz"

    def test_empty_word(self, trained_tokenizer):
        tokens = trained_tokenizer._tokenize_word("")
        assert tokens == ["</w>"]

    def test_single_char(self, trained_tokenizer):
        tokens = trained_tokenizer._tokenize_word("a")
        assert "</w>" in tokens[-1] or tokens[-1] == "</w>"


# =============================================================================
# ENCODE TESTS
# =============================================================================


class TestEncode:
    """Tests for the encode method."""

    def test_returns_list_of_ints(self, trained_tokenizer):
        result = trained_tokenizer.encode("the cat")
        assert isinstance(result, list)
        assert all(isinstance(x, int) for x in result)

    def test_empty_string(self, trained_tokenizer):
        result = trained_tokenizer.encode("")
        assert result == []

    def test_single_word(self, trained_tokenizer):
        result = trained_tokenizer.encode("cat")
        assert len(result) >= 1

    def test_known_words_encode(self, trained_tokenizer):
        result = trained_tokenizer.encode("the cat sat")
        assert len(result) > 0
        # All tokens should be valid vocab ids
        assert all(x in trained_tokenizer.vocab.values() for x in result)

    def test_repeated_words(self, trained_tokenizer):
        result = trained_tokenizer.encode("the the the")
        # Should have repeating pattern
        assert len(result) >= 3

    def test_whitespace_handling(self, trained_tokenizer):
        result1 = trained_tokenizer.encode("a b")
        result2 = trained_tokenizer.encode("a  b")  # double space
        # Both should produce tokens for 'a' and 'b'
        assert len(result1) >= 2
        assert len(result2) >= 2


# =============================================================================
# DECODE TESTS
# =============================================================================


class TestDecode:
    """Tests for the decode method."""

    def test_empty_list(self, trained_tokenizer):
        result = trained_tokenizer.decode([])
        assert result == ""

    def test_roundtrip_simple(self, trained_tokenizer):
        original = "the cat"
        encoded = trained_tokenizer.encode(original)
        decoded = trained_tokenizer.decode(encoded)
        assert decoded == original

    def test_roundtrip_single_word(self, trained_tokenizer):
        original = "cat"
        encoded = trained_tokenizer.encode(original)
        decoded = trained_tokenizer.decode(encoded)
        assert decoded == original

    def test_roundtrip_multiple_words(self, trained_tokenizer):
        original = "the cat sat on the mat"
        encoded = trained_tokenizer.encode(original)
        decoded = trained_tokenizer.decode(encoded)
        assert decoded == original

    def test_unknown_id_handling(self, trained_tokenizer):
        # Use an ID that doesn't exist
        result = trained_tokenizer.decode([999999])
        assert "<unk>" in result

    def test_mixed_known_unknown_ids(self, trained_tokenizer):
        # Get a known ID and mix with unknown
        known_id = list(trained_tokenizer.vocab.values())[0]
        result = trained_tokenizer.decode([known_id, 999999])
        assert "<unk>" in result


# =============================================================================
# ROUNDTRIP / INTEGRATION TESTS
# =============================================================================


class TestRoundtrip:
    """Integration tests for encode-decode roundtrips."""

    def test_training_corpus_roundtrip(self, simple_corpus):
        tokenizer = BPETokenizer(vocab_size=300)
        tokenizer.train(simple_corpus)

        for text in simple_corpus:
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded)
            assert decoded == text

    def test_unseen_text_with_known_chars(self, trained_tokenizer):
        # Text not in training but uses same characters
        text = "mat cat hat"
        encoded = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(encoded)
        assert decoded == text

    def test_partial_overlap_with_training(self, trained_tokenizer):
        text = "the cat ran"  # 'ran' not in training
        encoded = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(encoded)
        assert decoded == text

    @pytest.mark.parametrize(
        "text",
        [
            "a",
            "ab",
            "abc",
            "the",
            "test",
            "hello world",
        ],
    )
    def test_various_lengths(self, trained_tokenizer, text):
        encoded = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(encoded)
        assert decoded == text


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_small_vocab(self):
        tokenizer = BPETokenizer(vocab_size=257)  # Just over base bytes
        tokenizer.train(["aa bb cc"])
        assert len(tokenizer.vocab) <= 257

    def test_vocab_larger_than_possible_merges(self):
        tokenizer = BPETokenizer(vocab_size=10000)
        tokenizer.train(["ab"])  # Very small corpus
        # Should not crash, vocab limited by possible merges
        assert len(tokenizer.vocab) < 10000

    def test_single_character_corpus(self):
        tokenizer = BPETokenizer(vocab_size=300)
        tokenizer.train(["a a a a a"])
        encoded = tokenizer.encode("a a")
        decoded = tokenizer.decode(encoded)
        assert decoded == "a a"

    def test_numbers_in_text(self):
        tokenizer = BPETokenizer(vocab_size=300)
        tokenizer.train(["123 456 789"])
        encoded = tokenizer.encode("123 456")
        decoded = tokenizer.decode(encoded)
        assert decoded == "123 456"

    def test_special_characters(self):
        tokenizer = BPETokenizer(vocab_size=300)
        tokenizer.train(["hello! world? yes."])
        encoded = tokenizer.encode("hello!")
        decoded = tokenizer.decode(encoded)
        assert decoded == "hello!"

    def test_unicode_characters(self):
        tokenizer = BPETokenizer(vocab_size=300)
        tokenizer.train(["café naïve résumé"])
        encoded = tokenizer.encode("café")
        decoded = tokenizer.decode(encoded)
        assert decoded == "café"

    def test_long_word(self):
        tokenizer = BPETokenizer(vocab_size=300)
        long_word = "a" * 100
        tokenizer.train([long_word])
        encoded = tokenizer.encode(long_word)
        decoded = tokenizer.decode(encoded)
        assert decoded == long_word

    def test_many_unique_words(self):
        tokenizer = BPETokenizer(vocab_size=500)
        corpus = [f"word{i}" for i in range(100)]
        tokenizer.train(corpus)
        # Should handle many unique words
        encoded = tokenizer.encode("word1 word50")
        decoded = tokenizer.decode(encoded)
        assert decoded == "word1 word50"


# =============================================================================
# MERGE ORDER TESTS
# =============================================================================


class TestMergeOrder:
    """Tests verifying merge behavior and ordering."""

    def test_merge_order_preserved(self):
        tokenizer = BPETokenizer(vocab_size=270)
        tokenizer.train(["ab ab ab ab", "cd cd"])

        # 'ab' pair appears more than 'cd', should be merged first
        merge_list = list(tokenizer.merges.keys())
        # Early merges should involve frequent pairs
        assert len(merge_list) > 0

    def test_merges_are_greedy(self):
        tokenizer = BPETokenizer(vocab_size=270)
        tokenizer.train(["aaa aaa aaa"])

        # After first merge of 'aa', should see 'aa' in vocab
        if ("a", "a") in tokenizer.merges:
            assert "aa" in tokenizer.vocab


# =============================================================================
# VOCAB CONSISTENCY TESTS
# =============================================================================


class TestVocabConsistency:
    """Tests for vocabulary consistency and integrity."""

    def test_all_merged_tokens_in_vocab(self, trained_tokenizer):
        for merged in trained_tokenizer.merges.values():
            assert merged in trained_tokenizer.vocab

    def test_merge_components_in_vocab(self, trained_tokenizer):
        """Both parts of each merge should exist in vocab."""
        for a, b in trained_tokenizer.merges.keys():
            assert a in trained_tokenizer.vocab
            assert b in trained_tokenizer.vocab

    def test_vocab_ids_are_sequential(self, simple_corpus):
        tokenizer = BPETokenizer(vocab_size=300)
        tokenizer.train(simple_corpus)

        ids = sorted(tokenizer.vocab.values())
        # Check that IDs from 0 to 255 exist (base bytes)
        for i in range(256):
            assert i in ids


# =============================================================================
# PERFORMANCE / STRESS TESTS
# =============================================================================


class TestPerformance:
    """Basic performance and stress tests."""

    def test_moderate_corpus_size(self):
        """Test with a moderately sized corpus."""
        corpus = ["the quick brown fox jumps over the lazy dog"] * 100
        tokenizer = BPETokenizer(vocab_size=400)
        tokenizer.train(corpus)

        encoded = tokenizer.encode("the quick brown fox")
        decoded = tokenizer.decode(encoded)
        assert decoded == "the quick brown fox"

    def test_many_short_words(self):
        corpus = ["a b c d e f g h i j k l m n o p"] * 50
        tokenizer = BPETokenizer(vocab_size=300)
        tokenizer.train(corpus)

        encoded = tokenizer.encode("a b c")
        decoded = tokenizer.decode(encoded)
        assert decoded == "a b c"

    def test_repeated_training_stable(self):
        """Multiple training calls should be stable (though not recommended)."""
        corpus = ["hello world"]
        tokenizer = BPETokenizer(vocab_size=280)
        tokenizer.train(corpus)
        merges1 = dict(tokenizer.merges)

        # Train again (overwrites)
        tokenizer.train(corpus)
        merges2 = dict(tokenizer.merges)

        assert merges1 == merges2


# =============================================================================
# DETERMINISM TESTS
# =============================================================================


class TestDeterminism:
    """Tests to verify deterministic behavior."""

    def test_same_input_same_output(self):
        corpus = ["hello world hello"]

        tokenizer1 = BPETokenizer(vocab_size=280)
        tokenizer1.train(corpus)

        tokenizer2 = BPETokenizer(vocab_size=280)
        tokenizer2.train(corpus)

        text = "hello world"
        assert tokenizer1.encode(text) == tokenizer2.encode(text)

    def test_encode_deterministic(self, trained_tokenizer):
        text = "the cat sat"
        result1 = trained_tokenizer.encode(text)
        result2 = trained_tokenizer.encode(text)
        assert result1 == result2

    def test_decode_deterministic(self, trained_tokenizer):
        ids = trained_tokenizer.encode("the cat")
        result1 = trained_tokenizer.decode(ids)
        result2 = trained_tokenizer.decode(ids)
        assert result1 == result2
