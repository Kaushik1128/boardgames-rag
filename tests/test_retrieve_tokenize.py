"""Tokenizer edge cases for the BM25 input pipeline."""

from __future__ import annotations

from boardgames_rag.retrieve import tokenize


class TestTokenize:
    def test_empty_string(self):
        assert tokenize("") == []

    def test_single_word_lowercased(self):
        assert tokenize("Hello") == ["hello"]

    def test_full_sentence_lowercased_and_split(self):
        tokens = tokenize("Players Collect Resources")
        assert tokens == ["players", "collect", "resources"]

    def test_splits_on_punctuation(self):
        tokens = tokenize("Magic: The Gathering; rules!")
        assert tokens == ["magic", "the", "gathering", "rules"]

    def test_preserves_dotted_identifier(self):
        # "100.6b" is a single token — critical for MTG rule citations.
        tokens = tokenize("Rule 100.6b applies here.")
        assert "100.6b" in tokens
        assert "100" not in tokens  # not split

    def test_preserves_apostrophe(self):
        # "card's" survives so retrieval finds possessive forms.
        tokens = tokenize("the card's text")
        assert "card's" in tokens

    def test_preserves_hyphenated_word(self):
        tokens = tokenize("a two-player game")
        assert "two-player" in tokens

    def test_handles_multiline(self):
        text = "Line one.\nLine two."
        tokens = tokenize(text)
        assert tokens.count("line") == 2
        assert "one" in tokens
        assert "two" in tokens

    def test_numeric_only_token(self):
        assert "100" in tokenize("turn 100")

    def test_brackets_and_parens_split(self):
        tokens = tokenize("(see 100.6) [optional]")
        assert "see" in tokens
        assert "100.6" in tokens
        assert "optional" in tokens
        assert "(" not in tokens
