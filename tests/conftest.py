"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
import tiktoken

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def tokenizer() -> tiktoken.Encoding:
    """A cl100k_base tokenizer shared across the test session."""
    return tiktoken.get_encoding("cl100k_base")


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR
