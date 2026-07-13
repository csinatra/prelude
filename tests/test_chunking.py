from ingest.chunking import split_oversized


def test_small_text_passes_through():
    assert split_oversized(text="import pandas as pd") == ["import pandas as pd"]


def test_oversized_text_splits_at_blank_lines():
    blocks = [f"def f{i}():\n    return {i}" for i in range(10)]
    text = "\n\n".join(blocks)
    chunks = split_oversized(text=text, max_chars=100)
    assert all(len(chunk) <= 100 for chunk in chunks)
    assert "".join(chunks).replace("\n\n", "") == text.replace("\n\n", "")


def test_single_giant_block_hard_splits():
    text = "x = 1;" * 100  # no blank lines anywhere
    chunks = split_oversized(text=text, max_chars=50)
    assert all(len(chunk) <= 50 for chunk in chunks)
    assert "".join(chunks) == text
