from pipeline.embeddings import MAX_BATCH_TEXTS, MAX_BATCH_TOKENS, _batches


def test_batches_respect_text_count_cap():
    batches = _batches(texts=["x" for _ in range(300)])
    assert all(len(batch) <= MAX_BATCH_TEXTS for batch in batches)
    assert sum(len(batch) for batch in batches) == 300


def test_batches_respect_token_budget():
    # 200 texts of ~2000 estimated tokens each — count cap alone would allow
    # 128 per batch, far over the token budget.
    texts = ["a" * 4000 for _ in range(200)]
    batches = _batches(texts=texts)
    for batch in batches:
        assert sum(len(text) // 2 for text in batch) <= MAX_BATCH_TOKENS
    assert sum(len(batch) for batch in batches) == 200


def test_single_oversized_text_still_emitted():
    batches = _batches(texts=["a" * (MAX_BATCH_TOKENS * 2 + 100)])
    assert len(batches) == 1
