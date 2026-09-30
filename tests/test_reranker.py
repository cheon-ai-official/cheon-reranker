import pytest
from conftest import CANDIDATES, QUERY

from cheon_reranker.reranker import CANDIDATE_TEMPLATE, QUERY_TEMPLATE


def test_same_input_gives_the_same_scores(reranker):
    assert reranker.score(QUERY, CANDIDATES) == pytest.approx(reranker.score(QUERY, CANDIDATES), abs=1e-6)


def test_rerank_orders_by_score_and_keeps_top_n(reranker):
    scores = reranker.score(QUERY, CANDIDATES)
    ranked = reranker.rerank(QUERY, CANDIDATES)
    assert [result.index for result in ranked] == sorted(range(len(CANDIDATES)), key=lambda index: -scores[index])
    assert [result.document for result in ranked] == [CANDIDATES[result.index] for result in ranked]
    assert [result.score for result in ranked] == pytest.approx(sorted(scores, reverse=True))
    assert [result.index for result in reranker.rerank(QUERY, CANDIDATES, top_n=2)] == [r.index for r in ranked[:2]]


def test_encode_lays_out_query_then_candidates(reranker):
    ids, spans = reranker.encode(QUERY, CANDIDATES)
    query_ids = reranker.tokenizer(QUERY_TEMPLATE.format(QUERY), add_special_tokens=False)["input_ids"]
    assert ids[: len(query_ids)] == query_ids
    assert spans[0][0] == len(query_ids)
    assert all(end == start for (_, end), (start, _) in zip(spans, spans[1:], strict=False))
    assert spans[-1][1] == len(ids)
    for (start, end), text in zip(spans, CANDIDATES, strict=True):
        written = reranker.tokenizer(CANDIDATE_TEMPLATE.format(text), add_special_tokens=False)["input_ids"]
        assert ids[start:end] == written


def test_long_candidates_share_the_context_equally(reranker):
    long_text = " ".join([CANDIDATES[1]] * 2000)
    documents = [long_text, CANDIDATES[0], long_text]
    ids, spans = reranker.encode(QUERY, documents)
    query_tokens = spans[0][0]
    budget = (reranker.max_tokens - query_tokens) // len(documents)
    assert len(ids) <= reranker.max_tokens
    assert [end - start for start, end in spans][0] == budget
    assert [end - start for start, end in spans][2] == budget
    assert len(reranker.score(QUERY, documents)) == len(documents)


@pytest.mark.parametrize(
    ("query", "documents"),
    [
        ("", CANDIDATES),
        ("   ", CANDIDATES),
        (QUERY, []),
        (QUERY, "not a list"),
        (QUERY, [CANDIDATES[0], "  "]),
    ],
)
def test_rejects_unusable_input(reranker, query, documents):
    with pytest.raises(ValueError):
        reranker.score(query, documents)


def test_rejects_more_candidates_than_the_model_takes(reranker):
    with pytest.raises(ValueError, match=f"at most {reranker.max_candidates}"):
        reranker.score(QUERY, [CANDIDATES[0]] * (reranker.max_candidates + 1))
