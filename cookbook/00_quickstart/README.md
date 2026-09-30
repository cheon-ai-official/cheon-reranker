# Quickstart

Score candidates for a query, then use the reranker as the second stage of a
retrieval pipeline.

## Start Here

From the repository root:

```bash
pip install -e ".[cookbook]"
python cookbook/00_quickstart/rerank_candidates.py
```

The minimum:

```python
from cheon_reranker import CheonReranker

reranker = CheonReranker()
for result in reranker.rerank("이 사건의 처리 기한은 언제까지인가요?", candidates, top_n=3):
    print(result.score, result.document)
```

`rerank` returns the candidates best first, each with its `index` in your
list, its `score` and the `document` text. Scores rank candidates within one
call; they are not probabilities.

## Recipes

| # | Recipe | What you learn | What it prints |
|---:|:---|:---|:---|
| 01 | [`rerank_candidates.py`](rerank_candidates.py) | The call: query and candidates in, scores out | The model card's four candidates, best first |
| 02 | [`two_stage_retrieval.py`](two_stage_retrieval.py) | BM25 over Kiwi morphemes, then the reranker on the top ten | For three AutoRAG questions, where BM25 and the reranker place the labelled answer, and the top three |

The questions in `two_stage_retrieval.py` are three of AutoRAG's own whose
labelled answer BM25 ranks low (`99_commerce`, `49_public`, `72_law`); the
write-up looks at why: https://cheon.ai.kr/blog/autorag-reranker-ndcg-mrr
