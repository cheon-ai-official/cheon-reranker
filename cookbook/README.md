# Cheon Reranker Cookbook

A few recipes for 체온 리랭커 (Cheon Reranker). Copy, paste, run.

Every recipe runs on its own and on public data: the model card's example or
the [AutoRAG retrieval dataset](https://huggingface.co/datasets/mteb/AutoRAGRetrieval).
From the repository root:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # skip on a CUDA machine
pip install -e ".[cookbook]"
python cookbook/00_quickstart/rerank_candidates.py
```

The first run downloads the model (2.4 GB) and, for the two-stage recipe, the
dataset.

## Where to Start

**New to the reranker?** Start with [00_quickstart](./00_quickstart): score a
list of candidates, then put the reranker behind BM25 the way a RAG pipeline
does.

The write-up [BM25가 8위에 둔 정답을 체온 리랭커는 1위로 올렸어요](https://cheon.ai.kr/blog/autorag-reranker-ndcg-mrr) (Korean) walks through the two-stage example.

| Folder | Recipe | What it shows |
| --- | --- | --- |
| [00_quickstart](./00_quickstart) | [`rerank_candidates.py`](./00_quickstart/rerank_candidates.py) | One query, one call, a score per candidate |
| [00_quickstart](./00_quickstart) | [`two_stage_retrieval.py`](./00_quickstart/two_stage_retrieval.py) | BM25 finds ten passages, the reranker orders them |
