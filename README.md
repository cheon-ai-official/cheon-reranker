# Cheon Reranker (체온 리랭커)

Reranking for search and RAG, from [CHEON:AI](https://cheon.ai.kr).

This repository has a cookbook of runnable recipes and the small Python
package they share. The weights are on Hugging Face.

| Model | Base | Parameters | Context | Candidates per call | License |
| --- | --- | --- | --- | --- | --- |
| [cheon-reranker-0.6b-v1](https://huggingface.co/cheonai/cheon-reranker-0.6b-v1) | Qwen3-0.6B | 609.2M | 12,288 tokens | 50 | CC-BY-NC-4.0 |

The model was built for legal and administrative document search. Try it in the
browser on the [playground](https://cheon.ai.kr/reranker#playground).

## Benchmarks

nDCG@10 on AutoRAGRetrieval from MTEB(kor): 114 questions over finance,
public-sector, healthcare, legal and commerce documents. Every model was run
from its public checkpoint under the same evaluation harness, and a score is
reported only if it reproduces the model's official number within ±0.003 where
one exists.

| Model | Parameters | nDCG@10 |
| --- | ---: | ---: |
| jina-reranker-v3.5 | 0.6B | 0.98381 |
| jina-reranker-v3 | 0.6B | 0.97734 |
| Qwen3-Reranker-4B | 4B | 0.97065 |
| **Cheon Reranker** | **0.609B** | **0.96648** |
| bge-reranker-v2-m3 | 0.568B | 0.96627 |
| Qwen3-Reranker-8B | 8B | 0.95462 |
| mxbai-rerank-large-v2 | 1.5B | 0.95311 |
| zerank-2 | 4.022B | 0.94361 |

At 0.609B parameters it ranks above Qwen3-Reranker-8B, zerank-2 and
mxbai-rerank-large-v2, models 2.5 to 13 times its size.

## Cookbook

| Recipe | What it shows |
| --- | --- |
| [00_quickstart/rerank_candidates.py](cookbook/00_quickstart/rerank_candidates.py) | One query, one call, a score per candidate |
| [00_quickstart/two_stage_retrieval.py](cookbook/00_quickstart/two_stage_retrieval.py) | BM25 finds ten passages, the reranker orders them |

```bash
git clone https://github.com/cheon-ai-official/cheon-reranker
cd cheon-reranker
pip install torch --index-url https://download.pytorch.org/whl/cpu   # skip on a CUDA machine
pip install -e ".[cookbook]"
python cookbook/00_quickstart/rerank_candidates.py
```

The first run downloads the weights (2.4 GB, float32). More in
[cookbook/README.md](cookbook/README.md).

## Quick start

```python
from cheon_reranker import CheonReranker

reranker = CheonReranker()  # cheonai/cheon-reranker-0.6b-v1 on CUDA, MPS or CPU

query = "이 사건의 처리 기한은 언제까지인가요?"
candidates = [
    "The appeal must be filed within 30 days of the decision.",
    "이의신청은 처분을 안 날부터 30일 이내에 제기해야 합니다.",
    "上诉必须在裁定后30天内提出。",
    "不服申立ては、決定を知った日から30日以内に行う必要があります。",
]

for result in reranker.rerank(query, candidates, top_n=3):
    print(f"{result.score:.3f}  {result.document}")
```

`rerank` returns the candidates best first, each with its `index` in your list,
its `score` and the `document` text; `score` returns the raw scores in input
order. The package also has the BM25 first stage the recipes use
(`cheon_reranker.bm25`, extra `[bm25]`).

## Example: AutoRAG

The two-stage recipe runs BM25 over Kiwi morphemes on the public
[AutoRAG retrieval set](https://huggingface.co/datasets/mteb/AutoRAGRetrieval)
(720 passages from finance, public-sector, legal and commerce documents) and
reranks the top ten. On its question about the live-commerce service with the
highest usage, BM25 ranks the labelled answer eighth, below a solution brochure
that shares many of the question's words; the reranker puts the answer first.
The write-up walks through it: [BM25 뒤에 체온 리랭커 붙이기: AutoRAG 실전 가이드](https://cheon.ai.kr/blog/autorag-reranker-ndcg-mrr) (Korean).

## Things to know

- A call holds the query and its candidates in 12,288 tokens. Long candidates
  are cut at the end, so rerank fewer, whole candidates rather than many cut
  ones.
- Pass candidates in your first stage's order.
- At most 50 candidates per call.
- Scores rank candidates within one call; they are not probabilities and are
  not comparable across calls.
- The weights are float32; expect more memory and latency than a half-precision
  model of this size.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The tests load the real model from Hugging Face.

## License

The code in this repository and the model weights are released under
[CC-BY-NC-4.0](LICENSE): research and evaluation, not commercial use. For a
commercial license or hosted API access, [contact us](https://cheon.ai.kr/contact).
