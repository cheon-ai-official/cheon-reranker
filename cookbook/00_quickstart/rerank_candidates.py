"""
Rerank Candidates - One Query, One Call
=======================================
Start here. One call takes the query and its candidates and returns a score
for each, best first.

The example is the model card's own: one question and the same answer in
English, Korean, Chinese and Japanese. Candidates do not need to share the
query's language.
"""

from cheon_reranker import CheonReranker

# ---------------------------------------------------------------------------
# Load the model
# ---------------------------------------------------------------------------
reranker = CheonReranker()  # cheonai/cheon-reranker-0.6b-v1 on CUDA, MPS or CPU

query = "이 사건의 처리 기한은 언제까지인가요?"
candidates = [
    "The appeal must be filed within 30 days of the decision.",
    "이의신청은 처분을 안 날부터 30일 이내에 제기해야 합니다.",
    "上诉必须在裁定后30天内提出。",
    "不服申立ては、決定を知った日から30日以内に行う必要があります。",
]

# ---------------------------------------------------------------------------
# Rerank
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"query: {query}  ({reranker.model_name} on {reranker.device})\n")
    for place, result in enumerate(reranker.rerank(query, candidates), start=1):
        print(f"{place}. {result.score:7.3f}  [{result.index}] {result.document}")
