"""
Two-Stage Retrieval - BM25 Finds, the Reranker Orders
=====================================================
The common RAG setup: a cheap first stage pulls ten passages, the reranker
puts the one that answers the question first, and only the top few go to the
LLM.

Runs on the public AutoRAG retrieval set (720 passages from finance,
public-sector, legal and commerce documents) with three of its own questions
whose labelled answer BM25 ranks low. BM25 works on Kiwi morphemes, as in the
write-up: https://cheon.ai.kr/blog/autorag-reranker-ndcg-mrr
"""

from datasets import load_dataset

from cheon_reranker import CheonReranker
from cheon_reranker.bm25 import KiwiBM25

DATASET = "mteb/AutoRAGRetrieval"
QUESTIONS = ["99_commerce", "49_public", "72_law"]
TOP_K = 10  # passages the reranker sees per question
KEEP = 3  # passages that would go to the LLM

# ---------------------------------------------------------------------------
# Load the data
# ---------------------------------------------------------------------------
corpus = load_dataset(DATASET, "corpus", split="test")
queries = {row["_id"]: row["text"] for row in load_dataset(DATASET, "queries", split="test")}
answers = {row["query-id"]: row["corpus-id"] for row in load_dataset(DATASET, "qrels", split="test")}
doc_ids, texts = corpus["_id"], corpus["text"]

# ---------------------------------------------------------------------------
# Build the two stages
# ---------------------------------------------------------------------------
bm25 = KiwiBM25(texts)
reranker = CheonReranker()


def place(ranked_ids: list[str], doc_id: str) -> str:
    return f"#{ranked_ids.index(doc_id) + 1}" if doc_id in ranked_ids else "not in the list"


# ---------------------------------------------------------------------------
# Retrieve, rerank, compare
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for query_id in QUESTIONS:
        query, answer = queries[query_id], answers[query_id]
        hits = [index for index, _ in bm25.search(query, TOP_K)]
        candidates = [doc_ids[index] for index in hits]
        ranked = reranker.rerank(query, [texts[index] for index in hits])
        reranked = [candidates[result.index] for result in ranked]

        print(f"\n{query_id}: {query}")
        print(f"  labelled answer: BM25 {place(candidates, answer)} -> reranked {place(reranked, answer)}")
        print(f"  top {KEEP} for the LLM:")
        for result in ranked[:KEEP]:
            mark = "*" if candidates[result.index] == answer else " "
            print(f"    {mark} {result.score:7.3f}  {candidates[result.index]}")
