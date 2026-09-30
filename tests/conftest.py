import pytest

from cheon_reranker import CheonReranker

# The model card's usage example: one question and the same answer in four languages.
QUERY = "이 사건의 처리 기한은 언제까지인가요?"
CANDIDATES = [
    "The appeal must be filed within 30 days of the decision.",
    "이의신청은 처분을 안 날부터 30일 이내에 제기해야 합니다.",
    "上诉必须在裁定后30天内提出。",
    "不服申立ては、決定を知った日から30日以内に行う必要があります。",
]


@pytest.fixture(scope="session")
def reranker() -> CheonReranker:
    return CheonReranker(device="cpu")
