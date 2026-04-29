import faiss
import numpy as np
from rank_bm25 import BM25Okapi


class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.chunks: list[dict] = []
        self._bm25: BM25Okapi | None = None

    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0

    def add(self, embeddings: np.ndarray, chunks: list[dict]) -> None:
        self.index.add(embeddings.astype("float32"))
        self.chunks.extend(chunks)
        tokenized = [c["text"].lower().split() for c in self.chunks]
        self._bm25 = BM25Okapi(tokenized)

    def _dense_ranked(self, query_embedding: np.ndarray, n: int) -> list[int]:
        _, indices = self.index.search(query_embedding.reshape(1, -1).astype("float32"), n)
        return [int(i) for i in indices[0] if i != -1]

    def _sparse_ranked(self, query_text: str, n: int) -> list[int]:
        scores = self._bm25.get_scores(query_text.lower().split())
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]

    @staticmethod
    def _rrf(*rank_lists: list[int], k: int = 60) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        for ranked in rank_lists:
            for rank, idx in enumerate(ranked):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def search(self, query_embedding: np.ndarray, query_text: str, top_k: int = 5) -> list[dict]:
        n = min(top_k * 3, len(self.chunks))
        fused = self._rrf(
            self._dense_ranked(query_embedding, n),
            self._sparse_ranked(query_text, n),
        )
        results = []
        for idx, score in fused[:top_k]:
            entry = dict(self.chunks[idx])
            entry["score"] = score
            results.append(entry)
        return results