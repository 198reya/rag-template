import os

import yaml
from dotenv import load_dotenv
from openai import OpenAI

from src.embedder import Embedder
from src.ingest import chunk_documents, load_documents
from src.multihop import multihop_query
from src.vector_store import VectorStore

load_dotenv()


def _load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class RAGPipeline:
    def __init__(self, config_path: str = "config.yaml"):
        cfg = _load_config(config_path)

        self._chunking   = cfg["chunking"]
        self._retrieval  = cfg["retrieval"]
        self._llm        = cfg["llm"]
        self._data_dir   = cfg["data"]["documents_dir"]

        self.embedder = Embedder(cfg["embedding"]["model"])
        self.store    = VectorStore(dim=cfg["embedding"]["dim"])
        self.client   = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    def ingest(self, directory: str | None = None) -> None:
        directory = directory or self._data_dir
        docs = load_documents(directory)

        if not docs:
            print(f"No documents found in '{directory}'. Add .txt, .pdf, or .md files.")
            return

        chunks = chunk_documents(
            docs,
            chunk_size=self._chunking["chunk_size"],
            chunk_overlap=self._chunking["chunk_overlap"],
            separators=self._chunking["separators"],
        )

        embeddings = self.embedder.embed([c["text"] for c in chunks])
        self.store.add(embeddings, chunks)

        unique_files = len({c["source"] for c in chunks})
        print(f"Indexed {len(chunks)} chunks from {unique_files} file(s).")

    def query(self, question: str, verbose: bool = False) -> dict:
        if self.store.is_empty:
            return {"answer": "No documents indexed yet. Run ingest() first.", "sources": []}

        return multihop_query(
            question=question,
            embedder=self.embedder,
            store=self.store,
            client=self.client,
            model=self._llm["model"],
            top_k=self._retrieval["top_k"],
            max_hops=self._retrieval.get("max_hops", 3),
            temperature=self._llm["temperature"],
            max_tokens=self._llm["max_tokens"],
            verbose=verbose,
        )