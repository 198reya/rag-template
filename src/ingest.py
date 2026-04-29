from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


def _load_txt(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [{"text": f.read(), "page": 1}]


def _load_md(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [{"text": f.read(), "page": 1}]


def _load_pdf(path: str) -> list[dict]:
    from pypdf import PdfReader
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"text": text, "page": i})
    return pages


_LOADERS = {
    ".txt": _load_txt,
    ".md":  _load_md,
    ".pdf": _load_pdf,
}


def load_documents(directory: str) -> list[dict]:
    docs = []
    for file in sorted(Path(directory).rglob("*")):
        loader = _LOADERS.get(file.suffix.lower())
        if loader is None:
            continue
        try:
            pages = loader(str(file))
        except Exception as exc:
            print(f"  Warning: could not read {file.name}: {exc}")
            continue
        for p in pages:
            if p["text"].strip():
                docs.append({"text": p["text"], "source": file.name, "page": p["page"]})
    return docs


def chunk_documents(
    docs: list[dict],
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str],
) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )
    chunks = []
    for doc in docs:
        for i, text in enumerate(splitter.split_text(doc["text"])):
            chunks.append({
                "text": text,
                "source": doc["source"],
                "page": doc["page"],
                "chunk_index": i,
            })
    return chunks