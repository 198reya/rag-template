from openai import OpenAI

from src.prompts import SINGLE_HOP_SYSTEM


def _build_context(results: list[dict]) -> str:
    blocks = []
    for r in results:
        blocks.append(f"[{r['source']}, p.{r['page']}]\n{r['text']}")
    return "\n\n---\n\n".join(blocks)


def _unique_sources(results: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    sources = []
    for r in results:
        key = (r["source"], r["page"])
        if key not in seen:
            seen.add(key)
            sources.append({"file": r["source"], "page": r["page"]})
    return sources


class Generator:
    def __init__(self, client: OpenAI, model: str, temperature: float, max_tokens: int):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, question: str, results: list[dict]) -> dict:
        system_prompt = SINGLE_HOP_SYSTEM.format(context=_build_context(results))

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": question},
            ],
        )
        return {
            "answer":  response.choices[0].message.content,
            "sources": _unique_sources(results),
        }