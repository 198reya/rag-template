from openai import OpenAI

from src.prompts import MULTIHOP_FOLLOWUP, MULTIHOP_SYNTHESIS, QUERY_REWRITE


def _call(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 128,
) -> str:
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def _rewrite_query(client: OpenAI, model: str, question: str) -> str:
    return _call(client, model, QUERY_REWRITE.format(question=question))


def _next_query(client: OpenAI, model: str, question: str, facts: list[dict]) -> str | None:
    facts_text = "\n".join(
        f"- [{f['source']}, p.{f['page']}] {f['text'][:300]}" for f in facts
    )
    reply = _call(
        client,
        model,
        MULTIHOP_FOLLOWUP.format(question=question, facts=facts_text or "None yet."),
    )
    return None if reply.upper() == "DONE" else reply


def _synthesize(
    client: OpenAI,
    model: str,
    question: str,
    facts: list[dict],
    temperature: float,
    max_tokens: int,
) -> str:
    facts_text = "\n".join(
        f"- [{f['source']}, p.{f['page']}] {f['text']}" for f in facts
    )
    return _call(
        client,
        model,
        MULTIHOP_SYNTHESIS.format(question=question, facts=facts_text),
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _unique_sources(chunks: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    out = []
    for c in chunks:
        key = (c["source"], c["page"])
        if key not in seen:
            seen.add(key)
            out.append({"file": c["source"], "page": c["page"]})
    return out


def multihop_query(
    question: str,
    embedder,
    store,
    client: OpenAI,
    model: str,
    top_k: int = 5,
    max_hops: int = 3,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    verbose: bool = False,
) -> dict:
    all_chunks: list[dict] = []
    seen_keys: set[tuple] = set()

    search_query = _rewrite_query(client, model, question)
    if verbose:
        print(f"  [rewrite] {search_query}")

    for hop in range(1, max_hops + 1):
        if verbose:
            print(f"  [hop {hop}] {search_query}")

        q_emb = embedder.embed([search_query])[0]
        results = store.search(q_emb, search_query, top_k=top_k)

        for r in results:
            key = (r["source"], r["page"], r.get("chunk_index"))
            if key not in seen_keys:
                all_chunks.append(r)
                seen_keys.add(key)

        if hop == max_hops:
            break

        follow_up = _next_query(client, model, question, all_chunks)
        if follow_up is None:
            if verbose:
                print(f"  [hop {hop}] sufficient context — stopping.")
            break

        search_query = follow_up

    return {
        "answer": _synthesize(client, model, question, all_chunks, temperature, max_tokens),
        "sources": _unique_sources(all_chunks),
        "hops": hop,
    }