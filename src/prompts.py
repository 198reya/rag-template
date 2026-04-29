QUERY_REWRITE = """\
Rewrite the following user question into a short, precise search query \
optimised for retrieving relevant document chunks.

Rules:
- Remove conversational filler ("can you tell me", "I was wondering", etc.)
- Keep named entities, dates, and specific terms exactly as written
- Return only the rewritten query, nothing else

Question: {question}"""


MULTIHOP_FOLLOWUP = """\
You are helping answer a complex question that may require multiple retrieval steps.

Original question: {question}

Facts gathered so far:
{facts}

Do the facts above contain enough information to fully answer the original question?

- If YES, reply with exactly: DONE
- If NO, reply with the single most useful follow-up search query that would \
fill the biggest remaining gap. Reply with only the query, nothing else."""


MULTIHOP_SYNTHESIS = """\
Answer the following question using ONLY the facts provided below.
If the facts are insufficient, say exactly: \
"I don't have enough information in the provided documents to answer that."

Rules:
- Do not use any outside knowledge.
- Cite each fact inline as [filename, p.N].
- Be concise.

Question: {question}

Facts:
{facts}"""


SINGLE_HOP_SYSTEM = """\
You are a precise, helpful assistant. Answer the user's question using ONLY \
the information in the context blocks below.

Rules:
1. If the context does not contain enough information to answer, respond with exactly:
   "I don't have enough information in the provided documents to answer that."
2. Do not use any outside knowledge.
3. When a fact comes from a specific block, cite it inline as [filename, p.N].
4. Be concise.

Context:
{context}"""