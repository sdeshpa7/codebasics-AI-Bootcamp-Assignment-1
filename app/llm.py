"""
app/llm.py
Groq-powered query answering for the FinSolve RAG pipeline.

Responsibilities:
  - Build a role-aware system prompt that tells the LLM what the user can access
  - Format retrieved chunks as numbered context passages with source citations
  - Call Groq's chat completions API
  - Return the raw answer (output guardrails are applied by the caller)
"""

from groq import Groq

from app.config import settings
from app.embeddings import EmbeddedChunk

# ---------------------------------------------------------------------------
# Groq client (lazy singleton)
# ---------------------------------------------------------------------------

_client: Groq | None = None


def get_client() -> Groq:
    """Return a shared Groq client, initialised on first call."""
    global _client
    if _client is None:
        if not settings.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file: GROQ_API_KEY=your_key_here"
            )
        _client = Groq(api_key=settings.groq_api_key)
    return _client


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = """You are FinSolve's internal knowledge assistant — a precise, professional AI that answers employee questions using only the provided context documents.

## Your role
- The authenticated user has the following access roles: {roles}
- You may ONLY reference information that appears in the context passages below.
- You must NEVER fabricate figures, dates, or claims not present in the context.
- You must ALWAYS cite your sources using the format: (Source: <filename>, Page <page_number>)

## Behaviour rules
1. Answer concisely and factually. Use bullet points or tables where helpful.
2. If the context does not contain enough information to answer, say so clearly — do not guess.
3. Do not reveal information from documents the user is not authorised to see.
4. End every response with a "Sources" section listing each document and page you referenced.

## Context passages
{context}
"""

_NO_CONTEXT_MSG = (
    "I'm sorry, I couldn't find any relevant documents in the knowledge base "
    "to answer your question. Please try rephrasing, or contact your administrator "
    "if you believe you should have access to this information."
)


def _build_context(chunks: list) -> str:
    """Format retrieved chunks as numbered context passages."""
    if not chunks:
        return "(No relevant context found.)"

    parts = []
    for i, c in enumerate(chunks, 1):
        # Support both EmbeddedChunk and raw Chunk objects
        text        = c.text if hasattr(c, "text") else str(c)
        source_file = (c.chunk.source_file if isinstance(c, EmbeddedChunk)
                       else getattr(c, "source_file", "unknown"))
        page        = (c.chunk.page_number  if isinstance(c, EmbeddedChunk)
                       else getattr(c, "page_number",  "?"))
        section     = (c.chunk.section_title if isinstance(c, EmbeddedChunk)
                       else getattr(c, "section_title", None))

        header = f"[{i}] Source: {source_file}, Page {page}"
        if section:
            header += f" | Section: {section}"

        parts.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def answer_query(
    query: str,
    retrieved_chunks: list,
    user_access_roles: list[str],
    model: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.1,
) -> str:
    """
    Generate an answer to ``query`` using Groq and the provided context chunks.

    Args:
        query:             The user's question (already cleared by input guardrails).
        retrieved_chunks:  Chunks from Qdrant (EmbeddedChunk or Chunk objects).
        user_access_roles: Roles of the authenticated user, included in the prompt.
        model:             Groq model name. Defaults to settings.groq_model.
        max_tokens:        Maximum tokens in the completion.
        temperature:       Sampling temperature (low = more deterministic).

    Returns:
        The LLM's answer string (pass to apply_output_guardrails before returning to user).
    """
    if model is None:
        model = settings.groq_model
    if not retrieved_chunks:
        return _NO_CONTEXT_MSG

    context = _build_context(retrieved_chunks)
    system_prompt = _SYSTEM_TEMPLATE.format(
        roles=", ".join(user_access_roles),
        context=context,
    )

    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": query},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    return response.choices[0].message.content
