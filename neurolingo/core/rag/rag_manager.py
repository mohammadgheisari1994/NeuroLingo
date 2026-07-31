"""
RAGManager — orchestrates retrieval-augmented generation for the AI tutor.

Workflow for tutor_analyze_mistake():
  1. Embed the user's mistake context (target + user_input).
  2. Retrieve the top-k most relevant grammar rules / past mistakes from the
     VectorStore.
  3. Inject the retrieved context into a neuroscience-informed system prompt.
  4. Route the enriched prompt through the LLMRouter.

All retrieval and prompting logic lives here so the router and vector store
remain generic and reusable.
"""
from __future__ import annotations

from logger_config import get_logger
from neurolingo.core.llm.router import LLMRouter
from neurolingo.core.rag.base import EmbeddingProvider, VectorStore

_log = get_logger(__name__)

# ── Prompting ─────────────────────────────────────────────────────────────────

_TUTOR_SYSTEM_PROMPT = """\
You are NeuroLingo, an expert English language tutor for Farsi speakers.

Apply these neuroscience-backed teaching principles in every response:
1. HEBBIAN CHUNKING: NEVER explain grammar in isolation. Anchor every
   explanation to the specific target sentence the student was studying.
2. CONTEXTUAL REPAIR: Show the corrected form first, then explain WHY.
3. COMPREHENSIBLE INPUT: Use plain language the student already understands.
   Avoid dense jargon. A brief Farsi clarification of the key concept is
   welcome when the English term alone would be confusing.

Keep your response concise (under 120 words) and warm in tone.
"""


def _build_tutor_prompt(
    target_sentence: str,
    user_input: str,
    rag_context: str,
) -> str:
    context_block = (
        f"\n\nRelevant grammar context:\n{rag_context}" if rag_context.strip() else ""
    )
    return (
        f"Target sentence: \"{target_sentence}\"\n"
        f"Student's attempt: \"{user_input}\"\n"
        f"{context_block}\n\n"
        "Please analyse the student's mistake and explain how to correct it."
    )


# ── Manager ───────────────────────────────────────────────────────────────────

class RAGManager:
    """
    High-level interface used by the UI and SRS review flow.

    Args:
        vector_store:       Indexed knowledge base (grammar rules, user mistakes).
        embedding_provider: Converts text → vectors for indexing and retrieval.
        llm_router:         Routes the enriched prompt to the best available LLM.
        top_k:              Maximum number of retrieved contexts to inject.
        min_similarity:     Contexts below this cosine threshold are dropped.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        llm_router: LLMRouter,
        top_k: int = 3,
        min_similarity: float = 0.5,
    ) -> None:
        self._store = vector_store
        self._embedder = embedding_provider
        self._router = llm_router
        self._top_k = top_k
        self._min_similarity = min_similarity

    # ── Knowledge base ────────────────────────────────────────────────────────

    def add_knowledge(self, text: str, metadata: dict | None = None) -> str:
        """Index one piece of knowledge (e.g. a grammar rule or past mistake)."""
        vec = self._embedder.embed(text)
        doc_id = self._store.add(vec, text, metadata)
        _log.debug("Knowledge indexed | id=%s | text=%.60s", doc_id, text)
        return doc_id

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """Return the most relevant documents for an arbitrary query string."""
        k = top_k if top_k is not None else self._top_k
        vec = self._embedder.embed(query)
        results = self._store.search(vec, top_k=k, min_similarity=self._min_similarity)
        _log.debug(
            "RAG retrieved %d contexts with similarity > %.2f",
            len(results), self._min_similarity,
        )
        return results

    # ── Tutor entrypoint ──────────────────────────────────────────────────────

    async def tutor_analyze_mistake(
        self,
        target_sentence: str,
        user_input: str,
    ) -> str:
        """
        Core AI tutoring method.

        Given the correct target sentence and the student's attempt, retrieve
        relevant grammatical context and produce a neuroscience-informed
        correction explanation via the LLM router.

        Args:
            target_sentence: The sentence the student was supposed to produce.
            user_input:       What the student actually typed or said.

        Returns:
            A short, contextualised correction explanation from the LLM.
        """
        query = f"{target_sentence} {user_input}"
        contexts = self.retrieve(query)

        rag_context = "\n\n".join(
            f"[Context {i + 1}] {c['text']}"
            for i, c in enumerate(contexts)
        )

        prompt = _build_tutor_prompt(target_sentence, user_input, rag_context)

        _log.info(
            "tutor_analyze_mistake | target=%.50s | contexts=%d",
            target_sentence, len(contexts),
        )

        return await self._router.generate_explanation(prompt, system=_TUTOR_SYSTEM_PROMPT)
