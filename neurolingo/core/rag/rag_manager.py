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

from dataclasses import dataclass, field

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


# ── Multi-turn follow-ups ─────────────────────────────────────────────────────

@dataclass
class TutorConversation:
    """
    An ongoing, RAG-grounded dialogue about one target sentence.

    Created via RAGManager.start_conversation(), which seeds `messages` with
    the same retrieved-context prompt tutor_analyze_mistake() uses for its
    one-shot answer. Call send() with no argument to run that seeded first
    turn, then send(text) again for each follow-up the student asks — the
    full history (and the original RAG context, baked into the first turn)
    rides along via LLMRouter.chat() every time.
    """

    router: LLMRouter
    system_prompt: str
    messages: list[dict[str, str]] = field(default_factory=list)

    async def send(self, text: str | None = None) -> str:
        if text:
            self.messages.append({"role": "user", "content": text})
        try:
            response = await self.router.chat(self.messages, system=self.system_prompt)
        except Exception:
            # Roll back the just-added, unanswered user turn on failure so the
            # history stays strictly user/assistant-alternating for the next
            # attempt — some providers (e.g. Anthropic) reject a request
            # where two user turns appear back-to-back with no reply between.
            if text:
                self.messages.pop()
            raise
        self.messages.append({"role": "assistant", "content": response})
        return response


# ── Manager ───────────────────────────────────────────────────────────────────

def format_knowledge_entry(sentence_en: str, notes: str) -> str:
    """Build the RAG knowledge-base text for a sentence.

    Shared by seeding, manually saving a new sentence, and restoring a
    backup, so all three ways a sentence enters the app end up indexed the
    same way.
    """
    return f"{sentence_en} — {notes}" if notes else sentence_en


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
        prompt, num_contexts = self._build_seed_prompt(target_sentence, user_input)
        _log.info(
            "tutor_analyze_mistake | target=%.50s | contexts=%d",
            target_sentence, num_contexts,
        )
        return await self._router.generate_explanation(prompt, system=_TUTOR_SYSTEM_PROMPT)

    def start_conversation(self, target_sentence: str, user_input: str) -> TutorConversation:
        """
        Seed a follow-up-capable AI tutor conversation, grounded in the same
        retrieved context tutor_analyze_mistake() uses for its one-shot
        answer.

        Call TutorConversation.send() with no argument to get the first
        response, then send(text) again for each follow-up the student asks
        (e.g. "why?", "give me another example") — unlike
        tutor_analyze_mistake(), the conversation remembers everything said
        so far.
        """
        prompt, num_contexts = self._build_seed_prompt(target_sentence, user_input)
        _log.info(
            "start_conversation | target=%.50s | contexts=%d",
            target_sentence, num_contexts,
        )
        return TutorConversation(
            router=self._router,
            system_prompt=_TUTOR_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

    def _build_seed_prompt(self, target_sentence: str, user_input: str) -> tuple[str, int]:
        """Retrieve context and build the first-turn prompt shared by both
        tutor_analyze_mistake() and start_conversation(). Returns
        (prompt, number_of_contexts_retrieved)."""
        query = f"{target_sentence} {user_input}"
        contexts = self.retrieve(query)
        rag_context = "\n\n".join(
            f"[Context {i + 1}] {c['text']}"
            for i, c in enumerate(contexts)
        )
        prompt = _build_tutor_prompt(target_sentence, user_input, rag_context)
        return prompt, len(contexts)
