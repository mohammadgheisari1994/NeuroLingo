"""
NeuroLingo — main entry point & demo UI.

Wires together all built modules:
  • DatabaseRepository  (SQLite, FK-safe)
  • SM-2 algorithm      (spaced repetition)
  • LLMRouter + RAGManager (AI tutor: cloud LLMs with local llama.cpp fallback,
    grounded by a hashing-embedding NumPy vector store)

Run:
    python3.13 main.py
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import flet as ft

from logger_config import get_logger, setup_logging
from neurolingo.core.llm.base import LLMConfig, ProviderError
from neurolingo.core.llm.router import LLMRouter
from neurolingo.core.rag.embeddings import HashingEmbeddingProvider
from neurolingo.core.rag.rag_manager import RAGManager, TutorConversation
from neurolingo.core.rag.vectorstore import NumpyVectorStore
from neurolingo.core.srs.algorithm import (
    GRADE_AGAIN,
    GRADE_EASY,
    GRADE_GOOD,
    GRADE_HARD,
    apply_sm2,
)
from neurolingo.db.models import Card, CardStatus, ReviewLog, Sentence
from neurolingo.db.repository import DatabaseRepository

log = setup_logging()
_log = get_logger(__name__)

DB_PATH = Path("data/neurolingo.db")
VECTOR_STORE_PATH = Path("data/knowledge")

# ── Sample sentences loaded on first run ──────────────────────────────────────

_SAMPLES: list[tuple[str, str, str]] = [
    (
        "She has been waiting for the bus since morning.",
        "او از صبح منتظر اتوبوس بوده است.",
        "Present perfect continuous — ongoing action from past to now",
    ),
    (
        "If I were you, I would study harder every day.",
        "اگر جای تو بودم، هر روز بیشتر درس می‌خواندم.",
        "Second conditional — hypothetical present/future situation",
    ),
    (
        "The cake was eaten by the children before dinner.",
        "کیک قبل از شام توسط بچه‌ها خورده شد.",
        "Passive voice — past simple passive construction",
    ),
    (
        "Despite the heavy rain, they went for a long walk.",
        "علی‌رغم باران شدید، آن‌ها برای پیاده‌روی طولانی رفتند.",
        "Contrast connector — despite + noun phrase",
    ),
    (
        "By the time you arrive, I will have finished cooking dinner.",
        "تا وقتی که تو برسی، من پختن شام را تمام کرده‌ام.",
        "Future perfect — action completed before a future reference point",
    ),
    (
        "The more you practise speaking, the more confident you become.",
        "هر چه بیشتر تمرین صحبت کنی، اعتماد به نفس بیشتری پیدا می‌کنی.",
        "Double comparative — the more... the more structure",
    ),
]


def _seed(repo: DatabaseRepository) -> None:
    """Populate DB with sample sentences on first run only."""
    if repo.get_new_cards(limit=1) or repo.get_due_cards(limit=1):
        return
    for en, fa, notes in _SAMPLES:
        s = repo.add_sentence(Sentence(sentence_en=en, sentence_fa=fa, context_notes=notes))
        repo.add_card(Card(
            sentence_id=s.id,
            next_review_date=datetime.now(timezone.utc) - timedelta(seconds=1),
            status=CardStatus.NEW.value,
        ))
    _log.info("Seeded %d sample sentences into fresh database", len(_SAMPLES))


def _seed_knowledge(rag: RAGManager) -> None:
    """Index each sample sentence's grammar note so the AI tutor has real
    context to retrieve instead of answering with no grounding at all."""
    for en, _fa, notes in _SAMPLES:
        rag.add_knowledge(f"{en} — {notes}", metadata={"source": "seed"})
    _log.info("Indexed %d grammar notes into the knowledge base", len(_SAMPLES))


# ── Colour palette ────────────────────────────────────────────────────────────

_AGAIN = ft.Colors.RED_400
_HARD = ft.Colors.ORANGE_400
_GOOD = ft.Colors.BLUE_400
_EASY = ft.Colors.GREEN_400
_SURFACE = ft.Colors.with_opacity(0.08, ft.Colors.WHITE)
_ACCENT = ft.Colors.INDIGO_300


# ── Helper: Padding shorthands ────────────────────────────────────────────────

def _pad_all(n: int) -> ft.Padding:
    return ft.Padding(top=n, bottom=n, left=n, right=n)


def _pad_sym(*, v: int = 0, h: int = 0) -> ft.Padding:
    return ft.Padding(top=v, bottom=v, left=h, right=h)


# ── App ───────────────────────────────────────────────────────────────────────

class NeuroLingoApp:
    """Stateful single-page application controller."""

    TAB_HOME = 0
    TAB_REVIEW = 1
    TAB_ADD = 2

    def __init__(self, page: ft.Page, repo: DatabaseRepository, rag: RAGManager) -> None:
        self.page = page
        self.repo = repo
        self.rag = rag
        self._tab = self.TAB_HOME
        self._current_card: Card | None = None
        self._current_sentence: Sentence | None = None
        self._show_translation = False
        self._advance_task: asyncio.Task | None = None
        self._tutor_conversation: TutorConversation | None = None
        self._setup_page()
        self._build_ui()
        self._refresh_dashboard()

    # ── Page configuration ────────────────────────────────────────────────────

    def _setup_page(self) -> None:
        p = self.page
        p.title = "NeuroLingo"
        p.theme_mode = ft.ThemeMode.DARK
        p.theme = ft.Theme(color_scheme_seed=ft.Colors.INDIGO)
        p.window.width = 480
        p.window.height = 860
        p.window.min_width = 380
        p.padding = 0
        p.bgcolor = ft.Colors.GREY_900

    # ── UI skeleton ───────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # ── AppBar ──
        self._appbar = ft.AppBar(
            leading=ft.Icon(ft.Icons.PSYCHOLOGY, color=_ACCENT, size=28),
            leading_width=48,
            title=ft.Text("NeuroLingo", weight=ft.FontWeight.BOLD, size=20),
            center_title=False,
            bgcolor=ft.Colors.GREY_900,
            actions=[
                ft.IconButton(
                    ft.Icons.INFO_OUTLINE,
                    tooltip="Hebbian chunking · SM-2 spaced repetition · offline-first AI tutor",
                    icon_color=_ACCENT,
                )
            ],
        )

        # ── Content panels (one per tab) ──
        self._home_panel = self._build_home_panel()
        self._review_panel = self._build_review_panel()
        self._add_panel = self._build_add_panel()

        self._body = ft.Container(
            content=self._home_panel,
            expand=True,
            padding=_pad_all(16),
        )

        # ── Bottom NavigationBar ──
        self.page.navigation_bar = ft.NavigationBar(
            selected_index=self.TAB_HOME,
            bgcolor=ft.Colors.GREY_800,
            destinations=[
                ft.NavigationBarDestination(
                    icon=ft.Icons.HOME_OUTLINED,
                    selected_icon=ft.Icons.HOME,
                    label="Home",
                ),
                ft.NavigationBarDestination(
                    icon=ft.Icons.SCHOOL_OUTLINED,
                    selected_icon=ft.Icons.SCHOOL,
                    label="Review",
                ),
                ft.NavigationBarDestination(
                    icon=ft.Icons.ADD_CIRCLE_OUTLINE,
                    selected_icon=ft.Icons.ADD_CIRCLE,
                    label="Add",
                ),
            ],
            on_change=self._on_tab_change,
        )
        self.page.appbar = self._appbar
        self.page.add(self._body)

    # ── Tab switching ─────────────────────────────────────────────────────────

    def _on_tab_change(self, e: ft.ControlEvent) -> None:
        self._tab = int(e.data)
        panels = {
            self.TAB_HOME: self._home_panel,
            self.TAB_REVIEW: self._review_panel,
            self.TAB_ADD: self._add_panel,
        }
        if self._tab == self.TAB_HOME:
            self._refresh_dashboard()
        elif self._tab == self.TAB_REVIEW:
            self._load_next_card()
        self._body.content = panels[self._tab]
        self.page.update()

    # ══════════════════════════════════════════════════════════════════════════
    # HOME tab
    # ══════════════════════════════════════════════════════════════════════════

    def _build_home_panel(self) -> ft.Column:
        self._stat_due = ft.Text("—", size=32, weight=ft.FontWeight.BOLD, color=_ACCENT)
        self._stat_new = ft.Text("—", size=32, weight=ft.FontWeight.BOLD, color=_EASY)
        self._stat_total = ft.Text("—", size=32, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE70)

        stats_row = ft.Row(
            [
                self._stat_card("Due", self._stat_due, ft.Icons.TIMER_OUTLINED, _ACCENT),
                self._stat_card("New", self._stat_new, ft.Icons.FIBER_NEW_OUTLINED, _EASY),
                self._stat_card("Total", self._stat_total, ft.Icons.LIBRARY_BOOKS_OUTLINED, ft.Colors.WHITE54),
            ],
            spacing=8,
            alignment=ft.MainAxisAlignment.CENTER,
        )

        self._start_btn = ft.FilledButton(
            "Start Review Session",
            icon=ft.Icons.PLAY_ARROW_ROUNDED,
            style=ft.ButtonStyle(
                bgcolor=_ACCENT,
                color=ft.Colors.WHITE,
                padding=_pad_sym(v=14, h=24),
                shape=ft.RoundedRectangleBorder(radius=12),
            ),
            on_click=self._go_to_review,
            expand=True,
        )

        self._recent_list = ft.Column(spacing=8, scroll=ft.ScrollMode.AUTO)

        return ft.Column(
            [
                ft.Text("Welcome back!", size=22, weight=ft.FontWeight.W_600),
                ft.Text(
                    "Sentences reviewed in context — no isolated flashcards.",
                    size=12, color=ft.Colors.WHITE54,
                ),
                ft.Divider(height=20, color=ft.Colors.WHITE12),
                stats_row,
                ft.Container(height=16),
                ft.Row([self._start_btn]),
                ft.Container(height=20),
                ft.Text("Your Sentences", size=16, weight=ft.FontWeight.W_600),
                ft.Divider(height=8, color=ft.Colors.WHITE12),
                self._recent_list,
            ],
            spacing=8,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        )

    def _stat_card(self, label: str, value_widget: ft.Text, icon, color) -> ft.Container:
        return ft.Container(
            content=ft.Column(
                [
                    ft.Icon(icon, color=color, size=22),
                    value_widget,
                    ft.Text(label, size=11, color=ft.Colors.WHITE54),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=2,
            ),
            bgcolor=_SURFACE,
            border_radius=12,
            padding=_pad_sym(v=12, h=8),
            expand=True,
        )

    def _refresh_dashboard(self) -> None:
        due = self.repo.get_due_cards(limit=100)
        new = self.repo.get_new_cards(limit=100)

        due_count = len(due)
        new_count = len(new)

        self._stat_due.value = str(due_count)
        self._stat_new.value = str(new_count)
        self._stat_total.value = str(self.repo.count_sentences())

        # Populate recent sentence list
        all_due = self.repo.get_due_cards(limit=20)
        all_new = self.repo.get_new_cards(limit=20)
        combined = all_due[:3] + all_new[:3]

        self._recent_list.controls.clear()
        for card in combined[:6]:
            sentence = self.repo.get_sentence(card.sentence_id)
            if not sentence:
                continue
            self._recent_list.controls.append(
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Text(sentence.sentence_en, size=13, weight=ft.FontWeight.W_500),
                            ft.Row([
                                ft.Container(
                                    content=ft.Text(
                                        card.status.upper(), size=10, color=ft.Colors.WHITE,
                                    ),
                                    bgcolor=self._status_color(card.status),
                                    border_radius=4,
                                    padding=_pad_sym(v=2, h=6),
                                ),
                                ft.Text(f"Interval: {card.interval}d", size=11, color=ft.Colors.WHITE54),
                            ], spacing=8),
                        ],
                        spacing=4,
                    ),
                    bgcolor=_SURFACE,
                    border_radius=10,
                    padding=_pad_all(12),
                )
            )

        if not combined:
            self._recent_list.controls.append(
                ft.Text(
                    "No cards yet. Add a sentence to get started!",
                    color=ft.Colors.WHITE54, size=13,
                )
            )

        self.page.update()

    def _status_color(self, status: str) -> str:
        return {
            "new": ft.Colors.BLUE_700,
            "learning": ft.Colors.ORANGE_700,
            "review": ft.Colors.INDIGO_600,
            "graduated": ft.Colors.GREEN_700,
        }.get(status, ft.Colors.GREY_700)

    def _go_to_review(self, _e=None) -> None:
        self.page.navigation_bar.selected_index = self.TAB_REVIEW
        self._tab = self.TAB_REVIEW
        self._load_next_card()
        self._body.content = self._review_panel
        self.page.update()

    # ══════════════════════════════════════════════════════════════════════════
    # REVIEW tab
    # ══════════════════════════════════════════════════════════════════════════

    def _build_review_panel(self) -> ft.Column:
        self._progress_bar = ft.ProgressBar(value=0, bgcolor=_SURFACE, color=_ACCENT)
        self._progress_label = ft.Text("0 reviewed this session", size=11, color=ft.Colors.WHITE54)
        self._session_count = 0

        # Card face — English sentence
        self._card_en = ft.Text(
            "Loading…",
            size=18,
            weight=ft.FontWeight.W_600,
            text_align=ft.TextAlign.CENTER,
            color=ft.Colors.WHITE,
        )
        self._card_context = ft.Text(
            "",
            size=11,
            color=ft.Colors.WHITE54,
            text_align=ft.TextAlign.CENTER,
            italic=True,
        )
        # Farsi translation (hidden until revealed)
        self._card_fa = ft.Text(
            "",
            size=15,
            color=ft.Colors.INDIGO_200,
            text_align=ft.TextAlign.CENTER,
            visible=False,
        )
        self._reveal_btn = ft.OutlinedButton(
            "Show Translation",
            icon=ft.Icons.VISIBILITY_OUTLINED,
            on_click=self._reveal_translation,
        )
        self._card_container = ft.Container(
            content=ft.Column(
                [
                    ft.Icon(ft.Icons.TRANSLATE, color=_ACCENT, size=32),
                    ft.Container(height=8),
                    self._card_en,
                    ft.Container(height=12),
                    self._card_fa,
                    ft.Divider(color=ft.Colors.WHITE12),
                    self._card_context,
                    ft.Container(height=8),
                    self._reveal_btn,
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=4,
            ),
            bgcolor=_SURFACE,
            border_radius=16,
            padding=_pad_all(24),
        )

        # Grade buttons (visible after translation is revealed)
        def _grade_btn(label, grade, color):
            return ft.Button(
                label,
                style=ft.ButtonStyle(
                    bgcolor=color,
                    color=ft.Colors.WHITE,
                    shape=ft.RoundedRectangleBorder(radius=10),
                    padding=_pad_sym(v=12),
                ),
                on_click=lambda _e, g=grade: self._submit_grade(g),
                expand=True,
            )

        self._grade_row = ft.Row(
            [
                _grade_btn("Again", GRADE_AGAIN, _AGAIN),
                _grade_btn("Hard", GRADE_HARD, _HARD),
                _grade_btn("Good", GRADE_GOOD, _GOOD),
                _grade_btn("Easy", GRADE_EASY, _EASY),
            ],
            spacing=8,
            visible=False,
        )

        # Result banner (shown momentarily after grading)
        self._result_text = ft.Text("", size=13, color=ft.Colors.WHITE)
        self._result_banner = ft.Container(
            content=ft.Row(
                [
                    ft.Icon(ft.Icons.CHECK_CIRCLE_OUTLINE, color=_EASY),
                    self._result_text,
                ],
                spacing=8,
            ),
            bgcolor=ft.Colors.with_opacity(0.15, ft.Colors.GREEN),
            border_radius=10,
            padding=_pad_sym(v=10, h=16),
            visible=False,
        )

        # AI Tutor — ask for a neuroscience-informed explanation of the sentence,
        # optionally grounded in the student's own attempt/paraphrase.
        self._tutor_input = ft.TextField(
            label="Your attempt (optional)",
            hint_text="Paraphrase the sentence or ask what's confusing...",
            multiline=True,
            min_lines=2,
            max_lines=3,
            border_radius=10,
            filled=True,
            bgcolor=_SURFACE,
            visible=False,
        )
        self._tutor_ask_btn = ft.OutlinedButton(
            "Ask AI Tutor",
            icon=ft.Icons.PSYCHOLOGY_ALT_OUTLINED,
            on_click=self._ask_tutor,
            visible=False,
        )
        self._tutor_loading = ft.ProgressRing(width=16, height=16, visible=False)

        # Conversation transcript — grows with each turn (initial answer +
        # any follow-ups), instead of a single static response block.
        self._tutor_transcript = ft.Column(spacing=8, visible=False)

        self._tutor_followup_input = ft.TextField(
            hint_text="Ask a follow-up (e.g. \"why?\")...",
            border_radius=10,
            filled=True,
            bgcolor=_SURFACE,
            expand=True,
            on_submit=self._ask_tutor_followup,
        )
        self._tutor_followup_btn = ft.IconButton(
            icon=ft.Icons.SEND_ROUNDED,
            icon_color=_ACCENT,
            on_click=self._ask_tutor_followup,
        )
        self._tutor_followup_row = ft.Row(
            [self._tutor_followup_input, self._tutor_followup_btn],
            visible=False,
        )

        self._tutor_section = ft.Column(
            [
                self._tutor_input,
                ft.Row([self._tutor_ask_btn, self._tutor_loading], spacing=12),
                self._tutor_transcript,
                self._tutor_followup_row,
            ],
            spacing=8,
        )

        # Empty state
        self._empty_state = ft.Container(
            content=ft.Column(
                [
                    ft.Icon(ft.Icons.CELEBRATION_OUTLINED, size=64, color=_EASY),
                    ft.Container(height=16),
                    ft.Text(
                        "All caught up!",
                        size=22, weight=ft.FontWeight.BOLD,
                        text_align=ft.TextAlign.CENTER,
                    ),
                    ft.Text(
                        "No cards due right now.\nAdd new sentences to keep learning.",
                        size=14, color=ft.Colors.WHITE54,
                        text_align=ft.TextAlign.CENTER,
                    ),
                    ft.Container(height=24),
                    ft.FilledButton("Add a Sentence", icon=ft.Icons.ADD, on_click=self._go_to_add),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=4,
            ),
            visible=False,
            padding=_pad_all(32),
        )

        return ft.Column(
            [
                ft.Row([self._progress_label], alignment=ft.MainAxisAlignment.CENTER),
                self._progress_bar,
                ft.Container(height=16),
                self._card_container,
                ft.Container(height=16),
                self._grade_row,
                self._result_banner,
                ft.Container(height=8),
                self._tutor_section,
                self._empty_state,
            ],
            spacing=8,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        )

    def _load_next_card(self) -> None:
        """Pull next due or new card from DB and display it."""
        self._show_translation = False
        self._result_banner.visible = False
        self._tutor_input.value = ""
        self._tutor_input.visible = False
        self._tutor_ask_btn.visible = False
        self._tutor_loading.visible = False
        self._tutor_transcript.controls.clear()
        self._tutor_transcript.visible = False
        self._tutor_followup_input.value = ""
        self._tutor_followup_row.visible = False
        self._tutor_conversation = None

        # Prefer overdue → then new
        due = self.repo.get_due_cards(limit=1)
        if due:
            card = due[0]
        else:
            new = self.repo.get_new_cards(limit=1)
            card = new[0] if new else None

        if card is None:
            self._current_card = None
            self._card_container.visible = False
            self._grade_row.visible = False
            self._empty_state.visible = True
            self.page.update()
            return

        self._current_card = card
        self._current_sentence = self.repo.get_sentence(card.sentence_id)
        self._card_container.visible = True
        self._empty_state.visible = False
        self._grade_row.visible = False

        if self._current_sentence:
            self._card_en.value = self._current_sentence.sentence_en
            self._card_fa.value = self._current_sentence.sentence_fa
            self._card_fa.visible = False
            self._card_context.value = self._current_sentence.context_notes
        self._reveal_btn.visible = True
        self.page.update()
        _log.info("Loaded card id=%s for review", card.id)

    def _reveal_translation(self, _e=None) -> None:
        self._card_fa.visible = True
        self._reveal_btn.visible = False
        self._grade_row.visible = True
        self._tutor_input.visible = True
        self._tutor_ask_btn.visible = True
        self.page.update()

    def _tutor_turn_widget(self, role: str, text: str) -> ft.Container:
        is_user = role == "user"
        return ft.Container(
            content=ft.Column(
                [
                    ft.Text(
                        "You" if is_user else "AI Tutor",
                        size=10, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE54,
                    ),
                    ft.Text(
                        text, size=13,
                        color=ft.Colors.WHITE70 if is_user else ft.Colors.INDIGO_200,
                    ),
                ],
                spacing=2,
            ),
            bgcolor=_SURFACE,
            border_radius=10,
            padding=_pad_all(10),
        )

    _TUTOR_UNAVAILABLE_MESSAGE = (
        "AI tutor isn't available right now — add an API key "
        "(ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY) or a "
        "local GGUF model (LOCAL_MODEL_PATH) to your .env file."
    )

    async def _ask_tutor(self, _e=None) -> None:
        """Start a follow-up-capable AI tutor conversation for the current
        sentence (+ the student's own attempt, if any) via
        RAGManager.start_conversation(), and show its first response."""
        if not self._current_sentence:
            return

        attempt = (self._tutor_input.value or "").strip()
        if not attempt:
            attempt = "(No attempt given — just explain this sentence.)"

        self._tutor_ask_btn.disabled = True
        self._tutor_loading.visible = True
        self.page.update()

        try:
            self._tutor_conversation = self.rag.start_conversation(
                target_sentence=self._current_sentence.sentence_en,
                user_input=attempt,
            )
            response = await self._tutor_conversation.send()
            self._tutor_transcript.controls.append(self._tutor_turn_widget("user", attempt))
            self._tutor_transcript.controls.append(self._tutor_turn_widget("assistant", response))
            self._tutor_input.visible = False
            self._tutor_ask_btn.visible = False
            self._tutor_followup_row.visible = True
        except ProviderError as exc:
            _log.warning("AI tutor unavailable: %s", exc)
            self._tutor_transcript.controls.append(
                self._tutor_turn_widget("assistant", self._TUTOR_UNAVAILABLE_MESSAGE)
            )
            self._tutor_conversation = None
        finally:
            self._tutor_ask_btn.disabled = False
            self._tutor_loading.visible = False
            self._tutor_transcript.visible = True
            self.page.update()

    async def _ask_tutor_followup(self, _e=None) -> None:
        """Continue the existing AI tutor conversation with one more
        student question, keeping the full prior history via
        TutorConversation.send()."""
        if not self._tutor_conversation:
            return

        question = (self._tutor_followup_input.value or "").strip()
        if not question:
            return

        self._tutor_followup_input.value = ""
        self._tutor_followup_btn.disabled = True
        self.page.update()

        try:
            response = await self._tutor_conversation.send(question)
            self._tutor_transcript.controls.append(self._tutor_turn_widget("user", question))
            self._tutor_transcript.controls.append(self._tutor_turn_widget("assistant", response))
        except ProviderError as exc:
            _log.warning("AI tutor follow-up unavailable: %s", exc)
            self._tutor_transcript.controls.append(
                self._tutor_turn_widget("assistant", self._TUTOR_UNAVAILABLE_MESSAGE)
            )
        finally:
            self._tutor_followup_btn.disabled = False
            self.page.update()

    def _submit_grade(self, grade: int) -> None:
        if not self._current_card:
            return

        card = self._current_card
        old_interval = card.interval
        result = apply_sm2(
            interval=card.interval,
            ease_factor=card.ease_factor,
            repetitions=card.repetitions,
            grade=grade,
        )

        self.repo.add_review_log(ReviewLog(
            card_id=card.id,
            grade=grade,
            old_interval=old_interval,
            new_interval=result.interval,
            old_ease_factor=card.ease_factor,
            new_ease_factor=result.ease_factor,
        ))

        card.interval = result.interval
        card.ease_factor = result.ease_factor
        card.repetitions = result.repetitions
        card.next_review_date = result.next_review_date
        card.status = result.status
        self.repo.update_card(card)

        grade_labels = {GRADE_AGAIN: "Again", GRADE_HARD: "Hard", GRADE_GOOD: "Good", GRADE_EASY: "Easy"}
        _log.debug(
            "Grade submitted | card=%d | grade=%s | %d→%d days | status=%s",
            card.id, grade_labels[grade], old_interval, result.interval, result.status,
        )

        grade_colors = {
            GRADE_AGAIN: ft.Colors.RED_700,
            GRADE_HARD: ft.Colors.ORANGE_700,
            GRADE_GOOD: ft.Colors.BLUE_700,
            GRADE_EASY: ft.Colors.GREEN_700,
        }
        self._result_banner.bgcolor = ft.Colors.with_opacity(0.18, grade_colors[grade])
        self._result_text.value = (
            f"{grade_labels[grade]} — next review in {result.interval} day(s)  ·  {result.status}"
        )
        self._result_banner.visible = True
        self._grade_row.visible = False
        self._session_count += 1
        self._progress_label.value = f"{self._session_count} reviewed this session"
        self.page.update()

        # Auto-advance after 1.2 s, scheduled on the page's own event loop so we
        # never touch Flet controls from a background thread (a raw
        # threading.Timer would). Cancel any still-pending advance first so
        # back-to-back grades can't race and load two cards on top of each other.
        if self._advance_task is not None:
            self._advance_task.cancel()
        self._advance_task = self.page.run_task(self._delayed_advance)

    async def _delayed_advance(self) -> None:
        await asyncio.sleep(1.2)
        self._load_next_card()

    def _go_to_add(self, _e=None) -> None:
        self.page.navigation_bar.selected_index = self.TAB_ADD
        self._tab = self.TAB_ADD
        self._body.content = self._add_panel
        self.page.update()

    # ══════════════════════════════════════════════════════════════════════════
    # ADD SENTENCE tab
    # ══════════════════════════════════════════════════════════════════════════

    def _build_add_panel(self) -> ft.Column:
        def _field(label: str, hint: str, multiline: bool = False) -> ft.TextField:
            return ft.TextField(
                label=label,
                hint_text=hint,
                multiline=multiline,
                min_lines=2 if multiline else 1,
                max_lines=4 if multiline else 1,
                border_radius=10,
                filled=True,
                bgcolor=_SURFACE,
                border_color=ft.Colors.INDIGO_300,
                focused_border_color=_ACCENT,
            )

        self._field_en = _field(
            "English Sentence",
            "e.g. She has been learning English for two years.",
        )
        self._field_fa = _field("Farsi Translation", "ترجمه فارسی...", multiline=True)
        self._field_notes = _field(
            "Context / Grammar Note (optional)",
            "e.g. Present perfect continuous",
            multiline=True,
        )

        self._save_status = ft.Text("", color=_EASY, size=13)

        save_btn = ft.FilledButton(
            "Save Sentence",
            icon=ft.Icons.SAVE_OUTLINED,
            style=ft.ButtonStyle(
                bgcolor=_ACCENT,
                color=ft.Colors.WHITE,
                padding=_pad_sym(v=14),
                shape=ft.RoundedRectangleBorder(radius=12),
            ),
            on_click=self._save_sentence,
            expand=True,
        )

        return ft.Column(
            [
                ft.Text("Add a New Sentence", size=20, weight=ft.FontWeight.W_600),
                ft.Text(
                    "Vocabulary is only learned in context — always provide the full sentence.",
                    size=12,
                    color=ft.Colors.WHITE54,
                ),
                ft.Divider(height=20, color=ft.Colors.WHITE12),
                ft.Text("Hebbian principle: no isolated words.", size=12, color=_ACCENT),
                ft.Container(height=8),
                self._field_en,
                self._field_fa,
                self._field_notes,
                ft.Container(height=8),
                ft.Row([save_btn]),
                self._save_status,
            ],
            spacing=12,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        )

    def _save_sentence(self, _e=None) -> None:
        en = (self._field_en.value or "").strip()
        fa = (self._field_fa.value or "").strip()
        notes = (self._field_notes.value or "").strip()

        if not en or not fa:
            self._save_status.value = "Both English and Farsi fields are required."
            self._save_status.color = _HARD
            self.page.update()
            return

        sentence = self.repo.add_sentence(Sentence(
            sentence_en=en,
            sentence_fa=fa,
            context_notes=notes,
        ))
        self.repo.add_card(Card(
            sentence_id=sentence.id,
            next_review_date=datetime.now(timezone.utc),
            status=CardStatus.NEW.value,
        ))

        _log.info("New sentence saved | id=%d | en=%.40s", sentence.id, en)

        self._field_en.value = ""
        self._field_fa.value = ""
        self._field_notes.value = ""
        self._save_status.value = f"Saved! Sentence #{sentence.id} added to your review deck."
        self._save_status.color = _EASY
        self.page.update()


# ── Entry point ───────────────────────────────────────────────────────────────

def main(page: ft.Page) -> None:
    try:
        repo = DatabaseRepository(DB_PATH)
        repo.create_schema()
        _seed(repo)

        router = LLMRouter(LLMConfig.from_env())
        store = NumpyVectorStore(persist_path=VECTOR_STORE_PATH)
        rag = RAGManager(store, HashingEmbeddingProvider(), router)
        if len(store) == 0:
            _seed_knowledge(rag)

        NeuroLingoApp(page, repo, rag)
        _log.info("NeuroLingo started successfully")
    except Exception:
        _log.exception("Fatal error during startup")
        page.add(ft.Text("Startup error — check logs/neurolingo.log", color=ft.Colors.RED_400))
        page.update()


if __name__ == "__main__":
    import os as _os
    try:
        # Use WEB_BROWSER view — no binary download required (desktop mode pulls
        # flet-macos.tar.gz from GitHub which fails if SSL certs aren't configured).
        # Port is read from the PORT env var so the preview harness can inject it.
        _port = int(_os.environ.get("PORT", "8550"))
        ft.app(
            target=main,
            view=ft.AppView.WEB_BROWSER,
            port=_port,
        )
    except Exception:
        log.exception("Fatal error in ft.app()")
        sys.exit(1)
