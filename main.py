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
import json
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import flet as ft
import flet_audio
import flet_audio_recorder as far

from logger_config import get_logger, setup_logging
from neurolingo.audio import tts as audio_tts
from neurolingo.audio.recorder import SAMPLE_RATE, pcm16_bytes_to_wav
from neurolingo.audio.similarity import score_shadowing
from neurolingo.core.llm.base import LLMConfig, ProviderError
from neurolingo.core.llm.providers import (
    AnthropicProvider,
    GeminiProvider,
    LocalLlamaProvider,
    OpenAIProvider,
)
from neurolingo.core.llm.router import LLMRouter
from neurolingo.core.rag.base import EmbeddingProvider
from neurolingo.core.rag.embeddings import build_embedding_provider
from neurolingo.core.rag.rag_manager import RAGManager, TutorConversation
from neurolingo.core.rag.vectorstore import NumpyVectorStore
from neurolingo.core.srs.algorithm import (
    GRADE_AGAIN,
    GRADE_EASY,
    GRADE_GOOD,
    GRADE_HARD,
    apply_sm2,
)
from neurolingo.db.backup import BackupFormatError, export_backup, import_backup
from neurolingo.db.models import Card, CardStatus, ReviewLog, Sentence
from neurolingo.db.repository import DatabaseRepository

log = setup_logging()
_log = get_logger(__name__)

DB_PATH = Path("data/neurolingo.db")
VECTOR_STORE_PATH = Path("data/knowledge")
SETTINGS_PATH = Path("data/settings.json")
TTS_CACHE_PATH = Path("data/tts_cache")
UPLOAD_DIR = Path("data/uploads")
SHADOWING_ATTEMPTS_PATH = Path("data/shadowing_attempts")

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


def _prepare_vector_store(path: Path, embedder: EmbeddingProvider) -> NumpyVectorStore:
    """
    Load the persisted knowledge base, discarding it if its vectors don't
    match the current embedder's dimension.

    This matters because which embedder we get is dynamic (build_embedding_
    provider() upgrades to cloud embeddings when a key is configured) — a
    store persisted under HashingEmbeddingProvider (e.g. 4096-dim) can't be
    compared against a query embedded with APIEmbeddingProvider (1536-dim),
    so switching providers must re-seed rather than crash on a shape
    mismatch the first time someone searches it.
    """
    store = NumpyVectorStore(persist_path=path)
    if store.vector_dim is not None and store.vector_dim != embedder.dim:
        _log.warning(
            "Knowledge base embedding dimension changed (%d -> %d) — "
            "clearing and re-seeding",
            store.vector_dim, embedder.dim,
        )
        store.clear()
    return store


# ── Colour palette — "Persian Modern" ─────────────────────────────────────────
# Grading colours (Again/Hard/Good/Easy) are semantic state, not brand accent —
# they stay Material red/orange/blue/green regardless of theme.
_AGAIN = ft.Colors.RED_400
_HARD = ft.Colors.ORANGE_400
_GOOD = ft.Colors.BLUE_400
_EASY = ft.Colors.GREEN_400

_BG = "#0F2C31"            # glazed-tile teal ground
_SURFACE = "#163A40"       # card/tile surface
_SURFACE_RAISED = "#1B454C"
_INK = "#F6F1E4"           # warm ivory text
_INK_SOFT = ft.Colors.with_opacity(0.66, "#F6F1E4")
_INK_FAINT = ft.Colors.with_opacity(0.38, "#F6F1E4")
_DIVIDER = ft.Colors.with_opacity(0.16, "#E4A93A")
_ACCENT = "#E4A93A"        # gold — brand accent, CTAs, headline colour
_ACCENT_DEEP = "#C98A26"   # deeper gold, gradient partner
_ACCENT_ON = "#17282A"     # dark ink used as text ON the gold accent
_TURQUOISE = "#37C6BE"     # secondary accent — section titles, sentence rules

_FONT_DISPLAY = "Reem Kufi"
_FONT_BODY = "Vazirmatn"
_FONT_BODY_BOLD = "Vazirmatn Bold"

_CARD_SHADOW = ft.BoxShadow(
    spread_radius=0,
    blur_radius=16,
    color=ft.Colors.with_opacity(0.35, ft.Colors.BLACK),
    offset=ft.Offset(0, 4),
)


# ── Helper: Padding shorthands ────────────────────────────────────────────────

def _pad_all(n: int) -> ft.Padding:
    return ft.Padding(top=n, bottom=n, left=n, right=n)


def _pad_sym(*, v: int = 0, h: int = 0) -> ft.Padding:
    return ft.Padding(top=v, bottom=v, left=h, right=h)


def _border_all(width: float, color: str) -> ft.Border:
    """This installed Flet version has no ft.border.all() convenience
    function — build a uniform Border from four identical BorderSides."""
    side = ft.BorderSide(width=width, color=color)
    return ft.Border(top=side, right=side, bottom=side, left=side)


# ── App ───────────────────────────────────────────────────────────────────────

class NeuroLingoApp:
    """Stateful single-page application controller."""

    TAB_TODAY = 0
    TAB_LIBRARY = 1
    TAB_ADD = 2

    def __init__(
        self, page: ft.Page, repo: DatabaseRepository, rag: RAGManager, settings_path: Path,
    ) -> None:
        self.page = page
        self.repo = repo
        self.rag = rag
        self.settings_path = settings_path
        self._tab = self.TAB_TODAY
        self._current_card: Card | None = None
        self._current_sentence: Sentence | None = None
        self._show_translation = False
        self._advance_task: asyncio.Task | None = None
        self._tutor_conversation: TutorConversation | None = None
        self._shadowing_recording = False
        self._shadowing_upload_done: asyncio.Event | None = None
        self._shadowing_upload_error: str | None = None
        self._shadowing_upload_filename: str | None = None
        self._setup_page()
        self._build_ui()
        self._refresh_today()

    # ── Page configuration ────────────────────────────────────────────────────

    def _setup_page(self) -> None:
        p = self.page
        p.title = "NeuroLingo"
        p.fonts = {
            _FONT_BODY: "fonts/Vazirmatn-Regular.ttf",
            _FONT_BODY_BOLD: "fonts/Vazirmatn-Bold.ttf",
            _FONT_DISPLAY: "fonts/ReemKufi.ttf",
        }
        p.theme_mode = ft.ThemeMode.DARK
        p.theme = ft.Theme(color_scheme_seed=ft.Colors.TEAL, font_family=_FONT_BODY)
        p.window.width = 480
        p.window.height = 860
        p.window.min_width = 380
        p.padding = 0
        p.bgcolor = _BG

    # ── Icons ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _icon_image(svg_name: str, *, color: str, size: int = 22) -> ft.Image:
        """Custom geometric icon (8-point star, leaning books, diamond+plus,
        sliders) tinted via SRC_IN blend — replaces Material glyphs so the
        nav bar reads as part of the same tile-work visual language."""
        return ft.Image(
            src=f"icons/{svg_name}.svg",
            width=size, height=size,
            color=color,
            color_blend_mode=ft.BlendMode.SRC_IN,
        )

    @staticmethod
    def _sunburst_gradient() -> ft.SweepGradient:
        """Repeating conic 'sunburst' rays approximating the mockup's CSS
        repeating-conic-gradient (no direct Flet equivalent) via explicit
        alternating SweepGradient stops: each of the 15 segments is ~80%
        transparent then a hard-edged sliver of faint gold, matching the
        mockup's 20deg-transparent/5deg-gold 25deg period."""
        segments = 15
        transparent = ft.Colors.with_opacity(0.0, _ACCENT)
        faint_gold = ft.Colors.with_opacity(0.05, _ACCENT)
        colors: list[str] = []
        stops: list[float] = []
        for i in range(segments):
            start = i / segments
            edge = start + 0.8 / segments
            end = (i + 1) / segments
            stops.extend([start, edge, edge, end])
            colors.extend([transparent, transparent, faint_gold, faint_gold])
        stops[0] = 0.0
        stops[-1] = 1.0
        return ft.SweepGradient(colors=colors, stops=stops)

    # ── UI skeleton ───────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # ── AppBar ──
        self._appbar = ft.AppBar(
            leading=ft.Icon(ft.Icons.PSYCHOLOGY, color=_ACCENT, size=28),
            leading_width=48,
            title=ft.Text("NeuroLingo", size=20, color=_INK, font_family=_FONT_DISPLAY),
            center_title=False,
            bgcolor=_BG,
            actions=[
                ft.IconButton(
                    icon=self._icon_image("settings_sliders", color=_ACCENT, size=20),
                    tooltip="Settings — AI provider API keys",
                    on_click=self._go_to_settings,
                )
            ],
        )

        # ── Content panels (one per tab, plus Review — reached from Today's
        # "Continue Learning" button, not a tab itself — and Settings, reached
        # via the AppBar gear icon; both return to whichever tab was active) ──
        self._today_panel = self._build_today_panel()
        self._library_panel = self._build_library_panel()
        self._review_panel = self._build_review_panel()
        self._add_panel = self._build_add_panel()
        self._settings_panel = self._build_settings_panel()

        self._content_switcher = ft.AnimatedSwitcher(
            content=self._today_panel,
            transition=ft.AnimatedSwitcherTransition.FADE,
            duration=200,
            switch_in_curve=ft.AnimationCurve.EASE_OUT,
            switch_out_curve=ft.AnimationCurve.EASE_IN,
        )
        # Tile-work background: a repeating conic "sunburst" (no direct CSS
        # equivalent exists in Flet, so it's built from explicit alternating
        # SweepGradient stops) plus a small tiled diagonal-crosshatch SVG,
        # both behind the actual content — echoes the girih motif without
        # touching text contrast.
        textured_body = ft.Stack(
            [
                ft.Container(gradient=self._sunburst_gradient(), expand=True),
                ft.Image(
                    src="images/bg_pattern.svg", width=40, height=40,
                    repeat=ft.ImageRepeat.REPEAT, fit=ft.BoxFit.NONE, expand=True,
                ),
                self._content_switcher,
            ],
            expand=True,
        )
        self._body = ft.Container(
            content=textured_body,
            expand=True,
            padding=_pad_all(16),
        )

        # ── Bottom NavigationBar ──
        def _nav_destination(svg_name: str, label: str) -> ft.NavigationBarDestination:
            return ft.NavigationBarDestination(
                icon=self._icon_image(svg_name, color=_INK_SOFT),
                selected_icon=self._icon_image(svg_name, color=_ACCENT),
                label=label,
            )

        self.page.navigation_bar = ft.NavigationBar(
            selected_index=self.TAB_TODAY,
            bgcolor=_SURFACE,
            indicator_color=ft.Colors.with_opacity(0.25, _ACCENT),
            shadow_color=ft.Colors.BLACK,
            elevation=8,
            destinations=[
                _nav_destination("today", "Today"),
                _nav_destination("library", "Library"),
                _nav_destination("add_diamond", "Add"),
            ],
            on_change=self._on_tab_change,
        )
        self.page.appbar = self._appbar

        self._file_picker = ft.FilePicker()
        self.page.services.append(self._file_picker)

        self._audio_player = flet_audio.Audio(src="")
        self.page.services.append(self._audio_player)

        self._audio_recorder = far.AudioRecorder(
            configuration=far.AudioRecorderConfiguration(
                encoder=far.AudioEncoder.PCM16BITS,
                sample_rate=SAMPLE_RATE,
                channels=1,
            ),
            on_upload=self._on_shadowing_upload,
        )
        self.page.services.append(self._audio_recorder)

        self.page.add(self._body)

    # ── Tab switching ─────────────────────────────────────────────────────────

    def _on_tab_change(self, e: ft.ControlEvent) -> None:
        self._tab = int(e.data)
        panels = {
            self.TAB_TODAY: self._today_panel,
            self.TAB_LIBRARY: self._library_panel,
            self.TAB_ADD: self._add_panel,
        }
        if self._tab == self.TAB_TODAY:
            self._refresh_today()
        elif self._tab == self.TAB_LIBRARY:
            self._refresh_library()
        self._content_switcher.content = panels[self._tab]
        self.page.update()

    # ══════════════════════════════════════════════════════════════════════════
    # TODAY tab — action-first: what's next, not a stats dashboard
    # ══════════════════════════════════════════════════════════════════════════

    def _build_today_panel(self) -> ft.Column:
        # Next-up preview — the actual next card, replacing a decorative
        # illustration with the one thing this screen exists to tell you.
        self._next_eyebrow = ft.Text("", size=10, color=_TURQUOISE, weight=ft.FontWeight.BOLD)
        self._next_en = ft.Text(
            "", size=14, weight=ft.FontWeight.W_600, color=_INK, font_family=_FONT_BODY_BOLD,
        )
        self._next_fa = ft.Text("", size=12, color=_ACCENT, rtl=True)
        self._next_card_preview = ft.Container(
            content=ft.Column(
                [
                    self._next_eyebrow,
                    ft.Container(height=6),
                    self._next_en,
                    ft.Container(height=4),
                    self._next_fa,
                ],
                spacing=0,
            ),
            bgcolor=_SURFACE_RAISED,
            border=_border_all(1, ft.Colors.with_opacity(0.3, _ACCENT)),
            border_radius=16,
            padding=_pad_all(16),
        )

        # Compact stats ribbon — context, not the headline anymore.
        self._today_stat_due = ft.Text(
            "—", size=17, weight=ft.FontWeight.BOLD, color=_ACCENT, font_family=_FONT_BODY_BOLD,
        )
        self._today_stat_new = ft.Text(
            "—", size=17, weight=ft.FontWeight.BOLD, color=_TURQUOISE, font_family=_FONT_BODY_BOLD,
        )
        self._today_stat_total = ft.Text(
            "—", size=17, weight=ft.FontWeight.BOLD, color=_INK_SOFT, font_family=_FONT_BODY_BOLD,
        )
        ribbon = ft.Container(
            content=ft.Row(
                [
                    self._ribbon_cell("Due", self._today_stat_due),
                    ft.VerticalDivider(width=1, color=_DIVIDER),
                    self._ribbon_cell("New", self._today_stat_new),
                    ft.VerticalDivider(width=1, color=_DIVIDER),
                    self._ribbon_cell("Total", self._today_stat_total),
                ],
                alignment=ft.MainAxisAlignment.SPACE_EVENLY,
            ),
            bgcolor=_SURFACE,
            border=_border_all(1, ft.Colors.with_opacity(0.18, _TURQUOISE)),
            border_radius=12,
            padding=_pad_sym(v=10),
        )

        self._continue_btn = self._gradient_button(
            "Continue Learning", ft.Icons.PLAY_ARROW_ROUNDED, self._go_to_review,
        )

        self._up_next_list = ft.Column(spacing=6)

        self._today_content = ft.Column(
            [
                self._next_card_preview,
                ft.Container(height=14),
                ribbon,
                ft.Container(height=16),
                ft.Row([self._continue_btn]),
                ft.Container(height=20),
                ft.Text("Up Next", size=13, color=_TURQUOISE, font_family=_FONT_DISPLAY),
                ft.Divider(height=8, color=_DIVIDER),
                self._up_next_list,
            ],
            spacing=4,
        )

        # Empty state — nothing at all to review yet (fresh install or every
        # sentence deleted); this is where the hero illustration lives now.
        self._today_empty_state = ft.Container(
            content=ft.Column(
                [
                    ft.Image(src="images/hero_learning.svg", fit=ft.BoxFit.CONTAIN, height=120),
                    ft.Container(height=12),
                    ft.Text(
                        "Nothing to review yet", size=18, weight=ft.FontWeight.BOLD,
                        color=_INK, text_align=ft.TextAlign.CENTER,
                    ),
                    ft.Text(
                        "Add your first sentence to start learning.",
                        size=13, color=_INK_SOFT, text_align=ft.TextAlign.CENTER,
                    ),
                    ft.Container(height=16),
                    ft.FilledButton("Add a Sentence", icon=ft.Icons.ADD, on_click=self._go_to_add),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            visible=False,
            padding=_pad_all(24),
        )

        return ft.Column(
            [
                ft.Text("Welcome back!", size=22, color=_INK, font_family=_FONT_DISPLAY),
                ft.Text(
                    "Sentences reviewed in context — no isolated flashcards.",
                    size=12, color=_INK_SOFT,
                ),
                ft.Divider(height=20, color=_DIVIDER),
                self._today_content,
                self._today_empty_state,
            ],
            spacing=8,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        )

    def _ribbon_cell(self, label: str, value_widget: ft.Text) -> ft.Column:
        return ft.Column(
            [value_widget, ft.Text(label, size=9, color=_INK_SOFT)],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=0,
        )

    def _gradient_button(self, label: str, icon, on_click) -> ft.Container:
        """A pill-shaped, gradient-filled primary action button — used for
        the app's two main calls to action (start a review, save a sentence)
        so they read as the same visual language."""
        return ft.Container(
            content=ft.Row(
                [
                    ft.Icon(icon, color=_ACCENT_ON),
                    ft.Text(
                        label, color=_ACCENT_ON, weight=ft.FontWeight.W_600, size=15,
                        font_family=_FONT_BODY_BOLD,
                    ),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=8,
            ),
            gradient=ft.LinearGradient(
                begin=ft.alignment.Alignment(-1, 0),
                end=ft.alignment.Alignment(1, 0),
                colors=[_ACCENT_DEEP, _ACCENT],
            ),
            border_radius=14,
            padding=_pad_sym(v=14, h=24),
            shadow=_CARD_SHADOW,
            ink=True,
            on_click=on_click,
            expand=True,
        )

    def _stat_card(self, label: str, value_widget: ft.Text, icon, color) -> ft.Container:
        icon_badge = ft.Container(
            content=ft.Icon(icon, color=color, size=18),
            bgcolor=ft.Colors.with_opacity(0.16, color),
            border_radius=999,
            padding=8,
        )
        return ft.Container(
            content=ft.Column(
                [
                    icon_badge,
                    ft.Container(height=6),
                    value_widget,
                    ft.Text(label, size=11, color=_INK_SOFT),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=2,
            ),
            bgcolor=_SURFACE,
            border_radius=16,
            border=_border_all(1, ft.Colors.with_opacity(0.18, _TURQUOISE)),
            padding=_pad_sym(v=14, h=8),
            expand=True,
        )

    def _refresh_today(self) -> None:
        due = self.repo.get_due_cards(limit=100)
        new = self.repo.get_new_cards(limit=100)
        queue = due + new

        self._today_stat_due.value = str(len(due))
        self._today_stat_new.value = str(len(new))
        self._today_stat_total.value = str(self.repo.count_sentences())

        if not queue:
            self._today_content.visible = False
            self._today_empty_state.visible = True
            self.page.update()
            return

        self._today_content.visible = True
        self._today_empty_state.visible = False

        next_card = queue[0]
        next_sentence = self.repo.get_sentence(next_card.sentence_id)
        if next_sentence:
            if next_card.status == CardStatus.NEW.value:
                self._next_eyebrow.value = "Next up · New"
            else:
                due_days = (datetime.now(timezone.utc) - next_card.next_review_date).days
                self._next_eyebrow.value = (
                    f"Next up · Due {due_days}d ago" if due_days > 0 else "Next up · Due now"
                )
            self._next_en.value = next_sentence.sentence_en
            self._next_fa.value = next_sentence.sentence_fa

        self._up_next_list.controls.clear()
        for card in queue[1:3]:
            sentence = self.repo.get_sentence(card.sentence_id)
            if not sentence:
                continue
            is_new = card.status == CardStatus.NEW.value
            self._up_next_list.controls.append(
                ft.Container(
                    content=ft.Row(
                        [
                            ft.Text(
                                sentence.sentence_en, size=11, color=_INK, expand=True,
                                max_lines=1, overflow=ft.TextOverflow.ELLIPSIS,
                            ),
                            ft.Container(
                                content=ft.Text(
                                    "NEW" if is_new else "DUE", size=8,
                                    weight=ft.FontWeight.BOLD,
                                    color=_ACCENT_ON if is_new else _INK,
                                ),
                                bgcolor=_TURQUOISE if is_new else _ACCENT,
                                border_radius=999,
                                padding=_pad_sym(v=2, h=7),
                            ),
                        ],
                        spacing=8,
                    ),
                    bgcolor=_SURFACE,
                    border_radius=8,
                    padding=_pad_sym(v=8, h=10),
                )
            )
        if not self._up_next_list.controls:
            self._up_next_list.controls.append(
                ft.Text("That's everything for now.", size=11, color=_INK_SOFT)
            )

        self.page.update()

    def _status_color(self, status: str) -> str:
        return {
            "new": ft.Colors.BLUE_700,
            "learning": ft.Colors.ORANGE_700,
            "review": _TURQUOISE,
            "graduated": ft.Colors.GREEN_700,
        }.get(status, ft.Colors.GREY_700)

    def _go_to_review(self, _e=None) -> None:
        self._load_next_card()
        self._content_switcher.content = self._review_panel
        self.page.update()

    def _back_from_review(self, _e=None) -> None:
        self._content_switcher.content = self._today_panel
        self._refresh_today()
        self.page.update()

    # ══════════════════════════════════════════════════════════════════════════
    # REVIEW tab
    # ══════════════════════════════════════════════════════════════════════════

    def _build_review_panel(self) -> ft.Column:
        self._progress_bar = ft.ProgressBar(value=0, bgcolor=_SURFACE, color=_ACCENT)
        self._progress_label = ft.Text("0 reviewed this session", size=11, color=_INK_SOFT)
        self._session_count = 0

        # Card face — English sentence
        self._card_en = ft.Text(
            "Loading…",
            size=18,
            weight=ft.FontWeight.W_600,
            text_align=ft.TextAlign.CENTER,
            color=_INK,
            font_family=_FONT_BODY_BOLD,
        )
        self._card_context = ft.Text(
            "",
            size=11,
            color=_INK_SOFT,
            text_align=ft.TextAlign.CENTER,
            italic=True,
        )
        # Farsi translation (hidden until revealed)
        self._card_fa = ft.Text(
            "",
            size=15,
            color=_ACCENT,
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
                    ft.Container(
                        content=ft.Icon(ft.Icons.TRANSLATE, color=_ACCENT, size=26),
                        bgcolor=ft.Colors.with_opacity(0.16, _ACCENT),
                        border_radius=999,
                        padding=12,
                    ),
                    ft.Container(height=8),
                    self._card_en,
                    ft.Container(height=12),
                    self._card_fa,
                    ft.Divider(color=_DIVIDER),
                    self._card_context,
                    ft.Container(height=8),
                    self._reveal_btn,
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=4,
            ),
            bgcolor=_SURFACE_RAISED,
            border_radius=18,
            border=_border_all(1, ft.Colors.with_opacity(0.2, _TURQUOISE)),
            shadow=_CARD_SHADOW,
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
        self._result_text = ft.Text("", size=13, color=_INK)
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

        # ── Shadowing exercise: play a TTS reference, record your attempt
        # via the browser's own microphone, get a rhythm/timing score. ──
        self._shadowing_status = ft.Text("", size=12, color=_INK_SOFT)
        self._shadowing_play_btn = ft.OutlinedButton(
            "Play Reference", icon=ft.Icons.VOLUME_UP_OUTLINED,
            on_click=self._play_reference_audio,
        )
        self._shadowing_record_btn = ft.OutlinedButton(
            "Record My Attempt", icon=ft.Icons.MIC_OUTLINED,
            on_click=self._toggle_shadowing_recording,
        )
        self._shadowing_section = ft.Column(
            [
                ft.Text(
                    "Shadow This Sentence", size=13, color=_TURQUOISE,
                    font_family=_FONT_DISPLAY,
                ),
                ft.Row([self._shadowing_play_btn, self._shadowing_record_btn], spacing=8),
                self._shadowing_status,
            ],
            spacing=8,
        )

        # Empty state
        self._empty_state = ft.Container(
            content=ft.Column(
                [
                    ft.Container(
                        content=ft.Icon(ft.Icons.CELEBRATION_OUTLINED, size=48, color=_EASY),
                        bgcolor=ft.Colors.with_opacity(0.14, _EASY),
                        border_radius=999,
                        padding=20,
                    ),
                    ft.Container(height=16),
                    ft.Text(
                        "All caught up!",
                        size=22, weight=ft.FontWeight.BOLD,
                        text_align=ft.TextAlign.CENTER,
                    ),
                    ft.Text(
                        "No cards due right now.\nAdd new sentences to keep learning.",
                        size=14, color=_INK_SOFT,
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
                ft.Row(
                    [
                        ft.IconButton(
                            icon=ft.Icons.ARROW_BACK, icon_color=_INK,
                            on_click=self._back_from_review,
                        ),
                        ft.Text("Review Session", size=16, color=_INK, font_family=_FONT_DISPLAY),
                    ],
                    spacing=4,
                ),
                ft.Row([self._progress_label], alignment=ft.MainAxisAlignment.CENTER),
                self._progress_bar,
                ft.Container(height=16),
                self._card_container,
                ft.Container(height=16),
                self._grade_row,
                self._result_banner,
                ft.Container(height=8),
                self._tutor_section,
                ft.Container(height=8),
                self._shadowing_section,
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
        self._shadowing_status.value = ""
        self._shadowing_record_btn.text = "Record My Attempt"
        self._shadowing_record_btn.icon = ft.Icons.MIC_OUTLINED
        self._shadowing_recording = False

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
                        size=10, weight=ft.FontWeight.BOLD, color=_INK_SOFT,
                    ),
                    ft.Text(
                        text, size=13,
                        color=_INK_SOFT if is_user else _ACCENT,
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

    # ── Shadowing exercise ───────────────────────────────────────────────────

    async def _play_reference_audio(self, _e=None) -> None:
        """Synthesize (or reuse the cached synthesis of) the current
        sentence and play it back, so the student hears a model
        pronunciation before recording their own attempt."""
        if not self._current_sentence:
            return
        if not audio_tts.is_available():
            self._shadowing_status.value = (
                "Text-to-speech isn't available on this device — "
                "no OS speech engine was found."
            )
            self._shadowing_status.color = _AGAIN
            self.page.update()
            return

        try:
            ref_path = audio_tts.get_reference_audio(
                self._current_sentence.sentence_en, TTS_CACHE_PATH,
            )
            self._audio_player.src = ref_path.read_bytes()
            self._audio_player.update()
            await self._audio_player.play()
        except Exception:
            _log.exception("Failed to play shadowing reference audio")
            self._shadowing_status.value = "Couldn't generate reference audio — check logs."
            self._shadowing_status.color = _AGAIN
            self.page.update()

    def _on_shadowing_upload(self, e) -> None:
        """Fires as the browser streams the recorded PCM bytes to our
        upload endpoint; signals _stop_shadowing_recording() once the
        upload is complete (progress==1.0) or has failed."""
        if e.error:
            _log.warning("Shadowing recording upload failed: %s", e.error)
            self._shadowing_upload_error = e.error
            if self._shadowing_upload_done is not None:
                self._shadowing_upload_done.set()
        elif e.progress is not None and e.progress >= 1.0:
            if self._shadowing_upload_done is not None:
                self._shadowing_upload_done.set()

    async def _toggle_shadowing_recording(self, _e=None) -> None:
        if self._shadowing_recording:
            await self._stop_shadowing_recording()
        else:
            await self._start_shadowing_recording()

    async def _start_shadowing_recording(self) -> None:
        if not self._current_sentence:
            return

        try:
            has_permission = await self._audio_recorder.has_permission()
        except Exception:
            _log.exception("Microphone permission check failed")
            has_permission = False
        if not has_permission:
            self._shadowing_status.value = (
                "Microphone permission was denied — check your browser's site "
                "settings and try again."
            )
            self._shadowing_status.color = _AGAIN
            self.page.update()
            return

        self._shadowing_upload_done = asyncio.Event()
        self._shadowing_upload_error = None
        file_name = f"shadow_{uuid.uuid4().hex}.pcm"
        self._shadowing_upload_filename = file_name
        upload_url = self.page.get_upload_url(file_name, expires=120)

        try:
            started = await self._audio_recorder.start_recording(
                upload=far.AudioRecorderUploadSettings(
                    upload_url=upload_url, file_name=file_name,
                ),
            )
        except Exception:
            _log.exception("Failed to start shadowing recording")
            started = False

        if not started:
            self._shadowing_status.value = "Couldn't start recording — check mic permissions."
            self._shadowing_status.color = _AGAIN
            self.page.update()
            return

        self._shadowing_recording = True
        self._shadowing_record_btn.text = "Stop Recording"
        self._shadowing_record_btn.icon = ft.Icons.STOP_CIRCLE_OUTLINED
        self._shadowing_status.value = "Recording — speak the sentence now."
        self._shadowing_status.color = _TURQUOISE
        self.page.update()

    async def _stop_shadowing_recording(self) -> None:
        self._shadowing_recording = False
        self._shadowing_record_btn.text = "Record My Attempt"
        self._shadowing_record_btn.icon = ft.Icons.MIC_OUTLINED

        try:
            await self._audio_recorder.stop_recording()
        except Exception:
            _log.exception("Failed to stop shadowing recording")

        self._shadowing_status.value = "Uploading your recording…"
        self._shadowing_status.color = _TURQUOISE
        self.page.update()

        assert self._shadowing_upload_done is not None
        try:
            await asyncio.wait_for(self._shadowing_upload_done.wait(), timeout=15)
        except asyncio.TimeoutError:
            self._shadowing_status.value = "Upload timed out — try recording again."
            self._shadowing_status.color = _AGAIN
            self.page.update()
            return

        if self._shadowing_upload_error:
            self._shadowing_status.value = f"Upload failed: {self._shadowing_upload_error}"
            self._shadowing_status.color = _AGAIN
            self.page.update()
            return

        raw_upload_path = UPLOAD_DIR / (self._shadowing_upload_filename or "")
        if not self._shadowing_upload_filename or not raw_upload_path.exists():
            self._shadowing_status.value = "Recording didn't arrive — try again."
            self._shadowing_status.color = _AGAIN
            self.page.update()
            return

        try:
            raw_bytes = raw_upload_path.read_bytes()
            wav_path = SHADOWING_ATTEMPTS_PATH / f"{raw_upload_path.stem}.wav"
            pcm16_bytes_to_wav(raw_bytes, wav_path)

            ref_path = audio_tts.get_reference_audio(
                self._current_sentence.sentence_en, TTS_CACHE_PATH,
            )
            score = score_shadowing(ref_path, wav_path)

            self._shadowing_status.value = f"Shadowing score: {score:.0f}/100"
            self._shadowing_status.color = _EASY if score >= 70 else (
                _HARD if score >= 40 else _AGAIN
            )
        except Exception:
            _log.exception("Shadowing scoring failed")
            self._shadowing_status.value = "Couldn't score your attempt — check logs."
            self._shadowing_status.color = _AGAIN
        finally:
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
        self._content_switcher.content = self._add_panel
        self.page.update()

    # ══════════════════════════════════════════════════════════════════════════
    # LIBRARY tab — browse everything (search, full list, real stats)
    # ══════════════════════════════════════════════════════════════════════════

    def _build_library_panel(self) -> ft.Column:
        self._lib_subtitle = ft.Text("", size=12, color=_INK_SOFT)

        self._lib_stat_due = ft.Text("—", size=24, weight=ft.FontWeight.BOLD, color=_ACCENT)
        self._lib_stat_new = ft.Text("—", size=24, weight=ft.FontWeight.BOLD, color=_TURQUOISE)
        self._lib_stat_graduated = ft.Text("—", size=24, weight=ft.FontWeight.BOLD, color=_EASY)
        self._lib_stat_total = ft.Text("—", size=24, weight=ft.FontWeight.BOLD, color=_INK_SOFT)

        stats_row = ft.Row(
            [
                self._stat_card("Due", self._lib_stat_due, ft.Icons.TIMER_OUTLINED, _ACCENT),
                self._stat_card("New", self._lib_stat_new, ft.Icons.FIBER_NEW_OUTLINED, _TURQUOISE),
                self._stat_card(
                    "Grad.", self._lib_stat_graduated, ft.Icons.WORKSPACE_PREMIUM_OUTLINED, _EASY,
                ),
                self._stat_card("Total", self._lib_stat_total, ft.Icons.LIBRARY_BOOKS_OUTLINED, _INK_SOFT),
            ],
            spacing=6,
        )

        self._lib_search = ft.TextField(
            hint_text="Search your sentences…",
            prefix_icon=ft.Icons.SEARCH,
            border_radius=10,
            filled=True,
            bgcolor=_SURFACE,
            border_color=ft.Colors.with_opacity(0.25, _TURQUOISE),
            focused_border_color=_ACCENT,
            on_change=self._on_library_search,
        )

        self._lib_list = ft.Column(spacing=8)

        return ft.Column(
            [
                ft.Text("Your Library", size=22, color=_INK, font_family=_FONT_DISPLAY),
                self._lib_subtitle,
                ft.Divider(height=20, color=_DIVIDER),
                stats_row,
                ft.Container(height=16),
                self._lib_search,
                ft.Container(height=8),
                self._lib_list,
            ],
            spacing=8,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        )

    def _on_library_search(self, _e=None) -> None:
        self._refresh_library(self._lib_search.value or "")

    def _refresh_library(self, query: str = "") -> None:
        due_count = len(self.repo.get_due_cards(limit=1000))
        new_count = len(self.repo.get_new_cards(limit=1000))
        graduated_count = self.repo.count_cards_by_status(CardStatus.GRADUATED.value)
        total = self.repo.count_sentences()

        self._lib_stat_due.value = str(due_count)
        self._lib_stat_new.value = str(new_count)
        self._lib_stat_graduated.value = str(graduated_count)
        self._lib_stat_total.value = str(total)
        self._lib_subtitle.value = f"{total} sentence(s), reviewed in context"

        query = query.strip()
        sentences = self.repo.search_sentences(query) if query else self.repo.get_all_sentences()

        self._lib_list.controls.clear()
        for sentence in sentences:
            card = self.repo.get_card_by_sentence_id(sentence.id)
            status = card.status if card else CardStatus.NEW.value
            interval_text = f"{card.interval}d" if card else "—"
            self._lib_list.controls.append(
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Text(
                                sentence.sentence_en, size=13,
                                weight=ft.FontWeight.W_500, color=_INK,
                            ),
                            ft.Row(
                                [
                                    ft.Container(
                                        content=ft.Text(
                                            status.upper(), size=10,
                                            weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE,
                                        ),
                                        bgcolor=self._status_color(status),
                                        border_radius=999,
                                        padding=_pad_sym(v=3, h=8),
                                    ),
                                    ft.Text(f"Interval: {interval_text}", size=11, color=_INK_SOFT),
                                ],
                                spacing=8,
                            ),
                        ],
                        spacing=4,
                    ),
                    bgcolor=_SURFACE,
                    border_radius=12,
                    border=_border_all(1, ft.Colors.with_opacity(0.16, _TURQUOISE)),
                    padding=_pad_all(12),
                )
            )

        if not sentences:
            message = (
                "No sentences match your search." if query
                else "No sentences yet. Add one to get started!"
            )
            self._lib_list.controls.append(ft.Text(message, color=_INK_SOFT, size=13))

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
                border_color=_TURQUOISE,
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

        save_btn = self._gradient_button("Save Sentence", ft.Icons.SAVE_OUTLINED, self._save_sentence)

        return ft.Column(
            [
                ft.Text("Add a New Sentence", size=20, color=_INK, font_family=_FONT_DISPLAY),
                ft.Text(
                    "Vocabulary is only learned in context — always provide the full sentence.",
                    size=12,
                    color=_INK_SOFT,
                ),
                ft.Divider(height=20, color=_DIVIDER),
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

    # ══════════════════════════════════════════════════════════════════════════
    # SETTINGS (reached via the AppBar gear icon, not a bottom-nav tab)
    # ══════════════════════════════════════════════════════════════════════════

    def _build_settings_panel(self) -> ft.Column:
        def _field(label: str, *, password: bool = False, hint: str = "") -> ft.TextField:
            return ft.TextField(
                label=label,
                hint_text=hint,
                password=password,
                can_reveal_password=password,
                border_radius=10,
                filled=True,
                bgcolor=_SURFACE,
            )

        self._settings_preferred = ft.Dropdown(
            label="Preferred provider",
            options=[
                ft.dropdown.Option("anthropic", "Anthropic"),
                ft.dropdown.Option("openai", "OpenAI"),
                ft.dropdown.Option("gemini", "Gemini"),
                ft.dropdown.Option("local", "Local (llama.cpp)"),
            ],
            border_radius=10,
            filled=True,
            bgcolor=_SURFACE,
        )
        self._settings_anthropic_key = _field("Anthropic API key", password=True)
        self._settings_openai_key = _field("OpenAI API key", password=True)
        self._settings_gemini_key = _field("Gemini API key", password=True)
        self._settings_local_model_path = _field(
            "Local GGUF model path", hint="/path/to/model.gguf",
        )

        self._settings_save_status = ft.Text("", size=12)
        self._settings_status_column = ft.Column(spacing=6)

        save_btn = self._gradient_button(
            "Save Settings", ft.Icons.SAVE_OUTLINED, self._save_settings,
        )
        back_btn = ft.IconButton(icon=ft.Icons.ARROW_BACK, on_click=self._back_from_settings)

        self._backup_status = ft.Text("", size=12)
        export_btn = ft.OutlinedButton(
            "Export Backup", icon=ft.Icons.DOWNLOAD_OUTLINED, on_click=self._export_backup,
        )
        import_btn = ft.OutlinedButton(
            "Import Backup", icon=ft.Icons.UPLOAD_OUTLINED, on_click=self._import_backup,
        )

        return ft.Column(
            [
                ft.Row(
                    [back_btn, ft.Text("Settings", size=20, color=_INK, font_family=_FONT_DISPLAY)],
                    spacing=4,
                ),
                ft.Text(
                    "Keys are stored locally in data/settings.json — never uploaded "
                    "anywhere by NeuroLingo itself. Changes take effect after restart.",
                    size=12, color=_INK_SOFT,
                ),
                ft.Divider(height=16, color=_DIVIDER),
                self._settings_preferred,
                self._settings_anthropic_key,
                self._settings_openai_key,
                self._settings_gemini_key,
                self._settings_local_model_path,
                ft.Container(height=8),
                ft.Row([save_btn]),
                self._settings_save_status,
                ft.Container(height=16),
                ft.Text("Provider status", size=14, color=_TURQUOISE, font_family=_FONT_DISPLAY),
                self._settings_status_column,
                ft.Container(height=16),
                ft.Text("Data", size=14, color=_TURQUOISE, font_family=_FONT_DISPLAY),
                ft.Text(
                    "Back up every sentence, card, and review history to a JSON file, "
                    "or restore from one.",
                    size=12, color=_INK_SOFT,
                ),
                ft.Row([export_btn, import_btn], spacing=8),
                self._backup_status,
            ],
            spacing=12,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        )

    def _load_settings_into_fields(self) -> None:
        config = LLMConfig.from_file(self.settings_path)
        self._settings_preferred.value = config.preferred_provider
        self._settings_anthropic_key.value = config.anthropic_api_key
        self._settings_openai_key.value = config.openai_api_key
        self._settings_gemini_key.value = config.gemini_api_key
        self._settings_local_model_path.value = config.local_model_path
        self._settings_save_status.value = ""
        self._refresh_settings_status(config)

    def _refresh_settings_status(self, config: LLMConfig) -> None:
        self._settings_status_column.controls.clear()
        providers = [
            ("Anthropic", AnthropicProvider(config)),
            ("OpenAI", OpenAIProvider(config)),
            ("Gemini", GeminiProvider(config)),
            ("Local (llama.cpp)", LocalLlamaProvider(config)),
        ]
        for label, provider in providers:
            available = provider.is_available()
            self._settings_status_column.controls.append(
                ft.Row(
                    [
                        ft.Icon(
                            ft.Icons.CHECK_CIRCLE if available else ft.Icons.CANCEL_OUTLINED,
                            color=_EASY if available else _INK_FAINT,
                            size=16,
                        ),
                        ft.Text(label, size=13),
                        ft.Text(
                            "available" if available else "not configured",
                            size=11, color=_INK_SOFT,
                        ),
                    ],
                    spacing=8,
                )
            )

    def _save_settings(self, _e=None) -> None:
        config = LLMConfig(
            preferred_provider=self._settings_preferred.value or "anthropic",
            anthropic_api_key=(self._settings_anthropic_key.value or "").strip(),
            openai_api_key=(self._settings_openai_key.value or "").strip(),
            gemini_api_key=(self._settings_gemini_key.value or "").strip(),
            local_model_path=(self._settings_local_model_path.value or "").strip(),
        )
        config.save_to_file(self.settings_path)
        _log.info("Settings saved | preferred=%s", config.preferred_provider)

        self._settings_save_status.value = (
            "Saved — restart NeuroLingo for the AI tutor to use these settings."
        )
        self._settings_save_status.color = _EASY
        self._refresh_settings_status(config)
        self.page.update()

    def _go_to_settings(self, _e=None) -> None:
        self._load_settings_into_fields()
        self._content_switcher.content = self._settings_panel
        self.page.update()

    def _back_from_settings(self, _e=None) -> None:
        panels = {
            self.TAB_TODAY: self._today_panel,
            self.TAB_LIBRARY: self._library_panel,
            self.TAB_ADD: self._add_panel,
        }
        self._content_switcher.content = panels[self._tab]
        self.page.update()

    async def _export_backup(self, _e=None) -> None:
        """Serialise every sentence/card/review-log into a JSON file and let
        the user save it via the browser's/OS's native save dialog."""
        backup = export_backup(self.repo)
        payload = json.dumps(backup, ensure_ascii=False, indent=2).encode("utf-8")
        file_name = f"neurolingo_backup_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json"
        try:
            await self._file_picker.save_file(
                dialog_title="Export NeuroLingo Backup",
                file_name=file_name,
                src_bytes=payload,
                allowed_extensions=["json"],
            )
            self._backup_status.value = f"Exported {len(backup['sentences'])} sentence(s)."
            self._backup_status.color = _EASY
        except Exception:
            _log.exception("Backup export failed")
            self._backup_status.value = "Export failed — check logs/neurolingo.log."
            self._backup_status.color = _AGAIN
        self.page.update()

    async def _import_backup(self, _e=None) -> None:
        """Let the user pick a previously-exported JSON file and restore it
        into the current database (adds new rows; never overwrites)."""
        try:
            files = await self._file_picker.pick_files(
                dialog_title="Import NeuroLingo Backup",
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=["json"],
                with_data=True,
            )
            if not files:
                return  # user cancelled the dialog

            data = json.loads(files[0].bytes.decode("utf-8"))
            imported = import_backup(self.repo, data)
            self._backup_status.value = f"Imported {imported} sentence(s)."
            self._backup_status.color = _EASY
        except BackupFormatError as exc:
            self._backup_status.value = f"Not a valid backup file: {exc}"
            self._backup_status.color = _AGAIN
        except Exception:
            _log.exception("Backup import failed")
            self._backup_status.value = "Import failed — check logs/neurolingo.log."
            self._backup_status.color = _AGAIN
        self.page.update()


# ── Entry point ───────────────────────────────────────────────────────────────

def main(page: ft.Page) -> None:
    try:
        repo = DatabaseRepository(DB_PATH)
        repo.create_schema()
        _seed(repo)

        config = LLMConfig.from_file(SETTINGS_PATH)
        router = LLMRouter(config)
        embedder = build_embedding_provider(config.openai_api_key)
        store = _prepare_vector_store(VECTOR_STORE_PATH, embedder)
        rag = RAGManager(store, embedder, router)
        if len(store) == 0:
            try:
                _seed_knowledge(rag)
            except Exception:
                # A cloud embedder can fail here (unreachable/invalid key) in
                # a way HashingEmbeddingProvider never could (pure local
                # computation) — don't let that take down the whole app, just
                # the tutor's grounding context for this session.
                _log.exception(
                    "Knowledge base seeding failed — AI tutor will have no "
                    "grounding context this session"
                )

        NeuroLingoApp(page, repo, rag, SETTINGS_PATH)
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
            # Required for the Shadowing exercise: the browser can't write to
            # the server's filesystem directly, so flet_audio_recorder streams
            # raw PCM16 bytes to this upload endpoint instead.
            upload_dir="data/uploads",
        )
    except Exception:
        log.exception("Fatal error in ft.app()")
        sys.exit(1)
