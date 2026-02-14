from __future__ import annotations

import asyncio
import calendar
import logging
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ChatAction
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from dotenv import load_dotenv

from cost_agent_mvp.agent.orchestrator import Orchestrator
from cost_agent_mvp.agent.planner import Planner
from cost_agent_mvp.reports.daily_report import run_standard_daily_report

# -----------------------------
# Config / Logging
# -----------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("telegram_bot")

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CSV_PATH = os.getenv("CSV_PATH", "").strip()

# Optional
OUTPUTS_ROOT = os.getenv("OUTPUTS_ROOT", "outputs/runs").strip()
DATASET_NAME = os.getenv("DATASET_NAME", "joint_costs_daily").strip()
TEMPLATES_PATH = os.getenv("REPORT_TEMPLATES_PATH", "configs/report_templates.yaml").strip()
TEMPLATE_ID = os.getenv("REPORT_TEMPLATE_ID", "standard_daily_report").strip()
# Add env vars (after fixing whitespace!)
SEMANTIC_LAYER_PATH = os.getenv("SEMANTIC_LAYER_PATH", "configs/semantic_layer.yaml").strip()
PROMPTS_DIR = os.getenv("PROMPTS_DIR", "prompts").strip()

_original_planner_plan = Planner.plan


def _patched_plan(
    self,
    *,
    user_text: str,
    semantic_layer: Any,
    mock_mode: bool = False,
    **kwargs: Any,
) -> Any:
    """Wrapper that safely ignores unexpected kwargs (like 'anchor_day') passed by orchestrator."""
    # Remove orchestrator bugs that pass invalid kwargs
    kwargs.pop("anchor_day", None)  # Critical fix for this error
    # Pass only valid kwargs to original method
    return _original_planner_plan(
        self,
        user_text=user_text,
        semantic_layer=semantic_layer,
        mock_mode=mock_mode,
        **kwargs,
    )


Planner.plan = _patched_plan  # type: ignore[method-assign]


# Initialize orchestrator (module-level)
orch = Orchestrator(
    outputs_root=OUTPUTS_ROOT,
    dataset_name=DATASET_NAME,
    semantic_layer_path=SEMANTIC_LAYER_PATH,
    prompts_dir=PROMPTS_DIR,
)

# Narrative mode for report generation
REPORT_MODE = os.getenv("REPORT_MODE", "llm").strip().lower()  # deterministic|llm
if REPORT_MODE not in ("deterministic", "llm"):
    REPORT_MODE = "llm"
MOCK_LLM = os.getenv("MOCK_LLM", "0").strip().lower() in ("1", "true", "yes", "on")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in environment / .env")
if not CSV_PATH:
    raise RuntimeError("CSV_PATH is not set in environment / .env")
if not Path(CSV_PATH).exists():
    raise RuntimeError(f"CSV_PATH does not exist: {CSV_PATH}")

bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

# -----------------------------
# Session state
# -----------------------------


@dataclass
class UserSession:
    report_day: date | None = None  # selected day in picker or /setdate
    cal_year: int | None = None
    cal_month: int | None = None

    # edit-in-place "UI screen" tracking
    ui_message_id: int | None = None
    ui_chat_id: int | None = None


user_sessions: dict[int, UserSession] = {}


def get_session(user_id: int) -> UserSession:
    if user_id not in user_sessions:
        user_sessions[user_id] = UserSession()
    return user_sessions[user_id]


def parse_yyyy_mm_dd(s: str) -> date:
    return datetime.strptime(s.strip(), "%Y-%m-%d").date()


# -----------------------------
# Dataset date range cache
# -----------------------------

_DATASET_DATE_RANGE: tuple[date, date] | None = None


def get_dataset_date_range() -> tuple[date, date]:
    """Returns (min_date, max_date) available in CSV_PATH. Cached in-memory."""
    global _DATASET_DATE_RANGE
    if _DATASET_DATE_RANGE is not None:
        return _DATASET_DATE_RANGE

    import pandas as pd

    df_dates = pd.read_csv(CSV_PATH, usecols=["date"])
    ds = pd.to_datetime(df_dates["date"], errors="coerce").dropna().dt.date
    if ds.empty:
        # fallback: today-1 (rare), but keeps bot working
        y = datetime.utcnow().date() - timedelta(days=1)
        _DATASET_DATE_RANGE = (y, y)
    else:
        _DATASET_DATE_RANGE = (min(ds), max(ds))
    return _DATASET_DATE_RANGE


def clamp_to_range(d: date, lo: date, hi: date) -> date:
    if d < lo:
        return lo
    if d > hi:
        return hi
    return d


def last_available_day() -> date:
    _, hi = get_dataset_date_range()
    return hi


# -----------------------------
# UI helpers
# -----------------------------

HOME_TEXT = "🏠 <b>Choose next action:</b>"
CHAT_SOON_TEXT = (
    "🧠 <b>Chat with companion</b> is coming soon.\n\n"
    "For now, please use:\n"
    "• 📊 Daily KPI report (latest)\n"
    "• 📅 Daily KPI report (pick a date)"
)


def build_main_keyboard() -> InlineKeyboardMarkup:
    # 3-button UX
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="📊 Daily KPI report (latest)",
                    callback_data="report:latest",
                )
            ],
            [
                InlineKeyboardButton(
                    text="📅 Daily KPI report (pick a date)",
                    callback_data="cal:open",
                )
            ],
            [
                InlineKeyboardButton(
                    text="🧠 Chat with companion",
                    callback_data="chat:qa",
                )
            ],
        ]
    )


def _month_nav(year: int, month: int, delta: int) -> tuple[int, int]:
    # delta in months (±1)
    m = month + delta
    y = year
    if m < 1:
        m = 12
        y -= 1
    elif m > 12:
        m = 1
        y += 1
    return y, m


def build_calendar_keyboard(selected: date, year: int, month: int) -> InlineKeyboardMarkup:
    lo, hi = get_dataset_date_range()

    # Header: month/year with nav
    month_name = calendar.month_name[month]
    rows: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton(text="◀️", callback_data="cal:prev"),
            InlineKeyboardButton(text=f"{month_name} {year}", callback_data="noop"),
            InlineKeyboardButton(text="▶️", callback_data="cal:next"),
        ],
        [
            InlineKeyboardButton(text="Mo", callback_data="noop"),
            InlineKeyboardButton(text="Tu", callback_data="noop"),
            InlineKeyboardButton(text="We", callback_data="noop"),
            InlineKeyboardButton(text="Th", callback_data="noop"),
            InlineKeyboardButton(text="Fr", callback_data="noop"),
            InlineKeyboardButton(text="Sa", callback_data="noop"),
            InlineKeyboardButton(text="Su", callback_data="noop"),
        ],
    ]

    # Calendar grid (weeks start Monday)
    cal = calendar.Calendar(firstweekday=0)
    week: list[InlineKeyboardButton] = []
    for d in cal.itermonthdates(year, month):
        if d.month != month:
            week.append(InlineKeyboardButton(text=" ", callback_data="noop"))
        else:
            in_range = lo <= d <= hi
            if not in_range:
                week.append(InlineKeyboardButton(text="·", callback_data="noop"))
            else:
                label = str(d.day)
                if d == selected:
                    label = f"✅{label}"
                week.append(
                    InlineKeyboardButton(text=label, callback_data=f"cal:pick:{d.isoformat()}")
                )

        if len(week) == 7:
            rows.append(week)
            week = []

    # Actions row
    rows.append(
        [InlineKeyboardButton(text=f"✅ Run {selected.isoformat()}", callback_data="cal:run")]
    )
    rows.append(
        [
            InlineKeyboardButton(text="🏁 Latest", callback_data="cal:latest"),
            InlineKeyboardButton(text="↩️ Back", callback_data="home"),
        ]
    )

    return InlineKeyboardMarkup(inline_keyboard=rows)


def sanitize_llm_text(text: str) -> str:
    """
    Minimal cleanup for Telegram when we send plain text (no parse_mode):
    - remove common markdown markers like **bold** and `code`
    - collapse trailing spaces
    """
    t = (text or "").strip()
    if not t:
        return ""
    # remove fenced/code markers/backticks
    t = t.replace("```", "").replace("`", "")
    # remove **bold** markers
    t = t.replace("**", "")
    # normalize line endings
    t = "\n".join([ln.rstrip() for ln in t.splitlines()])
    return t


def split_text_for_telegram(text: str, max_len: int = 3900) -> list[str]:
    """Telegram hard limit is 4096; keep margin and split on paragraph boundaries."""
    text = text or ""
    if len(text) <= max_len:
        return [text]

    parts: list[str] = []
    current: list[str] = []
    cur_len = 0

    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        add_len = len(block) + (2 if current else 0)
        if cur_len + add_len <= max_len:
            if current:
                current.append("")
            current.append(block)
            cur_len += add_len
        else:
            if current:
                parts.append("\n\n".join(current))
            current = [block]
            cur_len = len(block)

    if current:
        parts.append("\n\n".join(current))

    out: list[str] = []
    for p in parts:
        if len(p) <= max_len:
            out.append(p)
        else:
            for i in range(0, len(p), max_len):
                out.append(p[i : i + max_len])
    return out


def format_summary_html(raw: str) -> str:
    """
    Telegram-friendly formatting for the deterministic summary.

    Requirements:
    - Title includes date on the same line.
    - One KPI block (selected KPIs only), each KPI on a new line starting with '-'.
    - Exclude churn, dials analyzed, p95, avg margin.
    - Add an empty line before each major block.
    - Separate blocks for Top services and Top accounts.

    IMPORTANT: All numeric values are kept exactly as in the raw summary.
    """
    raw = (raw or "").strip()
    if not raw:
        return ""

    lines = [ln.rstrip() for ln in raw.splitlines()]

    # Extract report date from a line like: Date: YYYY-MM-DD
    report_date: str | None = None
    for ln in lines:
        m = re.match(r"^Date:\s*(\d{4}-\d{2}-\d{2})\s*$", ln.strip())
        if m:
            report_date = m.group(1)
            break

    # Title: use first line if present
    title = "Daily Cost Monitoring Summary"
    if lines and lines[0].strip():
        title = lines[0].strip()

    # Collect KPI lines we want to show
    kpis: list[str] = []

    def add_kpi(prefix_emoji: str, text_line: str) -> None:
        kpis.append(f"- {prefix_emoji} {text_line}")

    # Collect section bullets
    services: list[str] = []
    accounts: list[str] = []

    current_section: str | None = None

    for ln in lines:
        t = ln.strip()
        if not t:
            continue

        # Skip underline
        if set(t) <= {"="}:
            continue

        # Skip date line (we put it in the title)
        if t.startswith("Date:"):
            continue

        # Detect section headings
        if t.endswith(":") and not t.startswith("-"):
            low = t.lower()
            if "top" in low and "service" in low:
                current_section = "services"
            elif "top" in low and "account" in low:
                current_section = "accounts"
            else:
                current_section = None
            continue

        # Section bullets
        if t.startswith("-"):
            item = t.lstrip("-").strip()
            if current_section == "services":
                services.append(f"• {item}")
            elif current_section == "accounts":
                accounts.append(f"• {item}")
            continue

        # KPI candidates (keep exact text, just decorate)
        low = t.lower()
        if low.startswith("total cost"):
            add_kpi("💰", t)
        elif low.startswith("active users"):
            add_kpi("👥", t)
        elif low.startswith("avg cost per active account-day"):
            add_kpi("📉", t)
        elif low.startswith("7-day trend"):
            add_kpi("📈", t)
        elif "above cap" in low:
            add_kpi("🚨", t)
        else:
            # explicitly exclude other KPI-like lines
            continue

    # Build output
    date_part = f" — 🗓️ <b>{report_date}</b>" if report_date else ""
    out: list[str] = [f"📊 <b>{title}</b>{date_part}"]

    if kpis:
        out.append("")
        out.append("📌 <b>KPIs</b>")
        out.extend(kpis)

    # if services:
    #     out.append("")
    #     out.append("🧱 <b>Top 5 services by cost</b>")
    #     out.extend(services)

    if accounts:
        out.append("")
        out.append("👤 <b>Top 5 accounts by cost</b>")
        out.extend(accounts)

    return "\n".join(out)


async def _edit_or_send_ui(
    session: UserSession,
    chat_id: int,
    text: str,
    reply_markup: InlineKeyboardMarkup,
) -> None:
    """
    Edit-in-place UI message when possible, otherwise send a new one
    and remember its message_id.
    """
    if session.ui_chat_id == chat_id and session.ui_message_id is not None:
        try:
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=session.ui_message_id,
                text=text,
                reply_markup=reply_markup,
                parse_mode="HTML",
            )
            return
        except Exception:
            # message can't be edited anymore; fall back to sending a new one
            session.ui_message_id = None
            session.ui_chat_id = None

    msg = await bot.send_message(chat_id, text, reply_markup=reply_markup, parse_mode="HTML")
    session.ui_message_id = msg.message_id
    session.ui_chat_id = chat_id


async def show_home(session: UserSession, chat_id: int) -> None:
    await _edit_or_send_ui(session, chat_id, HOME_TEXT, build_main_keyboard())


async def show_calendar(session: UserSession, chat_id: int) -> None:
    lo, hi = get_dataset_date_range()

    selected = session.report_day or hi
    selected = clamp_to_range(selected, lo, hi)
    session.report_day = selected

    y = session.cal_year or selected.year
    m = session.cal_month or selected.month
    session.cal_year, session.cal_month = y, m

    text = (
        "📅 <b>Pick a date</b>\n"
        f"Available: {lo.isoformat()} … {hi.isoformat()}\n"
        f"Selected: <b>{selected.isoformat()}</b>"
    )
    await _edit_or_send_ui(session, chat_id, text, build_calendar_keyboard(selected, y, m))


async def send_text_chunked(chat_id: int, text: str, *, parse_mode: str | None = None) -> None:
    for chunk in split_text_for_telegram(text):
        await bot.send_message(chat_id, chunk, parse_mode=parse_mode)


async def run_and_send_daily_report(
    session: UserSession,
    chat_id: int,
    report_day: date,
    *,
    show_menu_after: bool = False,
) -> None:
    """Runs the daily report and sends dashboard + summary to Telegram."""
    await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    artifacts = run_standard_daily_report(
        csv_path=CSV_PATH,
        report_day=report_day,
        output_root=OUTPUTS_ROOT,
        dataset_name=DATASET_NAME,
        report_templates_path=TEMPLATES_PATH,
        template_id=TEMPLATE_ID,
        narrative_mode=REPORT_MODE,
        mock_llm=MOCK_LLM,
    )

    summary_path = Path(artifacts.summary_txt)
    dash_path = Path(artifacts.dashboard_png)

    raw = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
    if REPORT_MODE == "deterministic":
        # Deterministic summary can use HTML formatting for readability
        clean_text = format_summary_html(raw)
    else:
        # LLM summary: send plain text
        clean_text = sanitize_llm_text(raw)

        # 1) Send dashboard PNG with a very short caption (avoid duplicating summary)
        if dash_path.exists():
            await bot.send_photo(
                chat_id,
                FSInputFile(str(dash_path)),
                caption=f"This is KPI report for the {report_day.isoformat()}",
            )
        else:
            await bot.send_message(chat_id, "ℹ️ dashboard.png not found in run folder.")

        # Send summary (chunked, plain text)
        if clean_text:
            # If the message is still too long, Telegram will receive it in chunks.
            if REPORT_MODE == "deterministic":
                await send_text_chunked(chat_id, clean_text, parse_mode="HTML")
            else:
                await send_text_chunked(chat_id, clean_text)
        else:
            await bot.send_message(chat_id, "ℹ️ summary.txt not found in run folder.")

        # Run folder hint (short)
        # await bot.send_message(chat_id, f"✅ Done. Run folder: {artifacts.output_dir}")

        # Return user to Home UI (edit-in-place)
        await show_home(session, chat_id)
        if show_menu_after:
            await bot.send_message(
                chat_id,
                "🏠 Choose next action:",
                reply_markup=build_main_keyboard(),
                parse_mode="HTML",
            )


# -----------------------------
# Commands
# -----------------------------


@dp.message(Command("start"))
async def cmd_start(message: Message) -> None:
    session = get_session(message.from_user.id)
    session.report_day = None
    session.cal_year = None
    session.cal_month = None

    lo, hi = get_dataset_date_range()
    intro = (
        "👋 Hi! I can generate a deterministic daily KPI report.\n\n"
        f"🗂️ Dataset range: <b>{lo.isoformat()} … {hi.isoformat()}</b>\n\n"
        "Tip: use <b>Pick a date</b> to select from the dataset.\n"
    )
    # Put intro as a regular message; UI is edit-in-place separately.
    await message.answer(intro, parse_mode="HTML")
    await show_home(session, message.chat.id)


@dp.message(Command("setdate"))
async def cmd_setdate(message: Message) -> None:
    session = get_session(message.from_user.id)
    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2:
        await message.answer("Usage: /setdate YYYY-MM-DD (example: /setdate 2025-07-20)")
        return

    try:
        d = parse_yyyy_mm_dd(parts[1])
    except Exception:
        await message.answer("Invalid date format. Use YYYY-MM-DD (e.g., 2025-12-02).")
        return

    lo, hi = get_dataset_date_range()
    if d < lo or d > hi:
        await message.answer(
            f"Date {d.isoformat()} is outside the dataset range "
            f"({lo.isoformat()} … {hi.isoformat()})."
        )
        return

    session.report_day = d
    session.cal_year = d.year
    session.cal_month = d.month
    await message.answer(f"✅ Report day set to: {d.isoformat()}", parse_mode="HTML")
    await show_home(session, message.chat.id)


# -----------------------------
# Callbacks (buttons)
# -----------------------------


@dp.callback_query(F.data == "home")
async def cb_home(callback: CallbackQuery) -> None:
    session = get_session(callback.from_user.id)
    await callback.answer()
    await show_home(session, callback.message.chat.id)


@dp.callback_query(F.data == "chat:qa")
async def cb_chat_qa(callback: CallbackQuery) -> None:
    session = get_session(callback.from_user.id)
    await callback.answer()

    examples = (
        "💬 Ask me anything about costs! Examples:\n\n"
        "• What was total cost yesterday?\n"
        "• Top 5 accounts by cost on 2025-07-20\n"
        "• Compare whatsapp vs telegram costs last week\n"
        "• Which account had highest classification costs?\n"
        "• Show trend of total costs for last 7 days\n\n"
        "Type your question below 👇"
    )
    await bot.send_message(callback.message.chat.id, examples)
    await show_home(session, callback.message.chat.id)


@dp.callback_query(F.data == "report:latest")
async def cb_report_latest(callback: CallbackQuery) -> None:
    session = get_session(callback.from_user.id)
    d = last_available_day()
    session.report_day = d
    session.cal_year = d.year
    session.cal_month = d.month

    await callback.answer()
    status = await bot.send_message(callback.message.chat.id, "⏳ Generating KPI report…")
    try:
        await run_and_send_daily_report(session, callback.message.chat.id, d)
    except Exception as exc:
        logger.exception("Failed to run latest report: %s", exc)
        await bot.send_message(
            callback.message.chat.id,
            "❌ Error while generating report. Check server logs.",
        )
    finally:
        # Try to remove the status message (ignore if it fails)
        try:
            await bot.delete_message(callback.message.chat.id, status.message_id)
        except Exception:
            pass


@dp.callback_query(F.data == "cal:open")
async def cb_cal_open(callback: CallbackQuery) -> None:
    session = get_session(callback.from_user.id)
    await callback.answer()
    await show_calendar(session, callback.message.chat.id)


@dp.callback_query(F.data == "cal:prev")
async def cb_cal_prev(callback: CallbackQuery) -> None:
    session = get_session(callback.from_user.id)
    lo, hi = get_dataset_date_range()

    selected = session.report_day or hi
    selected = clamp_to_range(selected, lo, hi)

    y = session.cal_year or selected.year
    m = session.cal_month or selected.month
    y2, m2 = _month_nav(y, m, -1)
    session.cal_year, session.cal_month = y2, m2

    await callback.answer()
    await show_calendar(session, callback.message.chat.id)


@dp.callback_query(F.data == "cal:next")
async def cb_cal_next(callback: CallbackQuery) -> None:
    session = get_session(callback.from_user.id)
    lo, hi = get_dataset_date_range()

    selected = session.report_day or hi
    selected = clamp_to_range(selected, lo, hi)

    y = session.cal_year or selected.year
    m = session.cal_month or selected.month
    y2, m2 = _month_nav(y, m, +1)
    session.cal_year, session.cal_month = y2, m2

    await callback.answer()
    await show_calendar(session, callback.message.chat.id)


@dp.callback_query(F.data == "cal:latest")
async def cb_cal_latest(callback: CallbackQuery) -> None:
    session = get_session(callback.from_user.id)
    lo, hi = get_dataset_date_range()

    selected = hi
    session.report_day = selected
    session.cal_year, session.cal_month = selected.year, selected.month

    await callback.answer()
    await show_calendar(session, callback.message.chat.id)


@dp.callback_query(F.data.startswith("cal:pick:"))
async def cb_cal_pick(callback: CallbackQuery) -> None:
    session = get_session(callback.from_user.id)
    lo, hi = get_dataset_date_range()

    # cal:pick:YYYY-MM-DD
    try:
        iso = callback.data.split("cal:pick:", 1)[1]
        picked = date.fromisoformat(iso)
    except Exception:
        await callback.answer("Invalid date", show_alert=False)
        return

    picked = clamp_to_range(picked, lo, hi)
    session.report_day = picked
    session.cal_year, session.cal_month = picked.year, picked.month

    await callback.answer(picked.isoformat())
    await show_calendar(session, callback.message.chat.id)


@dp.callback_query(F.data == "cal:run")
async def cb_cal_run(callback: CallbackQuery) -> None:
    session = get_session(callback.from_user.id)
    lo, hi = get_dataset_date_range()

    d = session.report_day or hi
    d = clamp_to_range(d, lo, hi)
    session.report_day = d

    await callback.answer()
    status = await bot.send_message(callback.message.chat.id, "⏳ Generating KPI report…")
    try:
        await run_and_send_daily_report(session, callback.message.chat.id, d, show_menu_after=True)
    except Exception as exc:
        logger.exception("Failed to run selected-day report: %s", exc)
        await bot.send_message(
            callback.message.chat.id,
            "❌ Error while generating report. Check server logs.",
        )
    finally:
        # Try to remove the status message (ignore if it fails)
        try:
            await bot.delete_message(callback.message.chat.id, status.message_id)
        except Exception:
            pass


@dp.callback_query(F.data == "noop")
async def cb_noop(callback: CallbackQuery) -> None:
    await callback.answer()


# -----------------------------
# Scenario: any typed message
# -----------------------------


@dp.message()
async def handle_qa_query(message: Message) -> None:
    if not (message.text or "").strip():
        return

    # Show "thinking" indicator
    await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)

    try:
        # Get latest available day from dataset as anchor
        anchor = last_available_day()

        # Run REAL ad-hoc query through orchestrator
        out = orch.run_ad_hoc(
            csv_path=CSV_PATH,
            user_text=message.text.strip(),
            anchor_day=anchor,
            output_root=OUTPUTS_ROOT,
            mock_planner=False,  # REAL planner (not mocked!)
            mock_analyst=False,  # REAL analyst (not mocked!)
            mode="llm" if not MOCK_LLM else "deterministic",
            build_dashboard=False,
        )

        # Format answer with verification status
        answer = out.answer_text.strip()
        if not out.verifier.passed:
            if out.verifier.status == "UNSUPPORTED":
                answer = f"❌ Unsupported query: {answer}"
            else:
                answer = f"⚠️ Answer with low confidence:\n\n{answer}"

        # Send answer (chunked for Telegram limits)
        await send_text_chunked(message.chat.id, answer)

        # Show menu again
        session = get_session(message.from_user.id)
        await show_home(session, message.chat.id)

    except Exception as exc:
        logger.exception("QA query failed: %s", exc)
        error_msg = "❌ Error processing your query. Please try rephrasing."
        if "unsupported" in str(exc).lower() or "not allowed" in str(exc).lower():
            error_msg = f"❌ Unsupported query: {str(exc)}"
        await bot.send_message(message.chat.id, error_msg)
        session = get_session(message.from_user.id)
        await show_home(session, message.chat.id)


# -----------------------------
# Entrypoint
# -----------------------------


async def main() -> None:
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
