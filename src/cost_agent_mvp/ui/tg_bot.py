# src/ui/telegram_bot.py
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ChatAction, ContentType
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from dotenv import load_dotenv

from src.agent.orchestrator import Orchestrator

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
USE_LLM_ANALYST_DEFAULT = os.getenv("USE_LLM_ANALYST", "1").strip() not in (
    "0",
    "false",
    "False",
    "no",
    "NO",
)

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in environment / .env")
if not CSV_PATH:
    raise RuntimeError("CSV_PATH is not set in environment / .env")
if not Path(CSV_PATH).exists():
    raise RuntimeError(f"CSV_PATH does not exist: {CSV_PATH}")

bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

orchestrator = Orchestrator(
    outputs_root=OUTPUTS_ROOT,
    dataset_name=DATASET_NAME,
)


# -----------------------------
# Session state (minimal)
# -----------------------------


@dataclass
class UserSession:
    report_day: date | None = None  # if None -> yesterday (UTC)
    use_llm: bool = USE_LLM_ANALYST_DEFAULT
    show_debug: bool = False


user_sessions: dict[int, UserSession] = {}


def get_session(user_id: int) -> UserSession:
    if user_id not in user_sessions:
        user_sessions[user_id] = UserSession()
    return user_sessions[user_id]


def compute_yesterday_utc() -> date:
    return datetime.utcnow().date() - timedelta(days=1)


def parse_yyyy_mm_dd(s: str) -> date:
    # strict YYYY-MM-DD
    return datetime.strptime(s.strip(), "%Y-%m-%d").date()


def build_main_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="📊 Standard daily report (yesterday)",
                    callback_data="report:yesterday",
                )
            ],
            [
                InlineKeyboardButton(
                    text="📅 Set report date (use /setdate YYYY-MM-DD)",
                    callback_data="help:setdate",
                )
            ],
            [InlineKeyboardButton(text="🧪 Toggle LLM summary", callback_data="toggle:llm")],
            [InlineKeyboardButton(text="🧾 Toggle debug (verifier)", callback_data="toggle:debug")],
        ]
    )


def split_text_for_telegram(text: str, max_len: int = 3900) -> list[str]:
    """
    Telegram hard limit is 4096; we keep margin for safety.
    Splits on paragraph boundaries when possible.
    """
    text = text or ""
    if len(text) <= max_len:
        return [text]

    parts = []
    current = []
    cur_len = 0
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue

        # +2 for spacing
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

    # final fallback for huge single blocks
    out = []
    for p in parts:
        if len(p) <= max_len:
            out.append(p)
        else:
            for i in range(0, len(p), max_len):
                out.append(p[i : i + max_len])
    return out


async def send_text_chunked(chat_id: int, text: str) -> None:
    for chunk in split_text_for_telegram(text):
        await bot.send_message(chat_id, chunk)


async def send_run_artifacts(
    message_or_chat_id: int,
    *,
    answer_text: str,
    dashboard_png: str | None,
    run_id: str,
    verifier_status: str,
    show_debug: bool,
) -> None:
    # Send narrative
    await send_text_chunked(message_or_chat_id, answer_text)

    # Send dashboard
    if dashboard_png and Path(dashboard_png).exists():
        await bot.send_photo(
            message_or_chat_id,
            FSInputFile(dashboard_png),
            caption=f"📎 Dashboard (run_id={run_id})",
        )
    else:
        await bot.send_message(
            message_or_chat_id, f"ℹ️ No dashboard image produced (run_id={run_id})."
        )

    # Optional debug
    if show_debug:
        await bot.send_message(
            message_or_chat_id,
            f"🧾 Verifier: {verifier_status}\nRun ID: {run_id}",
        )


# -----------------------------
# Commands
# -----------------------------


@dp.message(Command("start"))
async def cmd_start(message: Message) -> None:
    session = get_session(message.from_user.id)
    session.report_day = None
    session.use_llm = USE_LLM_ANALYST_DEFAULT
    session.show_debug = False

    text = (
        "👋 Hi! I’m your cost-monitoring companion.\n\n"
        "I can:\n"
        "• Generate the standard daily report (button)\n"
        "• Answer ad-hoc questions about costs/usage (type a question)\n\n"
        "Defaults:\n"
        f"• report day: yesterday (UTC)\n"
        f"• LLM summary: {'ON' if session.use_llm else 'OFF'}\n\n"
        "Use:\n"
        "• /report — run report\n"
        "• /setdate YYYY-MM-DD — set report day\n"
        "• /help — examples\n"
    )
    await message.answer(text, reply_markup=build_main_keyboard())


@dp.message(Command("help"))
async def cmd_help(message: Message) -> None:
    text = (
        "📌 Examples you can ask:\n"
        "• Why did costs spike yesterday?\n"
        "• Show top accounts by total cost yesterday\n"
        "• Compare WhatsApp vs Telegram costs yesterday\n"
        "• Account 123 cost breakdown yesterday\n\n"
        "Buttons:\n"
        "• Standard daily report (yesterday)\n\n"
        "Commands:\n"
        "• /report\n"
        "• /setdate YYYY-MM-DD\n"
        "• /llm on|off\n"
        "• /debug on|off\n"
    )
    await message.answer(text, reply_markup=build_main_keyboard())


@dp.message(Command("setdate"))
async def cmd_setdate(message: Message) -> None:
    session = get_session(message.from_user.id)
    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2:
        await message.answer("Usage: /setdate YYYY-MM-DD")
        return

    try:
        d = parse_yyyy_mm_dd(parts[1])
    except Exception:
        await message.answer("Invalid date format. Use YYYY-MM-DD (e.g., 2025-12-02).")
        return

    session.report_day = d
    await message.answer(
        f"✅ Report day set to: {d.isoformat()}.\nNow run /report or press the button.",
        reply_markup=build_main_keyboard(),
    )


@dp.message(Command("llm"))
async def cmd_llm(message: Message) -> None:
    session = get_session(message.from_user.id)
    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2:
        await message.answer(
            f"LLM summary is currently {'ON' if session.use_llm else 'OFF'}. Usage: /llm on|off"
        )
        return
    v = parts[1].strip().lower()
    if v in ("on", "1", "true", "yes"):
        session.use_llm = True
    elif v in ("off", "0", "false", "no"):
        session.use_llm = False
    else:
        await message.answer("Usage: /llm on|off")
        return
    await message.answer(
        f"✅ LLM summary: {'ON' if session.use_llm else 'OFF'}",
        reply_markup=build_main_keyboard(),
    )


@dp.message(Command("debug"))
async def cmd_debug(message: Message) -> None:
    session = get_session(message.from_user.id)
    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2:
        await message.answer(
            f"Debug is currently {'ON' if session.show_debug else 'OFF'}. Usage: /debug on|off"
        )
        return
    v = parts[1].strip().lower()
    if v in ("on", "1", "true", "yes"):
        session.show_debug = True
    elif v in ("off", "0", "false", "no"):
        session.show_debug = False
    else:
        await message.answer("Usage: /debug on|off")
        return
    await message.answer(
        f"✅ Debug: {'ON' if session.show_debug else 'OFF'}",
        reply_markup=build_main_keyboard(),
    )


@dp.message(Command("report"))
async def cmd_report(message: Message) -> None:
    session = get_session(message.from_user.id)
    report_day = session.report_day or compute_yesterday_utc()

    await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)

    try:
        out = orchestrator.run_button_standard_daily(
            csv_path=CSV_PATH,
            report_day=report_day,
            use_llm_analyst=session.use_llm,
            build_dashboard=True,
        )
        await send_run_artifacts(
            message.chat.id,
            answer_text=out.answer_text,
            dashboard_png=out.dashboard_png,
            run_id=out.run_id,
            verifier_status=out.verifier.status,
            show_debug=session.show_debug,
        )
    except Exception as exc:
        logger.exception("Failed to run daily report: %s", exc)
        await message.answer("❌ Error while generating report. Check server logs.")


# -----------------------------
# Callbacks (buttons)
# -----------------------------


@dp.callback_query(F.data.startswith("report:"))
async def cb_report(callback: CallbackQuery) -> None:
    session = get_session(callback.from_user.id)
    _, arg = callback.data.split(":", 1)

    if arg == "yesterday":
        report_day = compute_yesterday_utc()
        session.report_day = None  # keep default
    else:
        report_day = session.report_day or compute_yesterday_utc()

    await callback.answer()
    await bot.send_chat_action(chat_id=callback.message.chat.id, action=ChatAction.TYPING)

    try:
        out = orchestrator.run_button_standard_daily(
            csv_path=CSV_PATH,
            report_day=report_day,
            use_llm_analyst=session.use_llm,
            build_dashboard=True,
        )
        await send_run_artifacts(
            callback.message.chat.id,
            answer_text=out.answer_text,
            dashboard_png=out.dashboard_png,
            run_id=out.run_id,
            verifier_status=out.verifier.status,
            show_debug=session.show_debug,
        )
    except Exception as exc:
        logger.exception("Failed to run button report: %s", exc)
        await bot.send_message(
            callback.message.chat.id,
            "❌ Error while generating report. Check server logs.",
        )


@dp.callback_query(F.data == "toggle:llm")
async def cb_toggle_llm(callback: CallbackQuery) -> None:
    session = get_session(callback.from_user.id)
    session.use_llm = not session.use_llm
    await callback.answer(f"LLM summary: {'ON' if session.use_llm else 'OFF'}")
    await bot.send_message(
        callback.message.chat.id,
        f"✅ LLM summary: {'ON' if session.use_llm else 'OFF'}",
    )


@dp.callback_query(F.data == "toggle:debug")
async def cb_toggle_debug(callback: CallbackQuery) -> None:
    session = get_session(callback.from_user.id)
    session.show_debug = not session.show_debug
    await callback.answer(f"Debug: {'ON' if session.show_debug else 'OFF'}")
    await bot.send_message(
        callback.message.chat.id, f"✅ Debug: {'ON' if session.show_debug else 'OFF'}"
    )


@dp.callback_query(F.data == "help:setdate")
async def cb_help_setdate(callback: CallbackQuery) -> None:
    await callback.answer()
    await bot.send_message(
        callback.message.chat.id,
        "To set a specific report day, use:\n/setdate YYYY-MM-DD\nExample:\n/setdate 2025-12-02",
    )


# -----------------------------
# Default handler (ad-hoc Q&A)
# -----------------------------


@dp.message()
async def handle_message(message: Message) -> None:
    # Only handle text ad-hoc questions
    if message.content_type != ContentType.TEXT or not message.text:
        await message.answer("Please send a text message (questions only).")
        return

    text = message.text.strip()
    if not text:
        return

    session = get_session(message.from_user.id)

    # typing indicator
    await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)

    try:
        out = orchestrator.run_ad_hoc(
            csv_path=CSV_PATH,
            user_text=text,
            use_llm_analyst=session.use_llm,
            build_dashboard=True,
        )
        await send_run_artifacts(
            message.chat.id,
            answer_text=out.answer_text,
            dashboard_png=out.dashboard_png,
            run_id=out.run_id,
            verifier_status=out.verifier.status,
            show_debug=session.show_debug,
        )
    except Exception as exc:
        logger.exception("Failed to answer ad-hoc question: %s", exc)
        await message.answer("❌ Error while answering. Check server logs.")


# -----------------------------
# Entrypoint
# -----------------------------


async def main() -> None:
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
