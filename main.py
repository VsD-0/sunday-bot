"""
Корейский репетитор — Telegram бот с Mini App
Архитектура:
- Бэкенд: FastAPI + aiogram
- База данных: Supabase PostgreSQL (asyncpg)
- LLM: Groq API (qwen3.8-27b)
- TTS: edge_tts (ko-KR-SunHiNeural)
- Распознавание речи: Groq Whisper API
- Хостинг: Render.com
- Фронтенд: GitHub Pages (Mini App)
"""

import asyncio
import logging
import re
import os
import uuid
import tempfile
import edge_tts
from datetime import datetime, timedelta, date
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart, StateFilter
from aiogram.types import FSInputFile, ReplyKeyboardMarkup, KeyboardButton
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.exceptions import TelegramBadRequest
from openai import AsyncOpenAI
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
import asyncpg
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

# ============================================
# 1. КОНФИГУРАЦИЯ
# ============================================
BOT_TOKEN = os.getenv("BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "openai/gpt-oss-120b"

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
scheduler = AsyncIOScheduler()

# ============================================
# 2. FASTAPI (Mini App)
# ============================================
api = FastAPI()
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальный пул подключений к Supabase
db_pool = None


async def init_db():
    """Инициализация пула подключений к Supabase."""
    global db_pool
    database_url = os.getenv("DATABASE_URL")
    db_pool = await asyncpg.create_pool(
        database_url + "?sslmode=require",
        min_size=2,
        max_size=10,
        server_settings={'jit': 'off'}
    )
    logging.info("Подключение к Supabase установлено.")


# --- Health Check ---
@api.get("/health")
async def health_check():
    return {"status": "ok", "time": datetime.now().isoformat()}


# --- TTS (озвучка) ---
@api.get("/api/tts")
async def text_to_speech(text: str = Query(...)):
    """Генерирует аудио для корейского текста."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
        tmp_path = tmp.name

    try:
        voice = "ko-KR-SunHiNeural"
        communicate = edge_tts.Communicate(text, voice, rate="-20%")
        await communicate.save(tmp_path)
        return FileResponse(tmp_path, media_type="audio/mpeg", filename=f"tts_{text}.mp3")
    except Exception as e:
        logging.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail="Ошибка генерации аудио")
    finally:
        async def cleanup():
            await asyncio.sleep(10)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        asyncio.create_task(cleanup())


# --- Словарь ---
@api.get("/api/vocabulary")
async def get_vocabulary(user_id: int = Query(...)):
    """Возвращает последние 20 слов пользователя."""
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT word, translation, next_review FROM vocabulary WHERE user_id = $1 ORDER BY id DESC LIMIT 20",
            user_id
        )
        return [
            {"word": r["word"], "translation": r["translation"], "next_review": str(r["next_review"])}
            for r in rows
        ]


# --- Карточки (для Mini App) ---
@api.get("/api/due-words")
async def get_due_words_api(user_id: int = Query(...)):
    """Возвращает слова для повторения (next_review <= now())."""
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, word, translation, interval, next_review FROM vocabulary WHERE user_id = $1 AND next_review <= now() ORDER BY next_review ASC LIMIT 20",
            user_id
        )
        return [
            {
                "id": r["id"],
                "word": r["word"],
                "translation": r["translation"],
                "interval": r["interval"],
                "next_review": str(r["next_review"])
            }
            for r in rows
        ]


@api.post("/api/update-srs")
async def update_srs_api(data: dict):
    """Обновляет интервал повторения после ответа на карточке."""
    user_id = data.get("user_id")
    word_id = data.get("word_id")
    is_correct = data.get("is_correct")

    if not all([user_id, word_id, is_correct is not None]):
        raise HTTPException(status_code=400, detail="Missing parameters")

    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT interval FROM vocabulary WHERE id = $1 AND user_id = $2",
            word_id, user_id
        )
        if not row:
            raise HTTPException(status_code=404, detail="Word not found")

        current_interval = row["interval"] or 1
        new_interval = min(current_interval * 2, 30) if is_correct else 1
        next_review = datetime.now() + timedelta(days=new_interval)

        await conn.execute(
            "UPDATE vocabulary SET interval = $1, next_review = $2 WHERE id = $3 AND user_id = $4",
            new_interval, next_review, word_id, user_id
        )

        return {"success": True, "new_interval": new_interval}


# --- Статистика ---
@api.get("/api/stats")
async def get_stats(user_id: int = Query(...)):
    """Возвращает статистику обучения."""
    async with db_pool.acquire() as conn:
        total = await conn.fetchval("SELECT COUNT(*) FROM vocabulary WHERE user_id = $1", user_id) or 0
        due = await conn.fetchval("SELECT COUNT(*) FROM vocabulary WHERE user_id = $1 AND next_review <= now()",
                                  user_id) or 0

        # Streak
        activity = await conn.fetch(
            "SELECT DISTINCT DATE(created_at) as day FROM vocabulary WHERE user_id = $1 ORDER BY day DESC LIMIT 60",
            user_id
        )
        streak = 0
        if activity:
            today = date.today()
            last_day = activity[0]['day']
            if last_day == today or last_day == today - timedelta(days=1):
                streak = 1
                for i in range(1, len(activity)):
                    expected = last_day - timedelta(days=i)
                    if activity[i]['day'] == expected:
                        streak += 1
                    else:
                        break

        # Прогресс за неделю
        week_data = await conn.fetch(
            "SELECT DATE(created_at) as day, COUNT(*) as count FROM vocabulary WHERE user_id = $1 AND created_at >= now() - INTERVAL '7 days' GROUP BY DATE(created_at) ORDER BY day",
            user_id
        )
        weekly = []
        for i in range(6, -1, -1):
            day = today - timedelta(days=i)
            count = next((r['count'] for r in week_data if r['day'] == day), 0)
            weekly.append({'day': day.strftime('%d.%m'), 'count': count})

        return {"total": total, "due_today": due, "streak": streak, "weekly": weekly}


# ============================================
# 3. GROQ API (LLM)
# ============================================
client = AsyncOpenAI(
    base_url='https://api.groq.com/openai/v1',
    api_key=GROQ_API_KEY
)

SYSTEM_PROMPT = (
    "Ты — ИИ-репетитор корейского для русскоязычного новичка. Пиши объяснения ТОЛЬКО НА РУССКОМ. "
    "Корейские слова пиши хангылем (слоги 가-힣 и буквы алфавита ㄱ, ㄴ, ㄷ...), "
    "в скобках давай произношение кириллицей и перевод. "
    "Если ученик ошибся — мягко исправь. Будь краток. "
    "НИКОГДА не используй теги и разметку вроде [TTS] — пиши обычным текстом, оформление бот добавит сам. "
    "Если просят алфавит хангыль — перечисли согласные: ㄱ ㄴ ㄷ ㄹ ㅁ ㅂ ㅅ ㅇ ㅈ ㅊ ㅋ ㅌ ㅍ ㅎ "
    "и гласные: ㅏ ㅑ ㅓ ㅕ ㅗ ㅛ ㅜ ㅠ ㅡ ㅣ. "
    "КРИТИЧЕСКОЕ ПРАВИЛО: если система сообщает, что распознанный текст СОВПАДАЕТ с целевым словом, "
    "ты НЕ ИМЕЕШЬ ПРАВА писать 'почти' — хвали ученика и давай следующее слово."
)

# ============================================
# 4. УТИЛИТЫ ДЛЯ ОБРАБОТКИ ТЕКСТА
# ============================================
HANGUL = '가-힣㄰-㆏'
HANGUL_RE = re.compile(f'[{HANGUL}]+')


def strip_tts_garbage(text: str) -> str:
    """Вырезаем теги [TTS] и мусор вида [TT5a]."""
    text = re.sub(r'\[/?TTS\]', '', text, flags=re.I)
    text = re.sub(r'\[[A-Za-z0-9/]{1,12}\]', '', text)
    text = re.sub(r'\bttss?\b', '', text, flags=re.I)
    text = re.sub(r'\s*/\s*(?=\s|$)', '', text)
    return text


def extract_voice(text: str) -> str:
    """Извлекает слово для озвучки."""
    m = re.findall(r'\[TTS\](.*?)\[/TTS\]', text, flags=re.I)
    if m:
        return m[-1].strip()
    cleaned = strip_tts_garbage(text)
    m = re.search(rf'(?:слово|повтори)[^{HANGUL}]*?([{HANGUL}]+)', cleaned, re.I)
    if m:
        return m.group(1)
    runs = HANGUL_RE.findall(cleaned)
    if len(runs) > 4:
        return ''
    return runs[-1] if runs else ''


def strip_model_tags(text: str) -> str:
    """Удаляет HTML-теги, которые модель вставила сама."""
    return re.sub(r'</?[a-zA-Z][^>]*>', '', text)


def to_html(text: str) -> str:
    """Форматирует текст для Telegram (выделяет хангыль жирным)."""
    text = strip_model_tags(text)
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    text = HANGUL_RE.sub(lambda m: f'<b>{m.group(0)}</b>', text)
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ============================================
# 5. РАБОТА С БАЗОЙ ДАННЫХ (Supabase)
# ============================================
async def save_message(user_id: int, role: str, content: str):
    """Сохраняет сообщение в историю."""
    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO history (user_id, role, content) VALUES ($1, $2, $3)",
            user_id, role, content
        )


async def get_history_messages(user_id: int, limit: int = 10) -> list:
    """Получает последние сообщения из истории."""
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT role, content FROM history WHERE user_id = $1 ORDER BY id DESC LIMIT $2",
            user_id, limit
        )
        rows.reverse()
        return [{"role": r["role"], "content": r["content"]} for r in rows]


async def save_word(user_id: int, word: str, translation: str):
    """Сохраняет новое слово в словарь."""
    next_review = datetime.now() + timedelta(days=1)
    async with db_pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO vocabulary (user_id, word, translation, interval, next_review) 
               VALUES ($1, $2, $3, $4, $5)
               ON CONFLICT (user_id, word) DO NOTHING""",
            user_id, word.strip(), translation.strip(), 1, next_review
        )


async def get_due_words(user_id: int):
    """Получает слова для повторения (для бота)."""
    now = datetime.now()
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, word, translation, interval FROM vocabulary WHERE user_id = $1 AND next_review <= $2",
            user_id, now
        )
        return [(r["id"], r["word"], r["translation"], r["interval"]) for r in rows]


async def update_srs(user_id: int, word_id: int, current_interval: int, is_correct: bool):
    """Обновляет интервал повторения (для бота)."""
    new_interval = current_interval * 2 if is_correct else 1
    next_review = datetime.now() + timedelta(days=new_interval)
    async with db_pool.acquire() as conn:
        await conn.execute(
            "UPDATE vocabulary SET interval = $1, next_review = $2 WHERE id = $3 AND user_id = $4",
            new_interval, next_review, word_id, user_id
        )


# ============================================
# 6. ГЕНЕРАЦИЯ ГОЛОСА
# ============================================
async def generate_voice(text: str) -> str:
    """Генерирует аудиофайл через edge_tts."""
    voice = "ko-KR-SunHiNeural"
    filename = f"voice_{uuid.uuid4().hex}.mp3"
    communicate = edge_tts.Communicate(text, voice, rate="-20%")
    await communicate.save(filename)
    return filename


# ============================================
# 7. ОБРАБОТКА СООБЩЕНИЙ (LLM)
# ============================================
async def process_and_reply(message: types.Message, user_id: int, prompt_text: str, is_translation: bool = False,
                            word_to_save: str = None):
    """Отправляет запрос к LLM и обрабатывает ответ."""
    await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")
    try:
        history = await get_history_messages(user_id, limit=6)

        for attempt in range(2):
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages.extend(history)
            messages.append({"role": "user", "content": prompt_text})

            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.6,
                max_tokens=800,
                presence_penalty=0.3,
                frequency_penalty=0.3
            )
            response_text = response.choices[0].message.content

            if re.search(r'скажи слово|повтори', response_text, re.I) and not HANGUL_RE.search(response_text):
                logging.warning(f"Попытка {attempt + 1}: нет хангыля. Перегенерация...")
                prompt_text = ("ОШИБКА. Ты не написал корейское слово хангылем! "
                               "Дай новое слово хангылем, произношение кириллицей и перевод.")
                continue
            break

        voice_text = extract_voice(response_text)
        cleaned = strip_tts_garbage(response_text)
        cleaned = strip_model_tags(cleaned)
        cleaned = cleaned.replace("*", "").replace("`", "")

        await save_message(user_id, "user", prompt_text)
        await save_message(user_id, "assistant", cleaned)

        if is_translation and word_to_save:
            first = cleaned.split('\n')[0].strip().rstrip('.,;:!?-')
            m = re.search(r'[가-힣]+', first)
            korean = m.group(0) if m else word_to_save
            parts = re.split(r'\s*[—–-]\s*', first, maxsplit=1)
            translation = parts[1].strip() if len(parts) > 1 else first
            await save_word(user_id, korean, translation[:100])

        try:
            await message.answer(to_html(cleaned), parse_mode="HTML")
        except TelegramBadRequest:
            await message.answer(cleaned)

        if voice_text:
            await message.bot.send_chat_action(chat_id=message.chat.id, action="record_voice")
            audio_file = await generate_voice(voice_text)
            await message.answer_voice(voice=FSInputFile(audio_file))
            os.remove(audio_file)

    except Exception as e:
        logging.error(f"Ошибка LLM: {e}")
        await message.answer("Извини, что-то пошло не так. Попробуй ещё раз через пару секунд.")


# ============================================
# 8. FSM STATES
# ============================================
class BotStates(StatesGroup):
    waiting_for_translation = State()
    waiting_for_quiz_answer = State()


# ============================================
# 9. КЛАВИАТУРА
# ============================================
def get_main_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="📖 Перевод слова"), KeyboardButton(text="🗣 Объясни грамматику")],
            [KeyboardButton(text="💬 Режим диалога"), KeyboardButton(text="🎲 Случайное слово")],
            [KeyboardButton(text="📚 Мои слова"), KeyboardButton(text="🔄 Повторить слова")]
        ],
        resize_keyboard=True
    )


# ============================================
# 10. ОБРАБОТЧИКИ КОМАНД И КНОПОК
# ============================================
@dp.message(CommandStart())
async def command_start_handler(message: types.Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(
        f"Привет, {message.from_user.full_name}! 👋\n"
        "Теперь я умею <b>понимать твою речь</b>! Нажми на микрофон и скажи что-нибудь по-корейски или по-русски.",
        parse_mode="HTML",
        reply_markup=get_main_keyboard()
    )


@dp.message(lambda msg: msg.text == "📖 Перевод слова")
async def start_translation(message: types.Message, state: FSMContext):
    await state.set_state(BotStates.waiting_for_translation)
    await message.answer("Напишите слово (на русском или корейском), которое хотите перевести.")


@dp.message(StateFilter(BotStates.waiting_for_translation))
async def translate_word_handler(message: types.Message, state: FSMContext) -> None:
    user_text = message.text
    await state.clear()

    prompt = (
        f"Ученик попросил перевести слово: '{user_text}'. "
        f"В ПЕРВОЙ СТРОКЕ напиши ТОЛЬКО формат: корейское_слово — перевод. "
        f"Например: 가족 — семья. Никаких транскрипций в первой строке. "
        f"Со второй строки дай развёрнутое объяснение и пример."
    )

    await process_and_reply(message, message.from_user.id, prompt, is_translation=True, word_to_save=user_text)


@dp.message(lambda msg: msg.text == "📚 Мои слова")
async def my_words_handler(message: types.Message):
    user_id = message.from_user.id
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT word, translation, next_review FROM vocabulary WHERE user_id = $1 ORDER BY id DESC LIMIT 15",
            user_id
        )
    if not rows:
        await message.answer("Твой словарь пока пуст.")
    else:
        text = "📚 <b>Твои слова:</b>\n\n"
        for r in rows:
            text += f"• <b>{r['word']}</b> — {r['translation'][:30]}... (след. повтор: {str(r['next_review'])[:10]})\n"
        await message.answer(text, parse_mode="HTML")


@dp.message(lambda msg: msg.text == "🔄 Повторить слова")
async def start_quiz(message: types.Message, state: FSMContext):
    await state.clear()
    await message.answer("Запускаю карточки...")
    await send_next_quiz(message, message.from_user.id)


async def send_next_quiz(message: types.Message, user_id: int):
    due_words = await get_due_words(user_id)
    if not due_words:
        await message.answer("🎉 На сегодня всё! Все слова повторены.", reply_markup=get_main_keyboard())
        return
    word_id, word, translation, interval = due_words[0]
    await message.answer(f"Переведи на русский:\n\n👉 <b>{word}</b>", parse_mode="HTML")
    await dp.storage.set_state(chat_id=message.chat.id, user_id=user_id, state=BotStates.waiting_for_quiz_answer)
    await dp.storage.update_data(chat_id=message.chat.id, user_id=user_id,
                                 data={'word_id': word_id, 'translation': translation, 'interval': interval})


@dp.message(StateFilter(BotStates.waiting_for_quiz_answer))
async def quiz_answer_handler(message: types.Message, state: FSMContext):
    user_answer = message.text.lower().strip()
    data = await state.get_data()

    if user_answer in data['translation'].lower() or data['translation'].lower() in user_answer:
        await message.answer("✅ <b>Правильно!</b> Молодец!")
        await update_srs(message.from_user.id, data['word_id'], data['interval'], is_correct=True)
    else:
        await message.answer(f"❌ <b>Ошибка.</b> Правильный ответ:\n{data['translation']}")
        await update_srs(message.from_user.id, data['word_id'], data['interval'], is_correct=False)

    await state.clear()
    await send_next_quiz(message, message.from_user.id)


@dp.message()
async def chat_handler(message: types.Message, state: FSMContext) -> None:
    """Обработчик обычных сообщений."""
    user_id = message.from_user.id
    user_text = message.text
    if not user_text:
        return

    if user_text == "🗣 Объясни грамматику":
        user_text = "Ученик хочет разобрать грамматику. Предложи популярную тему для начинающих и объясни ее."
    elif user_text == "💬 Режим диалога":
        user_text = "Давай устроим диалог на корейском! Начни с простой фразы на корейском, напиши перевод и жди моего ответа."
    elif user_text == "🎲 Случайное слово":
        user_text = "Дай мне одно полезное случайное корейское слово на сегодня. Напиши его на корейском, транскрипцию, перевод и пример."

    await process_and_reply(message, user_id, user_text)


# ============================================
# 11. ГОЛОСОВЫЕ СООБЩЕНИЯ
# ============================================
@dp.message(F.voice)
async def voice_message_handler(message: types.Message, state: FSMContext):
    user_id = message.from_user.id
    await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")

    file_id = message.voice.file_id
    file = await bot.get_file(file_id)
    file_path = f"voice_{uuid.uuid4().hex}.ogg"
    await bot.download_file(file.file_path, file_path)

    try:
        # Получаем последнюю фразу бота
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT content FROM history WHERE user_id = $1 AND role = 'assistant' ORDER BY id DESC LIMIT 1",
                user_id
            )
            last_bot_text = row['content'][:250] if row else ""

        logging.info("Распознавание речи через Groq Whisper API...")

        with open(file_path, "rb") as audio_file:
            transcript = await client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=audio_file,
                language="ko",
                prompt="안녕하세요. 집, 사과, 가족, 친구, 물, 사랑, 행복."
            )

        recognized_text = transcript.text.strip()

        if not recognized_text or len(recognized_text) < 1:
            await message.answer("Я не разобрал, что ты сказал. Попробуй ещё раз.")
            return

        await message.answer(f"🎤 <i>Я услышал:</i> {recognized_text}", parse_mode="HTML")

        # Логика проверки
        target_word = ""
        word_match = re.search(rf'(?:слово|повтори)[^가-힣ㄱ-ㅎㅏ-ㅣ]*?([가-힣ㄱ-ㅎㅏ-ㅣ]+)', last_bot_text)
        if word_match:
            target_word = word_match.group(1).strip()
        else:
            runs = re.findall(r'[가-힣ㄱ-ㅎㅏ-ㅣ]+', last_bot_text)
            target_word = runs[-1] if runs else ""

        clean_recognized = re.sub(r'[^\w\s]', '', recognized_text).strip()
        clean_target = re.sub(r'[^\w\s]', '', target_word).strip() if target_word else ""

        if clean_target and (clean_recognized == clean_target or clean_target in clean_recognized):
            prompt = (
                f"Твое предыдущее сообщение: '{last_bot_text}'. "
                f"Ученик сказал в микрофон: '{recognized_text}'. "
                f"ВНИМАНИЕ: Распознанный текст ИДЕАЛЬНО СОВПАДАЕТ с целевым словом! "
                f"ОБЯЗАТЕЛЬНО похвали ученика и сразу дай НОВОЕ корейское слово. "
                f"Никаких 'почти' или исправлений!"
            )
        else:
            prompt = (
                f"Твое предыдущее сообщение: '{last_bot_text}'. "
                f"Система распознавания услышала: '{recognized_text}'. "
                f"Сравни эти тексты. Если ученик правильно повторил корейское слово, похвали его и дай новое. "
                f"Если он ошибся, мягко попроси повторить правильное корейское слово. Все объяснения только на РУССКОМ."
            )

        await process_and_reply(message, user_id, prompt)

    except Exception as e:
        logging.error(f"Ошибка Whisper API: {e}")
        await message.answer("Произошла ошибка при распознавании твоего голоса.")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


# ============================================
# 12. НАПОМИНАНИЯ
# ============================================
async def send_daily_reminders():
    """Отправляет напоминания о повторении слов."""
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("SELECT DISTINCT user_id FROM vocabulary WHERE next_review <= now()")
        users = [r['user_id'] for r in rows]

    for user_id in users:
        try:
            due_words = await get_due_words(user_id)
            if due_words:
                await bot.send_message(user_id, f"🔔 <b>Пора повторить корейский!</b>\nНажми «🔄 Повторить слова».",
                                       parse_mode="HTML")
        except Exception as e:
            logging.error(f"Ошибка отправки напоминания: {e}")


# ============================================
# 13. ЗАПУСК
# ============================================
async def run_api():
    """Запускает FastAPI сервер."""
    port = int(os.environ.get("PORT", 8000))
    config = uvicorn.Config(api, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


async def main() -> None:
    """Главная функция запуска."""
    await init_db()
    scheduler.add_job(send_daily_reminders, trigger='cron', hour=19, minute=0)
    scheduler.start()
    logging.info("Бот и API запущены.")

    await asyncio.gather(dp.start_polling(bot), run_api())


if __name__ == "__main__":
    asyncio.run(main())
