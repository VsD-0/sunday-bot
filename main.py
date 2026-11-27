import asyncio
import logging
import re
import os
import uuid
import edge_tts
import aiosqlite
from datetime import datetime, timedelta
from aiogram import Bot, Dispatcher, types, F  # <-- Добавили F для фильтров
from aiogram.filters import CommandStart, StateFilter
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.types import FSInputFile, ReplyKeyboardMarkup, KeyboardButton
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.exceptions import TelegramBadRequest
from openai import AsyncOpenAI
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi.responses import FileResponse
from fastapi import HTTPException
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)

BOT_TOKEN = os.getenv("BOT_TOKEN")
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# --- FASTAPI ДЛЯ MINI APP ---
api = FastAPI()
# Разрешаем запросы с GitHub Pages и локальных тестов
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Для продакшена лучше указать конкретный URL GitHub Pages
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@api.get("/health")
async def health_check():
    return {"status": "ok", "time": datetime.now().isoformat()}

@api.get("/api/tts")
async def text_to_speech(text: str = Query(...)):
    """Генерирует аудио для корейского текста и возвращает файл."""
    import tempfile

    # Создаём временный файл
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
        tmp_path = tmp.name

    try:
        # Генерируем аудио через edge_tts
        voice = "ko-KR-SunHiNeural"
        communicate = edge_tts.Communicate(text, voice, rate="-20%")
        await communicate.save(tmp_path)

        # Возвращаем файл
        return FileResponse(
            tmp_path,
            media_type="audio/mpeg",
            filename=f"tts_{text}.mp3"
        )
    except Exception as e:
        logging.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail="Ошибка генерации аудио")
    finally:
        # Удаляем файл после отправки (через 10 секунд, чтобы успеть отправить)
        import asyncio
        async def cleanup():
            await asyncio.sleep(10)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        asyncio.create_task(cleanup())

@api.get("/api/vocabulary")
async def get_vocabulary(user_id: int = Query(...)):
    async with aiosqlite.connect("bot_database.db") as db:
        async with db.execute(
                "SELECT word, translation, next_review FROM vocabulary WHERE user_id = ? ORDER BY id DESC LIMIT 20",
                (user_id,)
        ) as cursor:
            rows = await cursor.fetchall()
    return [
        {"word": w, "translation": t, "next_review": str(nr)}
        for w, t, nr in rows
    ]

# --- ПОДКЛЮЧЕНИЕ К ЛОКАЛЬНОЙ OLLAMA ---
# client = AsyncOpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
# MODEL_NAME = "gemma3:4b"

# --- ПОДКЛЮЧЕНИЕ К GROQ API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = AsyncOpenAI(
    base_url='https://api.groq.com/openai/v1',
    api_key=GROQ_API_KEY
)
MODEL_NAME = "openai/gpt-oss-120b"

scheduler = AsyncIOScheduler()


class BotStates(StatesGroup):
    waiting_for_translation = State()
    waiting_for_quiz_answer = State()


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

HANGUL = '가-힣㄰-㆏'
HANGUL_RE = re.compile(f'[{HANGUL}]+')

def strip_tts_garbage(text: str) -> str:
    """Вырезаем и правильные теги, и мусор вида [TT5a], [ttss], [/TTS]."""
    text = re.sub(r'\[/?TTS\]', '', text, flags=re.I)
    text = re.sub(r'\[[A-Za-z0-9/]{1,12}\]', '', text)   # битые [TT5a], [TT56]...
    text = re.sub(r'\bttss?\b', '', text, flags=re.I)
    text = re.sub(r'\s*/\s*(?=\s|$)', '', text)          # хвосты вида "a/"
    return text


def extract_voice(text: str) -> str:
    """Слово для озвучки: сначала из тегов (на всякий случай), потом — сами ищем хангыль."""
    m = re.findall(r'\[TTS\](.*?)\[/TTS\]', text, flags=re.I)
    if m:
        return m[-1].strip()
    cleaned = strip_tts_garbage(text)
    m = re.search(rf'(?:слово|повтори)[^{HANGUL}]*?([{HANGUL}]+)', cleaned, re.I)
    if m:
        return m.group(1)
    runs = HANGUL_RE.findall(cleaned)
    if len(runs) > 4:      # это список букв (алфавит) — озвучивать нечего
        return ''
    return runs[-1] if runs else ''

def strip_model_tags(text: str) -> str:
    """Удаляем HTML-теги, которые модель вставила сама (копирует из истории/весов),
    оставляя текст внутри них."""
    return re.sub(r'</?[a-zA-Z][^>]*>', '', text)

def to_html(text: str) -> str:
    text = strip_model_tags(text)                       # 1) сначала вырезаем чужие <b>
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')  # 2) экранируем то, что осталось
    text = HANGUL_RE.sub(lambda m: f'<b>{m.group(0)}</b>', text)  # 3) добавляем СВОИ теги
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# --- БАЗА ДАННЫХ (без изменений) ---
async def init_db():
    async with aiosqlite.connect("bot_database.db") as db:
        await db.execute(
            "CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, role TEXT, content TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
        await db.execute(
            "CREATE TABLE IF NOT EXISTS vocabulary (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, word TEXT, translation TEXT, interval INTEGER DEFAULT 1, next_review DATETIME, UNIQUE(user_id, word))")
        await db.commit()


async def save_message(user_id: int, role: str, content: str):
    async with aiosqlite.connect("bot_database.db") as db:
        await db.execute("INSERT INTO history (user_id, role, content) VALUES (?, ?, ?)", (user_id, role, content))
        await db.commit()


async def get_history_messages(user_id: int, limit: int = 10) -> list:
    async with aiosqlite.connect("bot_database.db") as db:
        async with db.execute("SELECT role, content FROM history WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                              (user_id, limit)) as cursor:
            rows = await cursor.fetchall()
            rows.reverse()
            return [{"role": r, "content": c} for r, c in rows]


async def save_word(user_id: int, word: str, translation: str):
    next_review = datetime.now() + timedelta(days=1)
    async with aiosqlite.connect("bot_database.db") as db:
        await db.execute(
            "INSERT OR IGNORE INTO vocabulary (user_id, word, translation, interval, next_review) VALUES (?, ?, ?, ?, ?)",
            (user_id, word.strip(), translation.strip(), 1, next_review))
        await db.commit()


# --- ЛОГИКА SRS ---
async def get_due_words(user_id: int):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    async with aiosqlite.connect("bot_database.db") as db:
        async with db.execute("SELECT id, word, translation FROM vocabulary WHERE user_id = ? AND next_review <= ?",
                              (user_id, now)) as cursor:
            return await cursor.fetchall()


async def update_srs(user_id: int, word_id: int, current_interval: int, is_correct: bool):
    async with aiosqlite.connect("bot_database.db") as db:
        new_interval = current_interval * 2 if is_correct else 1
        next_review = datetime.now() + timedelta(days=new_interval)
        await db.execute("UPDATE vocabulary SET interval = ?, next_review = ? WHERE id = ? AND user_id = ?",
                         (new_interval, next_review.strftime('%Y-%m-%d %H:%M:%S'), word_id, user_id))
        await db.commit()


# --- ФУНКЦИИ БОТА ---
def get_main_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="📖 Перевод слова"), KeyboardButton(text="🗣 Объясни грамматику")],
            [KeyboardButton(text="💬 Режим диалога"), KeyboardButton(text="🎲 Случайное слово")],
            [KeyboardButton(text="📚 Мои слова"), KeyboardButton(text="🔄 Повторить слова")]
        ],
        resize_keyboard=True
    )


async def generate_voice(text: str) -> str:
    voice = "ko-KR-SunHiNeural"
    filename = f"voice_{uuid.uuid4().hex}.mp3"
    communicate = edge_tts.Communicate(text, voice, rate="-20%")
    await communicate.save(filename)
    return filename


async def process_and_reply(message: types.Message, user_id: int, prompt_text: str, is_translation: bool = False, word_to_save: str = None):
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

            # Если просили слово, а хангыля нет — перегенерируем
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
            # Сохраняем только первую строку ответа как короткий перевод
            short_translation = response_text.split('\n')[0][:100].strip().rstrip('.,;:!?-')
            await save_word(user_id, word_to_save, short_translation)

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


# --- ОБРАБОТЧИК ГОЛОСОВЫХ СООБЩЕНИЙ (НОВЫЙ) ---
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
        async with aiosqlite.connect("bot_database.db") as db:
            async with db.execute(
                "SELECT content FROM history WHERE user_id = ? AND role = 'assistant' ORDER BY id DESC LIMIT 1",
                (user_id,)) as cursor:
                row = await cursor.fetchone()
                last_bot_text = row[0][:250] if row else ""

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

        # --- ЛОГИКА ПРОВЕРКИ (без изменений) ---
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


# --- ХЕНДЛЕРЫ КНОПОК (без изменений) ---
@dp.message(CommandStart())
async def command_start_handler(message: types.Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(
        f"Привет, {message.from_user.full_name}! 👋\n"
        "Теперь я умею <b>понимать твою речь</b>! Нажми на микрофон и скажи что-нибудь по-корейски или по-русски.",
        parse_mode="HTML",
        reply_markup=get_main_keyboard()
    )


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


async def send_next_quiz(message: types.Message, user_id: int):
    due_words = await get_due_words(user_id)
    if not due_words:
        await message.answer("🎉 На сегодня всё! Все слова повторены.", reply_markup=get_main_keyboard())
        return
    word_id, word, translation = due_words[0]
    async with aiosqlite.connect("bot_database.db") as db:
        async with db.execute("SELECT interval FROM vocabulary WHERE id = ?", (word_id,)) as cursor:
            row = await cursor.fetchone()
            interval = row[0] if row else 1
    await message.answer(f"Переведи на русский:\n\n👉 <b>{word}</b>", parse_mode="HTML")
    await dp.storage.set_state(chat_id=message.chat.id, user_id=user_id, state=BotStates.waiting_for_quiz_answer)
    await dp.storage.update_data(chat_id=message.chat.id, user_id=user_id,
                                 data={'word_id': word_id, 'translation': translation, 'interval': interval})


@dp.message(lambda msg: msg.text == "🔄 Повторить слова")
async def start_quiz(message: types.Message, state: FSMContext):
    await state.clear()
    await message.answer("Запускаю карточки...")
    await send_next_quiz(message, message.from_user.id)


@dp.message(lambda msg: msg.text == "📚 Мои слова")
async def my_words_handler(message: types.Message):
    user_id = message.from_user.id
    async with aiosqlite.connect("bot_database.db") as db:
        async with db.execute(
                "SELECT word, translation, next_review FROM vocabulary WHERE user_id = ? ORDER BY id DESC LIMIT 15",
                (user_id,)) as cursor:
            words = await cursor.fetchall()
    if not words:
        await message.answer("Твой словарь пока пуст.")
    else:
        text = "📚 <b>Твои слова:</b>\n\n"
        for w, t, nr in words:
            text += f"• <b>{w}</b> — {t[:30]}... (след. повтор: {nr[:10]})\n"
        await message.answer(text, parse_mode="HTML")


@dp.message(StateFilter(BotStates.waiting_for_translation))
async def translate_word_handler(message: types.Message, state: FSMContext) -> None:
    user_text = message.text
    await state.clear()

    # Формируем промпт для модели
    prompt = (
        f"Ученик попросил перевести слово: '{user_text}'. "
        f"В ПЕРВОЙ СТРОКЕ напиши только короткий перевод (1-3 слова). "
        f"Со второй строки дай развёрнутое объяснение и пример."
    )

    # Передаём исходное слово отдельно
    await process_and_reply(
        message,
        message.from_user.id,
        prompt,
        is_translation=True,
        word_to_save=user_text  # <-- вот это важно!
    )


@dp.message()
async def chat_handler(message: types.Message, state: FSMContext) -> None:
    user_id = message.from_user.id
    user_text = message.text
    if not user_text: return

    if user_text == "📖 Перевод слова":
        await state.set_state(BotStates.waiting_for_translation)
        await message.answer("Напишите слово (на русском или корейском), которое хотите перевести.")
        return
    elif user_text == "🗣 Объясни грамматику":
        user_text = "Ученик хочет разобрать грамматику. Предложи популярную тему для начинающих и объясни ее."
    elif user_text == "💬 Режим диалога":
        user_text = "Давай устроим диалог на корейском! Начни с простой фразы на корейском, напиши перевод и жди моего ответа."
    elif user_text == "🎲 Случайное слово":
        user_text = "Дай мне одно полезное случайное корейское слово на сегодня. Напиши его на корейском, транскрипцию, перевод и пример."

    await process_and_reply(message, user_id, user_text)


# --- НАПОМИНАНИЯ ---
async def send_daily_reminders():
    async with aiosqlite.connect("bot_database.db") as db:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        async with db.execute("SELECT DISTINCT user_id FROM vocabulary WHERE next_review <= ?", (now,)) as cursor:
            users = await cursor.fetchall()
    for (user_id,) in users:
        try:
            due_words = await get_due_words(user_id)
            if due_words:
                await bot.send_message(user_id, f"🔔 <b>Пора повторить корейский!</b>\nНажми «🔄 Повторить слова».",
                                       parse_mode="HTML")
        except Exception as e:
            logging.error(f"Ошибка отправки: {e}")


async def run_api():
    port = int(os.environ.get("PORT", 8000))  # Render даёт свой порт через $PORT
    config = uvicorn.Config(api, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


async def main() -> None:
    await init_db()
    await clean_vocabulary_once()
    scheduler.add_job(send_daily_reminders, 'cron', hour=19, minute=0)
    scheduler.start()
    logging.info("Бот и API запущены.")
    await asyncio.gather(dp.start_polling(bot), run_api())

async def clean_vocabulary_once():
    async with aiosqlite.connect("bot_database.db") as db:
        async with db.execute("SELECT id, word, translation FROM vocabulary") as cursor:
            rows = await cursor.fetchall()
        for row_id, word, translation in rows:
            # Если в колонке "word" есть текст "Ученик попросил" — это ошибка
            if "Ученик попросил" in word or len(word) > 50:
                # Пытаемся извлечь слово из translation (оно там есть в начале)
                import re
                match = re.search(r'[가-힣]+', translation)
                if match:
                    korean_word = match.group(0)
                    short_trans = translation.split('\n')[0][:100].strip().rstrip('.,;:!?-')
                    await db.execute(
                        "UPDATE vocabulary SET word = ?, translation = ? WHERE id = ?",
                        (korean_word, short_trans, row_id)
                    )
        await db.commit()
    logging.info("Словарь очищен.")

if __name__ == "__main__":
    asyncio.run(main())