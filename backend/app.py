# Импорты и настройки
import os
import re
import logging
import sqlite3
from flask import Flask, jsonify, request
from flask_cors import CORS
from openai import OpenAI
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Загрузка переменных окружения из файла .env
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация Flask приложения
app = Flask(__name__)

# Безопасная конфигурация CORS - в продакшене замените на конкретные домены
CORS(app, resources={r"/api/*": {"origins": os.getenv("ALLOWED_ORIGINS", "*")}})

# Настройки базы данных
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "cache.db")

# Конфигурация OpenAI API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY не найден в переменных окружения. Используйте .env файл или установите переменную окружения.")

client = OpenAI(api_key=OPENAI_API_KEY)

# Инициализация векторизатора для преобразования текста в векторы
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Константы
DEFAULT_SIMILARITY_THRESHOLD = 0.8
MIN_HINT_RELEVANCE = 0.1
MAX_HINT_RELEVANCE = 0.7
MAX_GENERATION_ATTEMPTS = 3

# ===================== УТИЛИТЫ И ПОМОЩНИКИ =====================

# Добавьте эту функцию в app.py после init_db()

def preload_context_transitions(num_initial_facts=5, transitions_per_fact=3):
    """
    Предзагружает популярные факты и создает для них контекстуальные переходы.

    Args:
        num_initial_facts: Количество начальных фактов для генерации
        transitions_per_fact: Количество переходов для каждого факта
    """
    logger.info(
        f"Предзагрузка контекстуальных переходов: {num_initial_facts} фактов с {transitions_per_fact} переходами каждый")

    with connect_db() as conn:
        cursor = conn.cursor()

        # Проверяем наличие фактов в базе
        cursor.execute('SELECT COUNT(*) FROM facts')
        fact_count = cursor.fetchone()[0]

        # Если есть достаточно фактов, просто выходим
        if fact_count >= num_initial_facts:
            logger.info(f"В базе уже есть {fact_count} фактов, пропускаем предзагрузку")
            return

        # Количество фактов для генерации
        facts_to_generate = num_initial_facts - fact_count

        # Генерируем начальные факты
        initial_fact_ids = []
        for _ in range(facts_to_generate):
            fact, hints = generate_initial_fact()
            if fact and hints:
                fact_id, _ = save_fact_with_hints(fact, hints)
                initial_fact_ids.append(fact_id)
                logger.info(f"Создан начальный факт (ID: {fact_id}): {fact[:50]}...")

    # Получаем все факты для создания переходов
    with connect_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT id, fact FROM facts ORDER BY id DESC LIMIT ?', (num_initial_facts,))
        facts = cursor.fetchall()

    # Для каждого факта генерируем контекстуальные переходы
    for fact_id, fact_text in facts:
        # Получаем подсказки этого факта
        with connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, hint_text FROM hints WHERE fact_id = ?', (fact_id,))
            hints = cursor.fetchall()

        # Для каждой подсказки создаем контекстуальный переход
        for hint_id, hint_text in hints[:transitions_per_fact]:  # Ограничиваем количество переходов
            # Проверяем, существует ли уже переход
            if get_contextual_transition(fact_id, hint_id):
                logger.info(f"Переход для факта {fact_id} и подсказки {hint_id} уже существует, пропускаем")
                continue

            # Генерируем новый контекстуальный факт
            new_fact, new_hints = generate_contextual_fact(fact_text, hint_text)

            if new_fact and new_hints:
                # Сохраняем и создаем переход
                new_fact_id, _ = save_fact_with_hints(
                    new_fact, new_hints, fact_id, hint_id
                )
                logger.info(f"Создан контекстуальный переход: {fact_id} --[{hint_text}]--> {new_fact_id}")
            else:
                logger.warning(f"Не удалось сгенерировать контекстуальный факт для подсказки {hint_id}")

    logger.info("Предзагрузка контекстуальных переходов завершена")


 # Запускаем предзагрузку при старте

def preprocess_text(text):
    """
    Предварительная обработка текста: приведение к нижнему регистру,
    удаление знаков препинания и лишних пробелов.

    Args:
        text (str): Исходный текст

    Returns:
        str: Обработанный текст
    """
    if not text:
        return ""
    # Приводим к нижнему регистру
    text = text.lower()
    # Удаляем знаки препинания и заменяем их пробелами
    text = re.sub(r'[^\w\s]', ' ', text)
    # Удаляем лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def calculate_similarity(text1, text2):
    """
    Вычисляет семантическую близость между двумя текстами
    с использованием TF-IDF и косинусного сходства.

    Args:
        text1 (str): Первый текст
        text2 (str): Второй текст

    Returns:
        float: Значение косинусного сходства от 0 до 1
    """
    if not text1 or not text2:
        return 0.0

    # Предварительная обработка текстов
    processed_text1 = preprocess_text(text1)
    processed_text2 = preprocess_text(text2)

    # Если тексты слишком короткие, используем простое сравнение
    if len(processed_text1) < 10 or len(processed_text2) < 10:
        common_words = set(processed_text1.split()) & set(processed_text2.split())
        total_words = set(processed_text1.split()) | set(processed_text2.split())
        if not total_words:
            return 0.0
        return len(common_words) / len(total_words)

    try:
        # Преобразуем тексты в TF-IDF векторы
        tfidf_matrix = tfidf_vectorizer.fit_transform([processed_text1, processed_text2])

        # Вычисляем косинусное сходство
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity
    except Exception as e:
        logger.warning(f"Ошибка при вычислении сходства текстов: {e}")
        return 0.0


def check_semantic_duplication(new_fact, existing_facts, threshold=DEFAULT_SIMILARITY_THRESHOLD):
    """
    Проверяет, является ли новый факт семантическим дубликатом
    существующих фактов.

    Args:
        new_fact (str): Новый факт для проверки
        existing_facts (list): Список существующих фактов
        threshold (float): Порог сходства для определения дубликатов

    Returns:
        tuple: (is_duplicate, similar_fact_index, similarity)
    """
    max_similarity = 0.0
    most_similar_index = -1

    for i, fact in enumerate(existing_facts):
        similarity = calculate_similarity(new_fact, fact)
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_index = i

    is_duplicate = max_similarity >= threshold
    return is_duplicate, most_similar_index, max_similarity


def check_hint_relevance(hint, fact, min_threshold=MIN_HINT_RELEVANCE, max_threshold=MAX_HINT_RELEVANCE):
    """
    Проверяет, насколько подсказка релевантна факту.
    Подсказка должна быть связана с фактом, но не должна полностью его дублировать.

    Args:
        hint (str): Текст подсказки
        fact (str): Текст факта
        min_threshold (float): Минимальный порог релевантности
        max_threshold (float): Максимальный порог релевантности

    Returns:
        tuple: (is_relevant, similarity)
    """
    similarity = calculate_similarity(hint, fact)

    # Подсказка должна быть достаточно связана с фактом,
    # но не должна быть слишком похожа (иначе она бесполезна)
    is_relevant = min_threshold <= similarity <= max_threshold

    return is_relevant, similarity


def filter_relevant_hints(hints, fact, min_threshold=MIN_HINT_RELEVANCE, max_threshold=MAX_HINT_RELEVANCE):
    """
    Фильтрует подсказки, оставляя только релевантные факту.

    Args:
        hints (list): Список подсказок
        fact (str): Факт, с которым проверяется релевантность
        min_threshold (float): Минимальный порог релевантности
        max_threshold (float): Максимальный порог релевантности

    Returns:
        list: Список релевантных подсказок
    """
    relevant_hints = []

    for hint in hints:
        is_relevant, similarity = check_hint_relevance(
            hint, fact, min_threshold, max_threshold
        )

        if is_relevant:
            relevant_hints.append(hint)
            logger.info(f"Подсказка релевантна (сходство: {similarity:.2f}): {hint}")
        else:
            logger.info(f"Подсказка отфильтрована (сходство: {similarity:.2f}): {hint}")

    # Если все подсказки отфильтрованы, возвращаем исходный список
    if not relevant_hints and hints:
        logger.warning("Все подсказки были отфильтрованы, возвращаем исходные")
        return hints

    return relevant_hints


# ===================== ФУНКЦИИ БАЗЫ ДАННЫХ =====================

def connect_db():
    """
    Создаёт подключение к базе данных SQLite.

    Returns:
        sqlite3.Connection: Объект подключения к БД
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Позволяет обращаться к колонкам по имени
    return conn


def init_db():
    """
    Инициализирует базу данных с необходимыми таблицами.
    """
    with connect_db() as conn:
        cursor = conn.cursor()

        # Создание таблиц с учетом нового дизайна
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fact TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS hints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hint_text TEXT NOT NULL,
            fact_id INTEGER NOT NULL,
            FOREIGN KEY (fact_id) REFERENCES facts(id)
        )''')

        # Таблица для контекстных переходов
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS context_transitions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_fact_id INTEGER NOT NULL,
            hint_id INTEGER NOT NULL,
            target_fact_id INTEGER NOT NULL,
            FOREIGN KEY (source_fact_id) REFERENCES facts(id),
            FOREIGN KEY (hint_id) REFERENCES hints(id),
            FOREIGN KEY (target_fact_id) REFERENCES facts(id)
        )''')

        conn.commit()
        logger.info("База данных инициализирована с обновленной схемой")


def save_fact_with_hints(fact, hints, source_fact_id=None, source_hint_id=None):
    """
    Сохраняет факт с подсказками и контекстуальными связями в БД.

    Args:
        fact (str): Текст факта
        hints (list): Список подсказок
        source_fact_id (int, optional): ID исходного факта при контекстуальном переходе
        source_hint_id (int, optional): ID подсказки при контекстуальном переходе

    Returns:
        tuple: (fact_id, hint_ids) - ID нового факта и список ID подсказок
    """
    try:
        with connect_db() as conn:
            cursor = conn.cursor()

            # Начинаем транзакцию
            conn.execute("BEGIN TRANSACTION")

            # Сохраняем новый факт
            cursor.execute('INSERT INTO facts (fact) VALUES (?)', (fact,))
            fact_id = cursor.lastrowid

            # Сохраняем подсказки
            hint_ids = []
            for hint in hints:
                if hint.strip():
                    cursor.execute('INSERT INTO hints (hint_text, fact_id) VALUES (?, ?)', (hint, fact_id))
                    hint_ids.append(cursor.lastrowid)

            # Если это контекстуальный переход, сохраняем связь
            if source_fact_id and source_hint_id:
                cursor.execute(
                    'INSERT INTO context_transitions (source_fact_id, hint_id, target_fact_id) VALUES (?, ?, ?)',
                    (source_fact_id, source_hint_id, fact_id)
                )

            conn.commit()
            return fact_id, hint_ids
    except Exception as e:
        logger.error(f"Ошибка при сохранении факта с подсказками: {e}")
        return None, None


def get_contextual_transition(source_fact_id, hint_id):
    """
    Находит контекстуальный переход для заданного факта и подсказки.

    Args:
        source_fact_id (int): ID исходного факта
        hint_id (int): ID подсказки

    Returns:
        int or None: ID целевого факта или None, если переход не найден
    """
    try:
        with connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT target_fact_id FROM context_transitions WHERE source_fact_id = ? AND hint_id = ?',
                (source_fact_id, hint_id)
            )
            result = cursor.fetchone()
            return result['target_fact_id'] if result else None
    except Exception as e:
        logger.error(f"Ошибка при получении контекстуального перехода: {e}")
        return None


def get_fact_with_hints(fact_id):
    """
    Получает факт и связанные с ним подсказки по ID.

    Args:
        fact_id (int): ID факта

    Returns:
        tuple: (fact, hints) - текст факта и список подсказок с ID
    """
    try:
        with connect_db() as conn:
            cursor = conn.cursor()

            # Получаем факт
            cursor.execute('SELECT fact FROM facts WHERE id = ?', (fact_id,))
            fact_row = cursor.fetchone()

            if not fact_row:
                return None, None

            fact = fact_row['fact']

            # Получаем подсказки
            cursor.execute('SELECT id, hint_text FROM hints WHERE fact_id = ?', (fact_id,))
            hints_rows = cursor.fetchall()

            # Преобразуем в структуру с ID и текстом
            hints = [{"id": row['id'], "text": row['hint_text']} for row in hints_rows]

            return fact, hints
    except Exception as e:
        logger.error(f"Ошибка при получении факта с подсказками: {e}")
        return None, None


def get_all_facts(limit=50, offset=0):
    """
    Получает список всех фактов с пагинацией.

    Args:
        limit (int): Максимальное количество фактов
        offset (int): Смещение для пагинации

    Returns:
        list: Список фактов с подсказками
    """
    try:
        with connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT id, fact FROM facts ORDER BY created_at DESC LIMIT ? OFFSET ?',
                (limit, offset)
            )
            facts = cursor.fetchall()

            results = []
            for fact_row in facts:
                fact_id = fact_row['id']
                cursor.execute('SELECT id, hint_text FROM hints WHERE fact_id = ?', (fact_id,))
                hints = [{"id": row['id'], "text": row['hint_text']} for row in cursor.fetchall()]
                results.append({
                    "id": fact_id,
                    "fact": fact_row['fact'],
                    "hints": hints
                })

            return results
    except Exception as e:
        logger.error(f"Ошибка при получении списка фактов: {e}")
        return []


def get_all_existing_facts():
    """
    Получает тексты всех существующих фактов для проверки дубликатов.

    Returns:
        list: Список текстов фактов
    """
    try:
        with connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT fact FROM facts')
            return [row['fact'] for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Ошибка при получении существующих фактов: {e}")
        return []

    # ===================== API ДЛЯ ГЕНЕРАЦИИ КОНТЕНТА =====================

def generate_initial_fact():
    """
    Генерирует начальный факт и подсказки через OpenAI API.

    Returns:
        tuple: (fact, hints) или (None, None) в случае ошибки
    """
    try:
        if not OPENAI_API_KEY:
            logger.error("Отсутствует API ключ OpenAI")
            return None, None

        response = client.chat.completions.create(
            model="ft:gpt-4o-mini-2024-07-18:personal:asf-ft-model:BACEfdxZ",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert on smoking facts."
                },
                {
                    "role": "user",
                    "content": (
                        "Generate a fact about smoking and three related different hints, "
                        "each 1-3 words long, separated by new lines. The hints "
                        "should suggest different directions to explore this topic "
                        "further. (Don't use 'Fact' before the fact and 'Hint 1', "
                        "'Hint 2', 'Hint 3', '-', '•' and other symbols before hints)"
                    )
                }
            ],
            max_tokens=150,
            temperature=0.7
        )
        content = response.choices[0].message.content.strip()
        parts = [part.strip() for part in content.split("\n") if part.strip()]

        if len(parts) >= 4:  # Убедимся, что есть факт и хотя бы 3 подсказки
            fact = parts[0]
            hints = parts[1:4]  # Берем первые 3 подсказки
            return fact, hints
        
        logger.warning("API не сгенерировал достаточно контента")
        return None, None

    except Exception as e:
        logger.error(f"Ошибка при генерации факта и подсказок: {e}")
        return None, None

def generate_contextual_fact(current_fact, hint_text):
    """
    Генерирует новый контекстуальный факт на основе предыдущего факта и выбранной подсказки.

    Args:
        current_fact (str): Текущий факт
        hint_text (str): Текст подсказки

    Returns:
        tuple: (new_fact, new_hints) или (None, None) в случае ошибки
    """
    try:
        if not OPENAI_API_KEY:
            logger.error("Отсутствует API ключ OpenAI")
            return None, None

        response = client.chat.completions.create(
            model="ft:gpt-4o-mini-2024-07-18:personal:asf-ft-model:BACEfdxZ",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert on smoking facts, providing "
                              "contextually relevant information."
                },
                {
                    "role": "user",
                    "content": (
                        f"Previous fact: '{current_fact}'\n\n"
                        f"Selected hint: '{hint_text}'\n\n"
                        "Generate a new fact about smoking that follows from the "
                        "previous fact and directly addresses the selected hint. "
                        "Then generate three new different hints (each 1-3 words) that could "
                        "expand on this new fact in different directions. Format "
                        "your response as a fact followed by three different hints."
                        "(Don't use 'Fact' before the fact and "
                        "'Hint 1', 'Hint 2', 'Hint 3', '- ', '•' and other symbols  before hints"
                        "Don't generate the same hints.)"
                    )
                }
            ],
            max_tokens=200,
            temperature=0.7
        )

        content = response.choices[0].message.content.strip()
        parts = [part.strip() for part in content.split("\n") if part.strip()]

        if len(parts) >= 4:  # Убедимся, что есть факт и хотя бы 3 подсказки
            new_fact = parts[0]
            new_hints = parts[1:4]  # Берем первые 3 подсказки
            return new_fact, new_hints

        logger.warning("API не сгенерировал достаточно контекстуального контента")
        return None, None

    except Exception as e:
        logger.error(f"Ошибка при генерации контекстуального факта: {e}")
        return None, None

def generate_contextual_fact_with_deduplication(current_fact, hint_text, existing_facts,
                                                similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
                                                max_attempts=MAX_GENERATION_ATTEMPTS):
    """
    Генерирует новый контекстуальный факт, проверяя его на дублирование
    с существующими фактами.

    Args:
        current_fact (str): Текущий факт
        hint_text (str): Текст подсказки
        existing_facts (list): Список существующих фактов для проверки дубликатов
        similarity_threshold (float): Порог для определения дубликатов
        max_attempts (int): Максимальное количество попыток генерации

    Returns:
        tuple: (new_fact, new_hints, is_duplicate, similarity)
    """
    for attempt in range(max_attempts):
        new_fact, new_hints = generate_contextual_fact(current_fact, hint_text)

        if not new_fact or not new_hints:
            continue

        # Проверяем на дублирование
        is_duplicate, similar_fact_index, similarity = check_semantic_duplication(
            new_fact, existing_facts, similarity_threshold
        )

        # Если это не дубликат или последняя попытка, возвращаем результат
        if not is_duplicate or attempt == max_attempts - 1:
            logger.info(f"Сгенерирован факт (попытка {attempt + 1}/{max_attempts}, "
                        f"сходство: {similarity:.2f}): {new_fact[:50]}...")
            return new_fact, new_hints, is_duplicate, similarity

        logger.info(f"Обнаружен семантический дубликат (сходство: {similarity:.2f}), "
                    f"повторная попытка {attempt + 1}/{max_attempts}")

    # Если все попытки не удались, возвращаем последний результат
    return new_fact, new_hints, is_duplicate, similarity

    # ===================== ИНИЦИАЛИЗАЦИЯ И ПРЕДЗАГРУЗКА ДАННЫХ =====================

def preload_context_transitions(num_initial_facts=5, transitions_per_fact=3):
    """
    Предзагружает популярные факты и создает для них контекстуальные переходы.

    Args:
        num_initial_facts: Количество начальных фактов для генерации
        transitions_per_fact: Количество переходов для каждого факта
    """
    logger.info(
        f"Предзагрузка контекстуальных переходов: {num_initial_facts} фактов с {transitions_per_fact} переходами каждый")

    # Проверяем наличие фактов в базе
    with connect_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) AS count FROM facts')
        fact_count = cursor.fetchone()['count']

        # Если есть достаточно фактов, просто выходим
        if fact_count >= num_initial_facts:
            logger.info(f"В базе уже есть {fact_count} фактов, пропускаем предзагрузку")
            return

    # Количество фактов для генерации
    facts_to_generate = num_initial_facts - fact_count

    # Генерируем начальные факты
    initial_fact_ids = []
    for _ in range(facts_to_generate):
        fact, hints = generate_initial_fact()
        if fact and hints:
            result = save_fact_with_hints(fact, hints)
            if result:
                fact_id, _ = result
                initial_fact_ids.append(fact_id)
                logger.info(f"Создан начальный факт (ID: {fact_id}): {fact[:50]}...")

    # Получаем все факты для создания переходов
    with connect_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT id, fact FROM facts ORDER BY id DESC LIMIT ?', (num_initial_facts,))
        facts = cursor.fetchall()

    # Для каждого факта генерируем контекстуальные переходы
    for fact_row in facts:
        fact_id = fact_row['id']
        fact_text = fact_row['fact']

        # Получаем подсказки этого факта
        with connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, hint_text FROM hints WHERE fact_id = ?', (fact_id,))
            hints = cursor.fetchall()

        # Для каждой подсказки создаем контекстуальный переход
        for hint_row in hints[:transitions_per_fact]:  # Ограничиваем количество переходов
            hint_id = hint_row['id']
            hint_text = hint_row['hint_text']

            # Проверяем, существует ли уже переход
            if get_contextual_transition(fact_id, hint_id):
                logger.info(f"Переход для факта {fact_id} и подсказки {hint_id} уже существует, пропускаем")
                continue

            # Генерируем новый контекстуальный факт
            new_fact, new_hints = generate_contextual_fact(fact_text, hint_text)

            if new_fact and new_hints:
                # Фильтруем подсказки для улучшения релевантности
                filtered_hints = filter_relevant_hints(new_hints, new_fact)

                # Если после фильтрации осталось менее 3 подсказок, дополняем их
                if len(filtered_hints) < 3 and len(filtered_hints) > 0:
                    logger.info(f"После фильтрации осталось {len(filtered_hints)} подсказок, дополняем")
                    while len(filtered_hints) < 3:
                        filtered_hints.append(filtered_hints[0])

                # Сохраняем и создаем переход
                result = save_fact_with_hints(
                    new_fact, filtered_hints, fact_id, hint_id
                )
                if result:
                    new_fact_id, _ = result
                    logger.info(f"Создан контекстуальный переход: {fact_id} --[{hint_text}]--> {new_fact_id}")
            else:
                logger.warning(f"Не удалось сгенерировать контекстуальный факт для подсказки {hint_id}")

    logger.info("Предзагрузка контекстуальных переходов завершена")

        # ===================== МАРШРУТЫ API =====================

@app.route('/api/all_facts', methods=['GET'])
def all_facts():
    """
    API для получения всех фактов с пагинацией.
    """
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)

        # Ограничиваем размер страницы для предотвращения DoS
        per_page = min(per_page, 50)
        offset = (page - 1) * per_page

        facts = get_all_facts(limit=per_page, offset=offset)
        return jsonify({"facts": facts, "page": page, "per_page": per_page})
    except Exception as e:
        logger.error(f"Ошибка при получении всех фактов: {e}")
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500

@app.route('/api/fact', methods=['GET'])
def get_fact():
    """
    API для генерации нового начального факта.
    """
    try:
        # Генерируем новый начальный факт
        fact, hints = generate_initial_fact()

        if fact and hints:
            # Сохраняем в БД
            result = save_fact_with_hints(fact, hints)

            if not result:
                return jsonify({"error": "Ошибка при сохранении факта в базу данных"}), 500

            fact_id, hint_ids = result

            # Преобразуем в формат с ID для фронтенда
            hints_with_ids = []
            for i, hint_id in enumerate(hint_ids):
                if i < len(hints):
                    hints_with_ids.append({"id": hint_id, "text": hints[i]})

            return jsonify({"id": fact_id, "fact": fact, "hints": hints_with_ids}), 200
        else:
            return jsonify({"error": "Не удалось сгенерировать факт и подсказки"}), 500
    except Exception as e:
        logger.error(f"Ошибка в маршруте /api/fact: {e}")
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500

@app.route('/api/contextual_hint', methods=['POST'])
def generate_contextual_hint():
    """
    API для генерации контекстуального перехода по подсказке.
    """
    try:
        # Валидация входных данных
        data = request.get_json()
        if not data:
            return jsonify({"error": "Отсутствуют данные JSON"}), 400

        source_fact_id = data.get("factId")
        hint_id = data.get("hintId")
        current_fact = data.get("currentFact")
        hint_text = data.get("hintText")

        # Проверка обязательных полей
        if not all([source_fact_id, hint_id, current_fact, hint_text]):
            missing_fields = []
            if not source_fact_id: missing_fields.append("factId")
            if not hint_id: missing_fields.append("hintId")
            if not current_fact: missing_fields.append("currentFact")
            if not hint_text: missing_fields.append("hintText")

            error_msg = f"Отсутствуют обязательные параметры: {', '.join(missing_fields)}"
            logger.warning(error_msg)
            return jsonify({"error": error_msg}), 400

        # Проверяем, есть ли уже сохраненный контекстуальный переход
        target_fact_id = get_contextual_transition(source_fact_id, hint_id)

        if target_fact_id:
            # Используем существующий переход
            fact, hints_with_ids = get_fact_with_hints(target_fact_id)

            if fact and hints_with_ids:
                logger.info(f"Используем существующий переход для факта {source_fact_id} и подсказки {hint_id}")
                return jsonify({
                    "id": target_fact_id,
                    "fact": fact,
                    "hints": hints_with_ids,
                    "is_cached": True
                }), 200
            else:
                logger.warning(f"Найден переход, но не удалось получить факт {target_fact_id}")

        # Если перехода нет, получаем все существующие факты для проверки дубликатов
        existing_facts = get_all_existing_facts()

        # Генерируем новый контекстуальный факт с проверкой дубликатов
        new_fact, new_hints, is_duplicate, similarity = generate_contextual_fact_with_deduplication(
            current_fact, hint_text, existing_facts
        )

        if not new_fact or not new_hints:
            return jsonify({"error": "Не удалось сгенерировать контекстуальный факт"}), 500

        # Фильтруем подсказки для улучшения релевантности
        filtered_hints = filter_relevant_hints(new_hints, new_fact)

        # Если после фильтрации осталось менее 3 подсказок, дополняем их
        if len(filtered_hints) < 3 and len(filtered_hints) > 0:
            logger.info(f"После фильтрации осталось {len(filtered_hints)} подсказок, дополняем")
            while len(filtered_hints) < 3:
                filtered_hints.append(filtered_hints[0])  # Дублируем первую подсказку

        # Сохраняем новый факт и создаем контекстуальную связь
        result = save_fact_with_hints(
            new_fact, filtered_hints, source_fact_id, hint_id
        )

        if not result:
            return jsonify({"error": "Ошибка при сохранении факта в базу данных"}), 500

        new_fact_id, new_hint_ids = result

        # Преобразуем в формат с ID для фронтенда
        hints_with_ids = []
        for i, hint_id in enumerate(new_hint_ids):
            if i < len(filtered_hints):
                hints_with_ids.append({"id": hint_id, "text": filtered_hints[i]})

        logger.info(f"Создан новый контекстуальный факт (ID: {new_fact_id}) для подсказки '{hint_text}'")

        # Возвращаем результат с информацией о дубликате и сходстве
        return jsonify({
            "id": new_fact_id,
            "fact": new_fact,
            "hints": hints_with_ids,
            "is_duplicate": bool(is_duplicate),
            "similarity": float(similarity),  # Преобразуем в float для корректной сериализации JSON
            "is_cached": False
        }), 200

    except Exception as e:
        logger.error(f"Ошибка в маршруте /api/contextual_hint: {e}")
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500


# ===================== ТОЧКА ВХОДА ПРИЛОЖЕНИЯ =====================

if __name__ == '__main__':
    init_db()
    preload_context_transitions()
    
    print("Зарегистрированные маршруты:")
    for rule in app.url_map.iter_rules():
        print(f"{rule.endpoint}: {rule.methods} - {rule}")

    # Предзагрузка данных (только в режиме разработки)
    # if os.getenv('FLASK_ENV') == 'development':
    #    preload_context_transitions()

    # Запуск сервера - debug=True только в режиме разработки
    debug_mode = os.getenv('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=debug_mode)