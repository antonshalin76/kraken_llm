# Kraken LLM Framework

Универсальный Python фреймворк для работы с большими языковыми моделями (LLM) с полной поддержкой OpenAI API.

## Обзор

Kraken LLM Framework предоставляет единый интерфейс для различных типов взаимодействия с LLM, включая стандартные запросы, потоковую передачу, структурированный вывод, мультимодальность и работу с речью.

### Ключевые особенности

- **Универсальный клиент**: UniversalLLMClient объединяет все возможности в едином интерфейсе
- **Полная поддержка OpenAI API**: chat completions, streaming, function calling, tool calling
- **Структурированный вывод**: Валидация Pydantic моделей с интеграцией Outlines и нативной поддержкой OpenAI
- **Асинхронность**: Построен на AsyncOpenAI для высокой производительности
- **Типобезопасность**: Полная поддержка type hints и IDE
- **Простая конфигурация**: Pydantic Settings с поддержкой переменных окружения
- **Расширяемость**: Архитектура плагинов для пользовательских функций и инструментов
- **Мультимодальность**: Поддержка текста, изображений, аудио и видео
- **Речевые технологии**: ASR (распознавание речи), TTS (синтез речи), диаризация спикеров
- **Рассуждающие модели**: Поддержка Chain of Thought и нативных thinking токенов
- **Адаптивность**: Автоматический выбор оптимального режима работы
- **Режим уверенной генерации (confidence-aware)**: интеграция logprobs, метрики уверенности для ответа и токенов, множественная перегенерация с фильтрацией по порогам
- **Цветовая визуализация уверенности**: ANSI/HTML-колоризация текста и токенов, легенды, интерактивный стриминговый чат‑бот с градиентом уверенности
- **Анализ возможностей**: Автоматическое определение возможностей моделей

## Установка

### Базовая установка

```bash
# Пакетом из PyPI
pip install kraken-llm

# Из исходников
git clone https://github.com/antonshalin76/kraken_llm
cd kraken-llm
pip install -e .

# С дополнительными зависимостями
pip install -e .[dev]  # Для разработки
pip install -e .[all]  # Все зависимости
```

### Системные требования

- Python 3.10+
- AsyncOpenAI 1.0.0+
- Pydantic 2.0.0+
- Outlines 0.0.30+
- Pillow 10.0.0+ (для работы с изображениями)

## Быстрый старт

### Простейший пример

```python
from kraken_llm import create_universal_client

async with create_universal_client() as client:
    response = await client.chat_completion([
        {"role": "user", "content": "Привет, мир!"}
    ])
    print(response)
```

### Анализ возможностей моделей

Перед началом работы рекомендуется проанализировать возможности ваших моделей:

```bash
# Быстрый анализ
python3 model_capabilities_analyzer.py --quick

# Полный анализ с Markdown отчетом
python3 model_capabilities_analyzer.py --output markdown

# Через Makefile
make capabilities-analyze-quick
```

## Конфигурация

### Переменные окружения

Все параметры конфигурации могут быть заданы через переменные окружения с префиксом `LLM_`:

```bash
export LLM_ENDPOINT="http://localhost:8080"
export LLM_API_KEY="your-api-key"
export LLM_MODEL="chat"
export LLM_TEMPERATURE=0.7
export LLM_MAX_TOKENS=2000
```

### Файл .env

```env
LLM_ENDPOINT=http://localhost:8080
LLM_API_KEY=your-api-key
LLM_MODEL=chat
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2000
LLM_STREAM=false
LLM_OUTLINES_SO_MODE=true
```

### Класс LLMConfig

```python
from kraken_llm import LLMConfig

config = LLMConfig(
    endpoint="http://localhost:8080",
    api_key="your-api-key",
    model="chat",
    temperature=0.7,
    max_tokens=2000
)
```

## Универсальный клиент

### Основные возможности

UniversalLLMClient - это универсальный клиент, который объединяет все возможности Kraken LLM в едином интерфейсе:

```python
from kraken_llm import (
    create_universal_client,
    create_basic_client,
    create_advanced_client,
    create_full_client,
    UniversalCapability
)

# Базовый клиент (chat + streaming)
async with create_basic_client() as client:
    response = await client.chat_completion([
        {"role": "user", "content": "Привет!"}
    ])

# Продвинутый клиент (+ structured output, function calling, reasoning)
async with create_advanced_client() as client:
    # Автоматический fallback для structured output
    from pydantic import BaseModel
    
    class Task(BaseModel):
        title: str
        priority: int
    
    task = await client.chat_completion_structured([
        {"role": "user", "content": "Создай задачу изучить Python"}
    ], response_model=Task)

# Полнофункциональный клиент (все возможности)
async with create_full_client() as client:
    capabilities = client.get_available_capabilities()
    print(f"Доступные возможности: {capabilities}")
```

### Создание на основе анализа возможностей

```python
from kraken_llm import create_universal_client_from_report

# Анализируем возможности модели
from model_capabilities_analyzer import ModelCapabilitiesAnalyzer

analyzer = ModelCapabilitiesAnalyzer()
report = await analyzer.analyze_all_models()

# Создаем оптимальный клиент
async with create_universal_client_from_report(report) as client:
    # Клиент автоматически настроен под возможности модели
    response = await client.chat_completion([
        {"role": "user", "content": "Тест"}
    ])
```

### Кастомная конфигурация

```python
from kraken_llm import create_universal_client, UniversalCapability

# Выбираем только нужные возможности
capabilities = {
    UniversalCapability.CHAT_COMPLETION,
    UniversalCapability.STREAMING,
    UniversalCapability.STRUCTURED_OUTPUT,
    UniversalCapability.FUNCTION_CALLING
}

async with create_universal_client(capabilities=capabilities) as client:
    # Используем только выбранные возможности
    pass
```

## Типы клиентов

### Специализированные клиенты

Kraken предоставляет специализированные клиенты для различных задач:

```python
from kraken_llm import (
    create_standard_client,      # Базовые операции
    create_streaming_client,     # Потоковая передача
    create_structured_client,    # Структурированный вывод
    create_reasoning_client,     # Рассуждающие модели
    create_multimodal_client,    # Мультимодальность
    create_adaptive_client,      # Адаптивный режим
    create_asr_client,          # Речевые технологии
    create_embeddings_client,   # Векторные представления
)

# Стандартный клиент
async with create_standard_client() as client:
    response = await client.chat_completion([
        {"role": "user", "content": "Привет"}
    ])

# Потоковый клиент
async with create_streaming_client() as client:
    async for chunk in client.chat_completion_stream([
        {"role": "user", "content": "Расскажи историю"}
    ]):
        print(chunk, end="", flush=True)
```

### Фабрика клиентов

```python
from kraken_llm import ClientFactory, create_client

# Автоматический выбор типа клиента
client = create_client(
    stream=True  # Автоматически выберет StreamingLLMClient
)

# Явное указание типа
client = ClientFactory.create_client(
    client_type="structured",
    endpoint="http://localhost:8080"
)
```

## Структурированный вывод

### Автоматический fallback

UniversalLLMClient автоматически выбирает оптимальный режим для structured output:

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    skills: list[str]

async with create_universal_client() as client:
    # Автоматически пробует:
    # 1. Нативный OpenAI structured output
    # 2. Outlines режим
    # 3. Fallback через JSON parsing
    person = await client.chat_completion_structured([
        {"role": "user", "content": "Создай профиль разработчика"}
    ], response_model=Person)
```

### Режимы работы

```python
async with create_structured_client() as client:
    # Принудительное использование Outlines
    person = await client.chat_completion_structured(
        messages=[{"role": "user", "content": "Создай профиль"}],
        response_model=Person,
        mode="outlines"
    )
    
    # Использование нативного режима OpenAI
    person = await client.chat_completion_structured(
        messages=[{"role": "user", "content": "Создай профиль"}],
        response_model=Person,
        mode="native"
    )
```

## Рассуждающие модели

### Chain of Thought

```python
from kraken_llm import create_reasoning_client, ReasoningConfig

config = ReasoningConfig(
    model_type="prompt_based",
    enable_cot=True,
    max_reasoning_steps=10
)

async with create_reasoning_client(reasoning_config=config) as client:
    response = await client.reasoning_completion([
        {"role": "user", "content": "Реши: 15 * 23 + 45"}
    ], problem_type="math")
    
    # Доступ к шагам рассуждения
    for step in response.steps:
        print(f"Шаг {step.step_number}: {step.thought}")
```

### Native Thinking

```python
config = ReasoningConfig(
    model_type="native_thinking",
    enable_thinking=True,
    thinking_max_tokens=5000
)

async with create_reasoning_client(reasoning_config=config) as client:
    response = await client.reasoning_completion([
        {"role": "user", "content": "Объясни квантовую физику"}
    ])
    
    # Доступ к thinking блокам
    if response.thinking_blocks:
        for block in response.thinking_blocks:
            print(f"Thinking: {block.content}")
```

## Мультимодальность

### Анализ изображений

```python
from kraken_llm import create_multimodal_client
from pathlib import Path

async with create_multimodal_client() as client:
    # Анализ изображения
    response = await client.vision_completion(
        text_prompt="Опиши что видишь на изображении",
        images="photo.jpg"
    )
    
    # Анализ нескольких изображений
    response = await client.vision_completion(
        text_prompt="Сравни эти изображения",
        images=["photo1.jpg", "photo2.jpg"]
    )
```

### Работа с аудио и видео

```python
# Обработка аудио
response = await client.audio_completion(
    text_prompt="Проанализируй содержание аудио",
    audio_files="recording.wav",
    task_type="analysis"
)

# Анализ видео
response = await client.video_completion(
    text_prompt="Опиши что происходит в видео",
    video_files="video.mp4"
)
```

## Речевые технологии

### ASR Client

```python
from kraken_llm import create_asr_client

async with create_asr_client() as client:
    # Распознавание речи
    result = await client.speech_to_text(
        audio_file="recording.wav",
        language="ru"
    )
    
    # Синтез речи
    audio_data = await client.text_to_speech(
        text="Привет, как дела?",
        voice="alloy"
    )
    
    # Диаризация спикеров
    diarization = await client.speaker_diarization(
        audio_file="meeting.wav",
        num_speakers=3
    )
```

## Function и Tool Calling

### Регистрация функций

```python
def get_weather(city: str) -> str:
    """Получить погоду в указанном городе."""
    return f"В городе {city} сейчас солнечно, +20°C"

async with create_universal_client() as client:
    # Регистрация функции
    client.register_function(
        name="get_weather",
        function=get_weather,
        description="Получить текущую погоду"
    )
    
    # Использование
    response = await client.chat_completion([
        {"role": "user", "content": "Какая погода в Москве?"}
    ])
```

### Декораторы для функций

```python
from kraken_llm.tools import register_function

@register_function("calculate", "Выполнить математические вычисления")
async def calculate(expression: str) -> float:
    """Безопасное вычисление математических выражений."""
    return eval(expression)  # В реальности используйте безопасный парсер
```

## Векторные представления

### Embeddings Client

```python
from kraken_llm import create_embeddings_client

async with create_embeddings_client() as client:
    # Получение embeddings
    embeddings = await client.create_embeddings([
        "Первый текст для векторизации",
        "Второй текст для векторизации"
    ])
    
    # Поиск похожих текстов
    similar = await client.similarity_search(
        query_text="поисковый запрос",
        candidate_texts=["текст 1", "текст 2", "текст 3"],
        top_k=2
    )
```

## Потоковые операции

### Цветовая визуализация уверенности (ANSI/HTML)

Начиная с версии 0.1.4, Kraken предоставляет утилиты для цветовой визуализации уверенности ответа и отдельных токенов как в терминале (ANSI), так и для HTML.

- colorize_text_ansi(text, confidence: float) — окрасить произвольный текст по агрегированной уверенности 0..1.
- colorize_tokens_ansi(token_confidences: list[dict]) — окрасить последовательность токенов, где каждый элемент содержит {token, confidence}.
- get_confidence_legend_ansi() — вернуть легенду (градиент) для терминала.
- Аналоги для HTML: colorize_text_html, colorize_tokens_html, get_confidence_legend_html.

Пример ANSI-колоризации по токенам:

```python path=null start=null
from kraken_llm.utils.color import colorize_tokens_ansi, get_confidence_legend_ansi

# token_confidences можно получить из client.chat_completion(... include_confidence=True)
# или из ensure_confident_chat при prefer_streaming=True
print(get_confidence_legend_ansi())
print(colorize_tokens_ansi([
    {"token": "Привет", "confidence": 0.92},
    {"token": ", ", "confidence": 0.75},
    {"token": "мир", "confidence": 0.48},
    {"token": "!", "confidence": 0.33},
]))
```

Пример агрегированной колоризации текста:

```python path=null start=null
from kraken_llm.utils.color import colorize_text_ansi

text = "Это пример ответа"
confidence = 0.81
print(colorize_text_ansi(text, confidence))
```

### Интерактивный стриминговый чат‑бот с градиентом уверенности

В examples добавлен интерактивный бот с двумя режимами стриминга, показывающий уверенность токенов цветом:

- REALTIME — токены выводятся по мере генерации, каждый окрашен по своей уверенности
- AGGREGATED — бот собирает весь ответ и выводит итоговую статистику и окрашенный текст

Запуск:

```bash path=null start=null
python3 examples/chatbot_streaming_colors.py \
  --mode realtime \
  --min-confidence 0.8 \
  --per-token-threshold 0.4 \
  --max-low-conf-fraction 0.34
```

Необходимые переменные окружения:

```env path=null start=null
LLM_ENDPOINT=http://localhost:8080
LLM_API_KEY=... # или LLM_TOKEN
LLM_MODEL=chat
```

Поддержка токенных метрик в стриме зависит от провайдера (logprobs). Для стабильной работы рекомендуется:

```env path=null start=null
LLM_FORCE_OPENAI_STREAMING=true
LLM_SUPPRESS_STREAM_WARNINGS=true
LLM_LOGPROBS=true
LLM_TOP_LOGPROBS=5
```

Команды внутри бота: 'exit'/'quit' — выход; 'clear' — очистка истории; 'mode' — смена режима.

### Streaming Handler

```python
from kraken_llm.streaming import StreamHandler, StreamAggregator

# Обработка потока
handler = StreamHandler()
aggregator = StreamAggregator()

async for chunk_data in handler.process_stream(response_stream):
    if chunk_data["type"] == "content":
        aggregator.add_content(chunk_data["data"])
    elif chunk_data["type"] == "function_call_complete":
        print(f"Function call: {chunk_data['data']}")

# Получение полного контента
full_content = aggregator.get_aggregated_content()
```

## Уверенность (logprobs), множественная генерация и фильтрация

Kraken умеет работать с вероятностями токенов (logprobs) и предоставляет удобные утилиты для:
- получения интегральной «уверенности» ответа целиком,
- сбора пер‑токенных метрик уверенности при потоковой генерации,
- многократной перегенерации с фильтрацией по порогам уверенности ответа и отдельного токена.

### Быстрая интеграция уверенности для обычных запросов

Добавьте include_confidence=True, чтобы получить словарь с текстом и метриками уверенности:

```python path=null start=null
from kraken_llm import LLMConfig, create_standard_client

config = LLMConfig(endpoint=..., api_key=..., model=...)
async with create_standard_client(config) as client:
    result = await client.chat_completion(
        messages=[{"role": "user", "content": "Кратко объясни ИИ"}],
        include_confidence=True,
        max_tokens=256,
    )
    print(result["text"], result["confidence"], result["confidence_label"])  # 0..1 и человеко‑читаемая метка
```

### Потоковая генерация с пер‑токенными метриками

Для моделей, отдающих logprobs в стриме (например, vLLM), можно собирать уверенность по каждому токену:

```python path=null start=null
from kraken_llm import LLMConfig, create_streaming_client

config = LLMConfig(endpoint=..., api_key=..., model=...)
async with create_streaming_client(config) as client:
    result = await client.chat_completion(
        messages=[{"role": "user", "content": "Объясни ML простыми словами"}],
        include_confidence=True,  # включает logprobs и агрегацию
        max_tokens=256,
    )
    # result содержит token_confidences: [{token, confidence, confidence_label, ...}], total_tokens
```

Совет: для более стабильной потоковой работы с vLLM используйте нативный стрим SDK и, при необходимости, подавляйте предупреждения об обрывах:

```env
LLM_FORCE_OPENAI_STREAMING=true
LLM_SUPPRESS_STREAM_WARNINGS=true
```

### Множественная генерация с фильтрацией по уверенности

Модуль фильтрации выполняет несколько попыток генерации и возвращает первую, удовлетворяющую порогам уверенности. Если ни одна попытка не прошла порог, возвращается лучшая по уверенности.

```python path=null start=null
from kraken_llm import LLMConfig, create_standard_client
from kraken_llm.confidence.filter import ConfidenceFilterConfig, ensure_confident_chat

config = LLMConfig(endpoint=..., api_key=..., model=..., force_openai_streaming=True)
async with create_standard_client(config) as client:
    messages = [{"role": "user", "content": "Дай краткое объяснение ИИ"}]

    cfg = ConfidenceFilterConfig(
        min_confidence=0.8,     # порог по средней уверенности ответа
        max_attempts=3,         # максимум перегенераций
        prefer_streaming=True,  # собрать пер‑токенные метрики в стриме, если поддерживается
        per_token_threshold=0.4,        # порог токенной уверенности
        max_low_conf_fraction=0.34,     # допустимая доля токенов ниже порога
        # max_low_conf_count=None,      # или абсолютное число
    )

    result = await ensure_confident_chat(
        client,
        messages=messages,
        cfg=cfg,
        max_tokens=400,
    )

    # Структура результата:
    # {
    #   "text": str,
    #   "confidence": float,             # усредненная уверенность (0..1)
    #   "confidence_label": str,
    #   "attempts_made": int,
    #   "success": bool,                 # True, если достигнут min_confidence и (опц.) токенные ограничения
    #   "all_attempts": [ {attempt, temperature, text, confidence, confidence_label}, ... ],
    #   "token_confidences": [...],      # при prefer_streaming и поддержке сервера
    #   "total_tokens": int
    # }

    # Если success=False, ensure_confident_chat вернет лучшую по уверенности попытку из всех.
```

Пояснения к параметрам фильтрации:
- min_confidence — целевой порог уверенности ответа целиком (по среднему значению per‑token вероятностей при наличии logprobs или по агрегированной метрике из non‑stream ответа)
- prefer_streaming — если True, попытка выполняется так, чтобы собрать пер‑токенные метрики; при поддержке сервером позволяет фильтровать по профилю токенов
- per_token_threshold — токены с confidence ниже порога считаются «низкоуверенными»
- max_low_conf_fraction / max_low_conf_count — ограничения на долю/число низкоуверенных токенов (используются только при prefer_streaming)

Практические рекомендации:
- Для стабильного стрима повысьте таймауты: LLM_READ_TIMEOUT=300, LLM_WRITE_TIMEOUT=300, LLM_CONNECT_TIMEOUT=10
- Для vLLM часто полезно FORCE_OPENAI_STREAMING=true
- Если видите предупреждения об «incomplete chunked read», можно включить SUPPRESS_STREAM_WARNINGS=true — библиотека всё равно вернёт накопленный контент или выполнит безопасный non‑stream fallback

### Область применения
- Клиентские ассистенты и чат‑боты, где качество и надежность важнее скорости: повторная генерация повышает шанс получить стабильный ответ
- Контент‑модерация/безопасность: отбрасывание ответов с низкой уверенности, усиление фильтров на уровне токенов
- Бизнес‑логика с «барьером качества»: генерация до достижения заданного порога уверенности
- UI со стримингом: показ промежуточного текста и накопление пер‑токенных метрик для мониторинга качества в реальном времени
- Автономные пайплайны: повторная генерация и выбор лучшего ответа для последующих шагов (RAG, верификация, пост‑обработка)

### Возможности режима уверенной генерации
- Агрегированная уверенность ответа (0..1) и человеко‑читаемые метки уровня уверенности
- Пер‑токенные метрики уверенности в потоке (при поддержке провайдером logprobs в streaming)
- Множественная перегенерация с термостатом: управление температурой по попыткам (start, step, max)
- Фильтрация по порогам: ответ целиком (min_confidence) + профиль токенов (per_token_threshold, max_low_conf_fraction/max_low_conf_count)
- Возврат лучшей попытки, если порог не достигнут за max_attempts
- Прозрачные фоллбеки при обрыве стрима: накопленный контент или безопасный non‑stream запрос с метриками
- Глобальные флаги управления стримом: FORCE_OPENAI_STREAMING, SUPPRESS_STREAM_WARNINGS

### Ограничения и важные замечания
- Поддержка logprobs зависит от провайдера; некоторые отдают logprobs только в стриме (или вовсе не поддерживают)
- Значения logprobs/уверенности не являются «истинной вероятностью» правильности ответа; это полезная, но эвристическая метрика
- Перегенерация увеличивает задержку и стоимость — используйте разумные значения max_attempts и max_tokens
- Потоковая передача чувствительна к таймаутам/прокси (SSE/CDN); рекомендуются повышенные таймауты и нативный стрим SDK
- Для function/tool calls метрики confidence применяются только к текстовой части; структурированный вывод не всегда имеет токенные logprobs
- Различия в токенизации и форматах провайдеров могут влиять на сравнимость метрик между моделями

### Рекомендации по конфигурации
- Умерьте длину ответов (max_tokens 256–512) и используйте стоп‑последовательности, если это применимо
- Для стабильности включите нативный стрим и подавление предупреждений:
  - LLM_FORCE_OPENAI_STREAMING=true
  - LLM_SUPPRESS_STREAM_WARNINGS=true
- Для стрима увеличьте таймауты: LLM_READ_TIMEOUT=300, LLM_WRITE_TIMEOUT=300, LLM_CONNECT_TIMEOUT=10

## Анализ возможностей моделей

### ModelCapabilitiesAnalyzer

```python
from model_capabilities_analyzer import ModelCapabilitiesAnalyzer

# Создание анализатора
analyzer = ModelCapabilitiesAnalyzer()

# Быстрый анализ
report = await analyzer.analyze_all_models(quick_mode=True)

# Полный анализ
report = await analyzer.analyze_all_models(quick_mode=False)

# Сохранение отчета
analyzer.save_report(report, output_format="markdown", filename="capabilities.md")
analyzer.save_report(report, output_format="json", filename="capabilities.json")
```

### Использование результатов анализа

```python
# Создание клиента на основе анализа
async with create_universal_client_from_report(report, model_name="my_model") as client:
    # Клиент настроен под возможности конкретной модели
    capabilities = client.get_available_capabilities()
    print(f"Подтвержденные возможности: {capabilities}")
```

## Уверенность ответов и LogProbs

Kraken LLM поддерживает запрос и обработку logprobs, а также вычисление метрик уверенности.

- Включите logprobs глобально через конфигурацию (переменные окружения или LLMConfig)
- Либо укажите их при вызове методов клиентов
- Для удобства добавлен модуль kraken_llm.confidence.metrics с утилитами расчёта

### Включение через переменные окружения

```bash
export LLM_LOGPROBS=true
export LLM_TOP_LOGPROBS=5   # 1..5 для chat completions
```

### Пример: получение метрик уверенности (non-stream)

```python path=null start=null
from kraken_llm import create_standard_client, LLMConfig

config = LLMConfig(logprobs=True, top_logprobs=5)
async with create_standard_client(config) as client:
    result = await client.chat_completion(
        messages=[{"role": "user", "content": "Столица Франции?"}],
        include_confidence=True
    )
    # result — словарь:
    # {
    #   "text": str,
    #   "confidence": float,
    #   "confidence_label": str,
    #   "confidence_metrics": {...}
    # }
```

### Пример: потоковая генерация с метриками уверенности (агрегация)

```python path=null start=null
from kraken_llm import create_streaming_client

async with create_streaming_client() as client:
    result = await client.chat_completion(
        messages=[{"role": "user", "content": "Кратко объясни ИИ"}],
        include_confidence=True  # Клиент соберёт метрики из streaming чанков
    )
    print(result["confidence"], result["confidence_label"])  # агрегированная уверенность по токенам
```

### Ручной расчёт по логпробам
Если вы получаете «сырые» ответы/чанки с logprobs, используйте утилиты:

```python path=null start=null
from kraken_llm.confidence.metrics import (
    confidence_from_chat_logprobs,
    token_confidences_from_stream_logprobs,
)

# Для обычного ответа (choice.logprobs)
metrics = confidence_from_chat_logprobs(choice.logprobs)
print(metrics["average_confidence"], metrics["confidence_label"]) 

# Для потоковых чанков (choice.logprobs)
per_token = token_confidences_from_stream_logprobs(chunk_choice.logprobs)
```

Примечание: поддержка logprobs зависит от провайдера/эндоинта и может отсутствовать.

### Фильтрация/перегенерация по уверенности

```python path=null start=null
from kraken_llm import create_standard_client
from kraken_llm.confidence.filter import ConfidenceFilterConfig, ensure_confident_chat

cfg = ConfidenceFilterConfig(
    min_confidence=0.8,
    max_attempts=3,
    prefer_streaming=True,          # Пытаться собирать пер-токенные метрики
    per_token_threshold=0.4,        # Порог низкой уверенности токена
    max_low_conf_fraction=0.3       # Не более 30% низкоуверенных токенов
)

async with create_standard_client() as client:
    result = await ensure_confident_chat(
        client,
        messages=[{"role": "user", "content": "Кратко объясни ИИ"}],
        cfg=cfg,
        max_tokens=200
    )

print(result["confidence"], result["confidence_label"], result["attempts_made"], result["success"])  
print(result["text"])  # итоговый текст
```

### Пример с UniversalLLMClient и logprobs

```python path=null start=null
from dotenv import load_dotenv
from kraken_llm import LLMConfig, create_universal_client

load_dotenv(".env")
config = LLMConfig()  # параметры возьмутся из .env

async with create_universal_client(config) as client:
    # Обычная генерация
    text = await client.chat_completion([{"role": "user", "content": "Поясни, что такое ИИ"}], max_tokens=150)

    # Генерация с метриками уверенности (logprobs)
    with_conf = await client.chat_completion(
        [{"role": "user", "content": "Поясни, что такое ИИ"}],
        include_confidence=True,
        max_tokens=150,
    )

    # Фильтрация/перегенерация по порогу уверенности
    from kraken_llm.confidence.filter import ConfidenceFilterConfig, ensure_confident_chat

    cfg = ConfidenceFilterConfig(min_confidence=0.8, max_attempts=3, prefer_streaming=True)
    filtered = await ensure_confident_chat(
        client,
        messages=[{"role": "user", "content": "Поясни, что такое ИИ простыми словами"}],
        cfg=cfg,
        max_tokens=180,
    )
```

### Подготовка окружения (.env)

```env
LLM_ENDPOINT=http://localhost:8080
LLM_API_KEY=your_api_key_or_token
LLM_MODEL=chat
# (опционально) глобально запросить logprobs
LLM_LOGPROBS=true
LLM_TOP_LOGPROBS=5
```

## Обработка ошибок

### Иерархия исключений

```python
from kraken_llm.exceptions import (
    KrakenError,           # Базовое исключение
    APIError,              # Ошибки API
    ValidationError,       # Ошибки валидации
    NetworkError,          # Сетевые ошибки
    AuthenticationError,   # Ошибки аутентификации
    RateLimitError,        # Превышение лимитов
)

try:
    response = await client.chat_completion([
        {"role": "user", "content": "Тест"}
    ])
except RateLimitError as e:
    print(f"Превышен лимит запросов: {e}")
    print(f"Повторить через: {e.retry_after} секунд")
except AuthenticationError as e:
    print(f"Ошибка аутентификации: {e}")
except ValidationError as e:
    print(f"Ошибка валидации: {e}")
    for detail in e.context.get("error_details", []):
        print(f"Поле {detail['field']}: {detail['message']}")
except KrakenError as e:
    print(f"Общая ошибка Kraken: {e}")
```

## Утилиты

### Работа с медиа файлами

```python
from kraken_llm.utils.media import MediaUtils

# Валидация медиа файла
validation = MediaUtils.validate_media_file(
    "image.jpg",
    media_type="image",
    max_size=10 * 1024 * 1024
)

# Изменение размера изображения
result = MediaUtils.resize_image(
    "large_image.jpg",
    "resized_image.jpg",
    max_width=1024,
    max_height=1024
)

# Создание data URL
data_url = MediaUtils.create_data_url("image.jpg")
```

## Тестирование

### Запуск тестов

```bash
# Все тесты
make test

# Только unit тесты
make test-unit

# Только integration тесты
make test-integration

# С покрытием
make test-coverage
```

### Тестирование возможностей

```python
async with create_universal_client() as client:
    # Автоматическое тестирование всех возможностей
    test_results = await client.test_capabilities()
    
    for capability, result in test_results.items():
        status = "✅" if result else "❌"
        print(f"{status} {capability}")
```

## Примеры использования

В папке `examples/` находятся подробные примеры:

- `quick_universal_example.py` - Быстрый старт с универсальным клиентом
- `universal_client_example.py` - Подробные примеры использования
- `complete_workflow_example.py` - Полный рабочий процесс
- `adaptive_capabilities_example.py` - Адаптивные возможности
- `structured_output_fallback_example.py` - Структурированный вывод с fallback
- `reasoning_example.py` - Рассуждающие модели
- `multimodal_example.py` - Мультимодальные операции
- `streaming_example.py` - Потоковые операции
- `chatbot_streaming_colors.py` - Интерактивный чат‑бот со стримингом и цветовой визуализацией уверенности
- `function_tool_example.py` - Функции и инструменты

## Архитектура

### Структура проекта

```
kraken_llm/
├── client/           # LLM клиенты
│   ├── base.py       # Базовый клиент
│   ├── standard.py   # Стандартный клиент
│   ├── streaming.py  # Потоковый клиент
│   ├── structured.py # Структурированный вывод
│   ├── reasoning.py  # Рассуждающие модели
│   ├── multimodal.py # Мультимодальный клиент
│   ├── adaptive.py   # Адаптивный клиент
│   ├── asr.py        # Речевые технологии
│   ├── embeddings.py # Векторные представления
│   ├── universal.py  # Универсальный клиент
│   └── factory.py    # Фабрика клиентов
├── tools/            # Система функций и инструментов
├── streaming/        # Потоковые операции
├── structured/       # Структурированный вывод
├── utils/           # Утилиты (медиа, логирование)
├── exceptions/       # Обработка ошибок
└── models/          # Модели данных
```

### Принципы архитектуры

1. **Модульность**: Каждый компонент имеет четко определенную ответственность
2. **Расширяемость**: Легко добавлять новые типы клиентов и функциональность
3. **Типобезопасность**: Полная поддержка type hints во всех компонентах
4. **Асинхронность**: Все операции построены на async/await
5. **Конфигурируемость**: Гибкая система настроек через Pydantic Settings
6. **Обработка ошибок**: Иерархическая система исключений с контекстом
7. **Автоопределение**: Автоматический выбор подходящего клиента через фабрику

## Лицензия

MIT License - делайте что хотите ;-). См. файл [LICENSE](LICENSE) для подробностей.

## Поддержка

- Примеры: [examples/](examples/)
- Тесты: [tests/](tests/)