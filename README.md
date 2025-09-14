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
- **Анализ возможностей**: Автоматическое определение возможностей моделей

## Установка

### Базовая установка

```bash
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

- Документация: [docs/](docs/)
- Примеры: [examples/](examples/)
- Тесты: [tests/](tests/)