# Kraken LLM - Быстрый старт

Это руководство поможет вам быстро начать работу с Kraken LLM Framework.

## Установка

```bash
# Клонирование репозитория
git clone https://github.com/example/kraken-llm
cd kraken-llm

# Установка зависимостей
pip install -e .

# Для разработки
pip install -e .[dev]
```

## Настройка окружения

Создайте файл `.env` в корне проекта:

```env
LLM_ENDPOINT=http://localhost:8080
LLM_API_KEY=your-api-key
LLM_MODEL=chat
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2000
```

Или установите переменные окружения:

```bash
export LLM_ENDPOINT="http://localhost:8080"
export LLM_API_KEY="your-api-key"
export LLM_MODEL="chat"
```

## Первые шаги

### 1. Простейший пример

```python
import asyncio
from kraken_llm import create_universal_client

async def main():
    async with create_universal_client() as client:
        response = await client.chat_completion([
            {"role": "user", "content": "Привет, мир!"}
        ])
        print(response)

asyncio.run(main())
```

### 2. Анализ возможностей модели

Перед началом работы рекомендуется проанализировать возможности вашей модели:

```bash
# Быстрый анализ
python3 model_capabilities_analyzer.py --quick

# Полный анализ
python3 model_capabilities_analyzer.py

# Сохранение в Markdown
python3 model_capabilities_analyzer.py --output markdown --filename capabilities.md
```

### 3. Создание клиента на основе анализа

```python
import asyncio
from kraken_llm import create_universal_client_from_report
from model_capabilities_analyzer import ModelCapabilitiesAnalyzer

async def main():
    # Анализируем возможности
    analyzer = ModelCapabilitiesAnalyzer()
    report = await analyzer.analyze_all_models(quick_mode=True)
    
    # Создаем оптимальный клиент
    async with create_universal_client_from_report(report) as client:
        print(f"Доступные возможности: {client.get_available_capabilities()}")
        
        response = await client.chat_completion([
            {"role": "user", "content": "Тест возможностей"}
        ])
        print(response)

asyncio.run(main())
```

## Основные возможности

### Базовый клиент

```python
from kraken_llm import create_basic_client

async with create_basic_client() as client:
    # Простой чат
    response = await client.chat_completion([
        {"role": "user", "content": "Как дела?"}
    ])
    
    # Потоковый ответ
    async for chunk in client.chat_completion_stream([
        {"role": "user", "content": "Расскажи историю"}
    ]):
        print(chunk, end="", flush=True)
```

### Продвинутый клиент

```python
from kraken_llm import create_advanced_client
from pydantic import BaseModel

class Task(BaseModel):
    title: str
    priority: int
    completed: bool = False

async with create_advanced_client() as client:
    # Структурированный вывод с автоматическим fallback
    task = await client.chat_completion_structured([
        {"role": "user", "content": "Создай задачу изучить Python"}
    ], response_model=Task)
    
    print(f"Задача: {task.title}, Приоритет: {task.priority}")
    
    # Рассуждения
    response = await client.reasoning_completion([
        {"role": "user", "content": "Сколько будет 15 + 27?"}
    ])
```

### Полнофункциональный клиент

```python
from kraken_llm import create_full_client

async with create_full_client() as client:
    # Все возможности доступны
    capabilities = client.get_available_capabilities()
    print(f"Все возможности: {capabilities}")
    
    # Тестирование возможностей
    test_results = await client.test_capabilities()
    for capability, result in test_results.items():
        status = "✅" if result else "❌"
        print(f"{status} {capability}")
```

## Кастомная конфигурация

### Выбор конкретных возможностей

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

### Настройка конфигурации

```python
from kraken_llm import LLMConfig, create_universal_client

# Кастомная конфигурация
config = LLMConfig(
    endpoint="http://localhost:8080",
    api_key="your-api-key",
    model="chat",
    temperature=0.7,
    max_tokens=1000,
    stream=False
)

async with create_universal_client(config=config) as client:
    response = await client.chat_completion([
        {"role": "user", "content": "Тест с кастомной конфигурацией"}
    ])
```

## Специализированные клиенты

### Структурированный вывод

```python
from kraken_llm import create_structured_client
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    skills: list[str]

async with create_structured_client() as client:
    person = await client.chat_completion_structured([
        {"role": "user", "content": "Создай профиль Python разработчика"}
    ], response_model=Person)
    
    print(f"Имя: {person.name}")
    print(f"Возраст: {person.age}")
    print(f"Навыки: {', '.join(person.skills)}")
```

### Потоковая передача

```python
from kraken_llm import create_streaming_client

async with create_streaming_client() as client:
    print("Потоковый ответ:")
    async for chunk in client.chat_completion_stream([
        {"role": "user", "content": "Расскажи длинную историю"}
    ]):
        print(chunk, end="", flush=True)
    print()
```

### Рассуждающие модели

```python
from kraken_llm import create_reasoning_client, ReasoningConfig

config = ReasoningConfig(
    enable_cot=True,
    max_reasoning_steps=5
)

async with create_reasoning_client(reasoning_config=config) as client:
    response = await client.reasoning_completion([
        {"role": "user", "content": "Реши математическую задачу: 15 * 23 + 45"}
    ], problem_type="math")
    
    # Доступ к шагам рассуждения
    if hasattr(response, 'steps'):
        for step in response.steps:
            print(f"Шаг {step.step_number}: {step.thought}")
```

### Мультимодальность

```python
from kraken_llm import create_multimodal_client

async with create_multimodal_client() as client:
    # Анализ изображения
    response = await client.vision_completion(
        text_prompt="Опиши что видишь на изображении",
        images="path/to/image.jpg"
    )
    print(response)
```

## Function Calling

### Регистрация функций

```python
def get_weather(city: str) -> str:
    """Получить погоду в указанном городе."""
    return f"В городе {city} сейчас солнечно, +20°C"

async with create_universal_client() as client:
    # Регистрируем функцию
    client.register_function(
        name="get_weather",
        function=get_weather,
        description="Получить текущую погоду",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "Название города"}
            },
            "required": ["city"]
        }
    )
    
    # Используем функцию
    response = await client.chat_completion([
        {"role": "user", "content": "Какая погода в Москве?"}
    ])
    print(response)
```

### Декораторы

```python
from kraken_llm.tools import register_function

@register_function("calculate", "Выполнить математические вычисления")
def calculate(expression: str) -> float:
    """Безопасное вычисление математических выражений."""
    # В реальности используйте безопасный парсер
    return eval(expression)

# Функция автоматически доступна во всех клиентах
```

## Обработка ошибок

```python
from kraken_llm.exceptions import (
    KrakenError,
    ValidationError,
    APIError,
    RateLimitError
)

try:
    async with create_universal_client() as client:
        response = await client.chat_completion([
            {"role": "user", "content": "Тест"}
        ])
except RateLimitError as e:
    print(f"Превышен лимит: {e}")
    print(f"Повторить через: {e.retry_after} секунд")
except ValidationError as e:
    print(f"Ошибка валидации: {e}")
except APIError as e:
    print(f"Ошибка API: {e}")
except KrakenError as e:
    print(f"Общая ошибка: {e}")
```

## Полезные команды

### Makefile команды

```bash
# Анализ возможностей
make capabilities-analyze-quick
make capabilities-analyze-full

# Тестирование
make test
make test-unit
make test-integration
make test-coverage

# Форматирование кода
make format
make lint

# Очистка
make clean
```

### Запуск примеров

```bash
# Быстрый старт с универсальным клиентом
python3 examples/quick_universal_example.py

# Подробные примеры
python3 examples/universal_client_example.py

# Полный рабочий процесс
python3 examples/complete_workflow_example.py

# Структурированный вывод с fallback
python3 examples/structured_output_fallback_example.py
```

## Следующие шаги

1. **Изучите примеры**: Посмотрите файлы в папке `examples/` для подробных примеров использования
2. **Проанализируйте модель**: Запустите `model_capabilities_analyzer.py` для анализа возможностей вашей модели
3. **Настройте конфигурацию**: Создайте оптимальную конфигурацию для ваших задач
4. **Изучите документацию**: Прочитайте полную документацию в `README.md`
5. **Экспериментируйте**: Попробуйте различные типы клиентов и возможности
