#!/usr/bin/env python3
"""
Пример использования StreamingLLMClient из Kraken фреймворка.

Этот скрипт демонстрирует различные возможности потокового LLM клиента:
- Базовые потоковые запросы с real-time выводом
- Агрегированные потоковые ответы
- Имитация Function calling (API не поддерживает)
- Имитация Tool calling (API не поддерживает)
- Обработка ошибок и мониторинг производительности

ПРИМЕЧАНИЕ: Текущий API поддерживает только streaming режим.
Tool/Function calling имитируется через локальные функции.
"""

import asyncio
import json
import os
import time
from dotenv import load_dotenv
from typing import Dict, Any

from kraken_llm.client.streaming import StreamingLLMClient
from kraken_llm.config.settings import LLMConfig
from kraken_llm.exceptions.base import KrakenError

load_dotenv()


async def basic_streaming_example():
    """Пример базового потокового запроса."""
    print("=== Базовый потоковый запрос ===")

    # Создание конфигурации
    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL"),
        temperature=0.7,
        max_tokens=200
    )

    # Создание клиента
    async with StreamingLLMClient(config) as client:
        messages = [
            {"role": "system", "content": "Ты полезный ассистент. Отвечай на русском языке."},
            {"role": "user", "content": "Расскажи короткую историю о роботе, который научился мечтать"}
        ]

        print("Отправка запроса...")
        print("Ответ (в реальном времени):")
        print("-" * 50)

        start_time = time.time()
        chunk_count = 0

        # Потоковый вывод
        async for chunk in client.chat_completion_stream(messages):
            print(chunk, end="", flush=True)
            chunk_count += 1

        elapsed_time = time.time() - start_time
        print(f"\n{'-' * 50}")
        print(f"Получено {chunk_count} chunks за {elapsed_time:.2f} секунд")


async def aggregated_streaming_example():
    """Пример агрегированного потокового ответа."""
    print("\n=== Агрегированный потоковый ответ ===")
    print("💡 Собираем streaming chunks в полный ответ")

    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL"),
        temperature=0.5,
        max_tokens=250
    )

    async with StreamingLLMClient(config) as client:
        messages = [
            {"role": "user", "content": "Объясни что такое машинное обучение простыми словами"}
        ]

        print("Отправка запроса для агрегированного ответа...")

        start_time = time.time()

        # Агрегированный ответ (собираем chunks вручную)
        response_chunks = []
        async for chunk in client.chat_completion_stream(messages):
            response_chunks.append(chunk)

        response = "".join(response_chunks)
        elapsed_time = time.time() - start_time

        print("Полный ответ:")
        print("-" * 50)
        print(response)
        print(f"{'-' * 50}")
        print(f"Время выполнения: {elapsed_time:.2f} секунд")
        print(f"Длина ответа: {len(response)} символов")
        print(f"Количество chunks: {len(response_chunks)}")


async def function_calling_streaming_example():
    """Пример имитации function calling в потоковом режиме (API не поддерживает function calling)."""
    print("\n=== Имитация Function Calling в потоковом режиме ===")
    print("⚠️  Примечание: Текущий API не поддерживает function calling, показываем альтернативный подход")

    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL"),
        temperature=0.3
    )

    async with StreamingLLMClient(config) as client:

        # Локальные функции
        def get_weather(city: str) -> str:
            """Получить информацию о погоде в указанном городе."""
            weather_data = {
                "москва": "Солнечно, +15°C, легкий ветер",
                "санкт-петербург": "Облачно, +12°C, дождь",
                "новосибирск": "Снег, -5°C, сильный ветер",
                "екатеринбург": "Пасмурно, +8°C, без осадков"
            }
            return weather_data.get(city.lower(), f"Данные о погоде для {city} недоступны")

        def calculate_distance(city1: str, city2: str) -> str:
            """Вычислить расстояние между городами."""
            distances = {
                ("москва", "санкт-петербург"): "635 км",
                ("москва", "новосибирск"): "3354 км",
                ("москва", "екатеринбург"): "1416 км",
                ("санкт-петербург", "новосибирск"): "3989 км"
            }
            key = tuple(sorted([city1.lower(), city2.lower()]))
            return distances.get(key, f"Расстояние между {city1} и {city2} неизвестно")

        # Сначала спрашиваем модель
        messages = [
            {"role": "user", "content": "Мне нужна информация о погоде в Москве и расстоянии от Москвы до Санкт-Петербурга. Ответь, что тебе нужно для этого."}
        ]

        print("Запрос к модели...")
        print("Ответ:")
        print("-" * 50)

        response_chunks = []
        async for chunk in client.chat_completion_stream(messages=messages):
            print(chunk, end="", flush=True)
            response_chunks.append(chunk)

        # Выполняем "function calls"
        print(f"\n{'-' * 50}")
        print("🔧 Выполнение локальных 'function calls':")

        weather_result = get_weather("москва")
        distance_result = calculate_distance("москва", "санкт-петербург")

        print(f"  get_weather('москва') -> {weather_result}")
        print(
            f"  calculate_distance('москва', 'санкт-петербург') -> {distance_result}")

        # Отправляем результаты обратно модели
        follow_up_messages = messages + [
            {"role": "assistant", "content": "".join(response_chunks)},
            {"role": "user", "content": f"Вот данные которые ты запросил:\n- Погода в Москве: {weather_result}\n- Расстояние от Москвы до Санкт-Петербурга: {distance_result}\n\nТеперь дай полный ответ."}
        ]

        print("\nФинальный ответ с данными:")
        print("-" * 50)

        async for chunk in client.chat_completion_stream(messages=follow_up_messages):
            print(chunk, end="", flush=True)

        print(f"\n{'-' * 50}")
        print("💡 Это демонстрация того, как можно имитировать function calling без поддержки API")


async def tool_calling_streaming_example():
    """Пример имитации tool calling в потоковом режиме (API не поддерживает tool calling)."""
    print("\n=== Имитация Tool Calling в потоковом режиме ===")
    print("⚠️  Примечание: Текущий API не поддерживает tool calling, показываем альтернативный подход")

    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL"),
        temperature=0.3
    )

    async with StreamingLLMClient(config) as client:

        # Локальные функции для имитации tool calling
        def search_database(query: str) -> str:
            """Поиск информации в базе данных."""
            database = {
                "python": "Высокоуровневый язык программирования общего назначения",
                "машинное обучение": "Область искусственного интеллекта, изучающая алгоритмы обучения",
                "нейронные сети": "Вычислительные системы, вдохновленные биологическими нейронными сетями",
                "api": "Интерфейс программирования приложений для взаимодействия между программами"
            }

            for key, value in database.items():
                if key.lower() in query.lower():
                    return f"Найдено: {value}"

            return f"Информация по запросу '{query}' не найдена в базе данных"

        # Сначала получаем ответ от модели
        messages = [
            {"role": "user", "content": "Расскажи о Python как языке программирования"}
        ]

        print("Получение ответа от модели...")
        print("Ответ:")
        print("-" * 50)

        response_chunks = []
        async for chunk in client.chat_completion_stream(messages=messages):
            print(chunk, end="", flush=True)
            response_chunks.append(chunk)

        response = "".join(response_chunks)

        # Имитируем "tool call" - ищем дополнительную информацию
        print(f"\n{'-' * 50}")
        print("🔧 Выполнение локального 'tool call': search_database('python')")

        tool_result = search_database("python")
        print(f"Результат: {tool_result}")

        # Можем отправить дополнительный запрос с контекстом
        follow_up_messages = [
            {"role": "user", "content": "Расскажи о Python как языке программирования"},
            {"role": "assistant", "content": response},
            {"role": "user", "content": f"Дополнительная информация из базы данных: {tool_result}. Можешь дополнить свой ответ?"}
        ]

        print("\nДополнительный ответ с учетом 'tool result':")
        print("-" * 50)

        async for chunk in client.chat_completion_stream(messages=follow_up_messages):
            print(chunk, end="", flush=True)

        print(f"\n{'-' * 50}")
        print(
            "💡 Это демонстрация того, как можно имитировать tool calling без поддержки API")


async def performance_monitoring_example():
    """Пример мониторинга производительности потоковых операций."""
    print("\n=== Мониторинг производительности ===")

    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL"),
        temperature=0.7,
        max_tokens=300
    )

    async with StreamingLLMClient(config) as client:
        messages = [
            {"role": "user", "content": "Расскажи о преимуществах потокового программирования"}
        ]

        print("Мониторинг производительности потокового запроса...")

        start_time = time.time()
        first_chunk_time = None
        chunk_times = []
        chunks = []

        async for chunk in client.chat_completion_stream(messages):
            current_time = time.time()

            if first_chunk_time is None:
                first_chunk_time = current_time - start_time

            chunk_times.append(current_time - start_time)
            chunks.append(chunk)

        total_time = time.time() - start_time

        # Анализ производительности
        print("\nСтатистика производительности:")
        print(f"  • Время до первого chunk: {first_chunk_time:.3f}s")
        print(f"  • Общее время выполнения: {total_time:.3f}s")
        print(f"  • Количество chunks: {len(chunks)}")
        print(f"  • Средняя скорость: {len(chunks)/total_time:.1f} chunks/s")

        if len(chunk_times) > 1:
            intervals = [chunk_times[i] - chunk_times[i-1]
                         for i in range(1, len(chunk_times))]
            avg_interval = sum(intervals) / len(intervals)
            print(f"  • Средний интервал между chunks: {avg_interval:.3f}s")

        total_chars = sum(len(chunk) for chunk in chunks)
        print(f"  • Общее количество символов: {total_chars}")
        print(
            f"  • Скорость генерации: {total_chars/total_time:.1f} символов/s")


async def error_handling_example():
    """Пример обработки ошибок в потоковом режиме."""
    print("\n=== Обработка ошибок ===")

    # Тест с некорректным endpoint
    print("1. Тест с некорректным endpoint:")
    config_invalid = LLMConfig(
        endpoint="http://invalid-endpoint-12345.com",
        api_key="test-key",
        model="test-model",
        connect_timeout=2.0,
        read_timeout=5.0
    )

    try:
        async with StreamingLLMClient(config_invalid) as client:
            messages = [{"role": "user", "content": "test"}]

            async for chunk in client.chat_completion_stream(messages):
                print(chunk, end="")

    except KrakenError as e:
        print(f"   Получена ожидаемая ошибка: {type(e).__name__}: {e}")

    # Тест валидационных ошибок
    print("\n2. Тест валидационных ошибок:")
    config_valid = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL"),
    )

    async with StreamingLLMClient(config_valid) as client:

        # Пустые сообщения
        try:
            async for chunk in client.chat_completion_stream([]):
                pass
        except KrakenError as e:
            print(f"   Пустые сообщения: {type(e).__name__}: {e}")

        # Некорректная структура сообщения
        try:
            async for chunk in client.chat_completion_stream([{"role": "invalid"}]):
                pass
        except KrakenError as e:
            print(f"   Некорректная структура: {type(e).__name__}: {e}")

        # function_call без functions
        try:
            async for chunk in client.chat_completion_stream(
                [{"role": "user", "content": "test"}],
                function_call="auto"
            ):
                pass
        except KrakenError as e:
            print(f"   function_call без functions: {type(e).__name__}: {e}")


async def concurrent_requests_example():
    """Пример параллельных потоковых запросов."""
    print("\n=== Параллельные потоковые запросы ===")

    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL"),
        temperature=0.5,
        max_tokens=100
    )

    async with StreamingLLMClient(config) as client:

        # Определяем разные запросы
        requests = [
            [{"role": "user", "content": "Что такое Python?"}],
            [{"role": "user", "content": "Что такое JavaScript?"}],
            [{"role": "user", "content": "Что такое машинное обучение?"}]
        ]

        async def process_request(request_id: int, messages: list):
            """Обработка одного запроса."""
            print(f"\nЗапрос {request_id + 1} начат...")

            start_time = time.time()
            chunks = []

            async for chunk in client.chat_completion_stream(messages):
                chunks.append(chunk)

            elapsed_time = time.time() - start_time
            response = "".join(chunks)

            print(f"Запрос {request_id + 1} завершен за {elapsed_time:.2f}s")
            print(f"Ответ {request_id + 1}: {response[:100]}...")

            return {
                "request_id": request_id + 1,
                "response": response,
                "elapsed_time": elapsed_time,
                "chunks_count": len(chunks)
            }

        print("Запуск 3 параллельных потоковых запросов...")
        start_time = time.time()

        # Выполняем запросы параллельно
        results = await asyncio.gather(
            *[process_request(i, req) for i, req in enumerate(requests)],
            return_exceptions=True
        )

        total_time = time.time() - start_time

        print(f"\nВсе запросы завершены за {total_time:.2f}s")

        # Анализ результатов
        successful_results = [
            r for r in results if not isinstance(r, Exception)]
        failed_results = [r for r in results if isinstance(r, Exception)]

        print(f"Успешных запросов: {len(successful_results)}")
        print(f"Неудачных запросов: {len(failed_results)}")

        if successful_results:
            avg_time = sum(r["elapsed_time"]
                           for r in successful_results) / len(successful_results)
            print(f"Среднее время выполнения: {avg_time:.2f}s")


async def main():
    """Главная функция с примерами использования StreamingLLMClient."""
    print("🚀 Примеры использования StreamingLLMClient")
    print("=" * 60)

    try:
        # Базовые примеры
        await basic_streaming_example()
        await aggregated_streaming_example()

        # Function и Tool calling (имитация, так как API не поддерживает)
        await function_calling_streaming_example()
        await tool_calling_streaming_example()

        # Мониторинг и производительность
        await performance_monitoring_example()

        # Обработка ошибок
        await error_handling_example()

        # Параллельные запросы
        await concurrent_requests_example()

        print("\n✅ Все примеры выполнены успешно!")

    except KeyboardInterrupt:
        print("\n⚠️ Выполнение прервано пользователем")
    except Exception as e:
        print(f"\n❌ Произошла ошибка: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Запуск примеров
    asyncio.run(main())
