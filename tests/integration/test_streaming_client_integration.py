"""
Integration тесты для StreamingLLMClient.

Этот модуль содержит интеграционные тесты для проверки работы потокового LLM клиента
с реальными API вызовами, включая потоковые операции, function/tool calling и обработку ошибок.
"""

import asyncio
import json
import os
from typing import List

import pytest

from kraken_llm.client.streaming import StreamingLLMClient
from kraken_llm.config.settings import LLMConfig
from kraken_llm.exceptions.api import APIError
from kraken_llm.exceptions.network import NetworkError
from kraken_llm.exceptions.validation import ValidationError

from dotenv import load_dotenv

load_dotenv()


@pytest.mark.integration
class TestStreamingLLMClientIntegration:
    """Интеграционные тесты для StreamingLLMClient."""

    @pytest.fixture
    def config(self):
        """Фикстура конфигурации для интеграционных тестов."""
        return LLMConfig(
            endpoint=os.getenv("LLM_ENDPOINT", "http://localhost:8080"),
            api_key=os.getenv("LLM_TOKEN", "test-key"),
            model=os.getenv("LLM_MODEL", "chat"),
            temperature=0.7,
            max_tokens=100,  # Ограничиваем для быстрых тестов
            connect_timeout=5.0,
            read_timeout=30.0
        )

    @pytest.fixture
    async def client(self, config):
        """Фикстура клиента для интеграционных тестов."""
        client = StreamingLLMClient(config)
        yield client
        await client.close()

    @pytest.mark.asyncio
    async def test_basic_streaming_chat_completion(self, client):
        """Тест базового потокового chat completion."""
        messages = [
            {"role": "user", "content": "Скажи 'Привет' и больше ничего"}
        ]

        # Тестируем потоковый режим
        chunks = []
        async for chunk in client.chat_completion_stream(messages):
            chunks.append(chunk)
            # Прерываем после получения достаточного количества chunks
            if len(chunks) >= 5:
                break

        # Проверяем что получили chunks
        assert len(chunks) > 0

        # Проверяем что chunks содержат текст
        full_response = "".join(chunks)
        assert len(full_response) > 0
        assert isinstance(full_response, str)

        print(
            f"Получено {len(chunks)} chunks, полный ответ: '{full_response}'")

    @pytest.mark.asyncio
    async def test_aggregated_streaming_chat_completion(self, client):
        """Тест агрегированного потокового chat completion."""
        messages = [
            {"role": "user", "content": "Ответь одним словом: 'Тест'"}
        ]

        # Тестируем агрегированный режим
        response = await client.chat_completion(messages)

        # Проверяем ответ
        assert isinstance(response, str)
        assert len(response) > 0

        print(f"Агрегированный ответ: '{response}'")

    @pytest.mark.asyncio
    async def test_streaming_with_system_message(self, client):
        """Тест потокового режима с системным сообщением."""
        messages = [
            {"role": "system", "content": "Ты полезный ассистент. Отвечай кратко."},
            {"role": "user", "content": "Что такое Python?"}
        ]

        chunks = []
        async for chunk in client.chat_completion_stream(messages):
            chunks.append(chunk)
            # Ограничиваем количество chunks для быстрого теста
            if len(chunks) >= 10:
                break

        assert len(chunks) > 0
        full_response = "".join(chunks)
        assert "python" in full_response.lower() or "Python" in full_response

        print(f"Ответ с системным сообщением: '{full_response}'")

    @pytest.mark.asyncio
    async def test_streaming_with_temperature_control(self, client):
        """Тест потокового режима с контролем температуры."""
        messages = [
            {"role": "user", "content": "Скажи число от 1 до 10"}
        ]

        # Тест с низкой температурой (более детерминированный)
        chunks_low_temp = []
        async for chunk in client.chat_completion_stream(messages, temperature=0.1):
            chunks_low_temp.append(chunk)
            if len(chunks_low_temp) >= 5:
                break

        # Тест с высокой температурой (более креативный)
        chunks_high_temp = []
        async for chunk in client.chat_completion_stream(messages, temperature=0.9):
            chunks_high_temp.append(chunk)
            if len(chunks_high_temp) >= 5:
                break

        # Проверяем что оба теста дали результаты
        assert len(chunks_low_temp) > 0
        assert len(chunks_high_temp) > 0

        response_low = "".join(chunks_low_temp)
        response_high = "".join(chunks_high_temp)

        print(f"Низкая температура: '{response_low}'")
        print(f"Высокая температура: '{response_high}'")

    @pytest.mark.asyncio
    async def test_streaming_with_max_tokens_limit(self, client):
        """Тест потокового режима с ограничением токенов."""
        messages = [
            {"role": "user", "content": "Расскажи длинную историю"}
        ]

        # Тест с очень малым лимитом токенов
        chunks = []
        async for chunk in client.chat_completion_stream(messages, max_tokens=20):
            chunks.append(chunk)

        full_response = "".join(chunks)

        # Ответ должен быть коротким из-за лимита токенов
        assert len(full_response) > 0
        # Не проверяем точную длину, так как токенизация может отличаться

        print(
            f"Ответ с лимитом токенов: '{full_response}' (длина: {len(full_response)})")

    @pytest.mark.asyncio
    async def test_function_calling_registration(self, client):
        """Тест регистрации функций для function calling."""

        def get_weather(location: str) -> str:
            """Получить погоду для указанного места."""
            return f"Солнечно, 25°C в {location}"

        def calculate_sum(a: int, b: int) -> int:
            """Вычислить сумму двух чисел."""
            return a + b

        # Регистрируем функции
        client.register_function("get_weather", get_weather, "Получить погоду")
        client.register_function(
            "calculate_sum", calculate_sum, "Вычислить сумму")

        # Проверяем регистрацию
        registered_functions = client.get_registered_functions()
        assert "get_weather" in registered_functions
        assert "calculate_sum" in registered_functions
        assert len(registered_functions) == 2

        print(f"Зарегистрированные функции: {registered_functions}")

    @pytest.mark.asyncio
    async def test_tool_calling_registration(self, client):
        """Тест регистрации инструментов для tool calling."""

        async def search_web(query: str) -> str:
            """Поиск в интернете."""
            await asyncio.sleep(0.1)  # Имитация асинхронной работы
            return f"Результаты поиска для: {query}"

        def format_text(text: str, style: str = "bold") -> str:
            """Форматирование текста."""
            if style == "bold":
                return f"**{text}**"
            elif style == "italic":
                return f"*{text}*"
            return text

        # Регистрируем инструменты
        client.register_tool("search_web", search_web, "Поиск в интернете")
        client.register_tool("format_text", format_text,
                             "Форматирование текста")

        # Проверяем регистрацию
        registered_tools = client.get_registered_tools()
        assert "search_web" in registered_tools
        assert "format_text" in registered_tools
        assert len(registered_tools) == 2

        print(f"Зарегистрированные инструменты: {registered_tools}")

    @pytest.mark.asyncio
    async def test_streaming_error_handling_invalid_endpoint(self):
        """Тест обработки ошибок при некорректном endpoint."""
        config = LLMConfig(
            endpoint="http://invalid-endpoint-12345.com",
            api_key="test-key",
            model="test-model",
            connect_timeout=2.0,  # Быстрый таймаут для теста
            read_timeout=5.0
        )

        client = StreamingLLMClient(config)
        messages = [{"role": "user", "content": "test"}]

        try:
            # Ожидаем NetworkError или TimeoutError
            chunks = []
            async for chunk in client.chat_completion_stream(messages):
                chunks.append(chunk)
        except (NetworkError, APIError) as e:
            # Это ожидаемое поведение
            print(f"Получена ожидаемая ошибка: {type(e).__name__}: {e}")
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_streaming_validation_errors(self, client):
        """Тест валидационных ошибок в потоковом режиме."""

        # Тест пустых сообщений
        with pytest.raises(ValidationError):
            async for chunk in client.chat_completion_stream([]):
                pass

        # Тест некорректной структуры сообщения
        with pytest.raises(ValidationError):
            async for chunk in client.chat_completion_stream([{"role": "invalid"}]):
                pass

        # Тест function_call без functions
        with pytest.raises(ValidationError):
            async for chunk in client.chat_completion_stream(
                [{"role": "user", "content": "test"}],
                function_call="auto"
            ):
                pass

        # Тест tool_choice без tools
        with pytest.raises(ValidationError):
            async for chunk in client.chat_completion_stream(
                [{"role": "user", "content": "test"}],
                tool_choice="auto"
            ):
                pass

        print("Все валидационные ошибки обработаны корректно")

    @pytest.mark.asyncio
    async def test_streaming_context_manager(self, config):
        """Тест использования клиента как context manager."""
        messages = [{"role": "user", "content": "Привет"}]

        async with StreamingLLMClient(config) as client:
            chunks = []
            async for chunk in client.chat_completion_stream(messages):
                chunks.append(chunk)
                if len(chunks) >= 3:
                    break

            assert len(chunks) > 0
            print(f"Context manager тест: получено {len(chunks)} chunks")

        # Клиент должен быть автоматически закрыт

    @pytest.mark.asyncio
    async def test_concurrent_streaming_requests(self, client):
        """Тест параллельных потоковых запросов."""
        messages1 = [{"role": "user", "content": "Скажи 'A'"}]
        messages2 = [{"role": "user", "content": "Скажи 'B'"}]
        messages3 = [{"role": "user", "content": "Скажи 'C'"}]

        # Запускаем три параллельных потоковых запроса
        async def collect_chunks(messages, limit=5):
            chunks = []
            async for chunk in client.chat_completion_stream(messages):
                chunks.append(chunk)
                if len(chunks) >= limit:
                    break
            return chunks

        # Выполняем запросы параллельно
        results = await asyncio.gather(
            collect_chunks(messages1),
            collect_chunks(messages2),
            collect_chunks(messages3),
            return_exceptions=True
        )

        # Проверяем что все запросы завершились успешно
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Запрос {i+1} завершился с ошибкой: {result}")
            else:
                assert len(result) > 0
                print(f"Запрос {i+1}: получено {len(result)} chunks")

    @pytest.mark.asyncio
    async def test_streaming_performance_monitoring(self, client):
        """Тест мониторинга производительности потоковых операций."""
        import time

        messages = [{"role": "user", "content": "Считай от 1 до 5"}]

        start_time = time.time()
        chunk_times = []
        chunks = []

        async for chunk in client.chat_completion_stream(messages):
            chunk_time = time.time()
            chunk_times.append(chunk_time - start_time)
            chunks.append(chunk)

            # Ограничиваем для теста
            if len(chunks) >= 10:
                break

        total_time = time.time() - start_time

        # Анализ производительности
        if chunk_times:
            first_chunk_time = chunk_times[0]
            avg_chunk_interval = (
                chunk_times[-1] - chunk_times[0]) / max(1, len(chunk_times) - 1)

            print(f"Производительность потока:")
            print(f"  - Время до первого chunk: {first_chunk_time:.3f}s")
            print(f"  - Общее время: {total_time:.3f}s")
            print(f"  - Количество chunks: {len(chunks)}")
            print(
                f"  - Средний интервал между chunks: {avg_chunk_interval:.3f}s")

            # Базовые проверки производительности
            assert first_chunk_time < 10.0  # Первый chunk должен прийти быстро
            assert total_time < 30.0  # Общее время не должно быть слишком большим

        assert len(chunks) > 0
