#!/usr/bin/env python3
"""
Тестирование возможностей моделей.
"""

import asyncio
import os
import time
import json
from typing import Dict, List, Any, Optional, AsyncGenerator
from dataclasses import dataclass
from dotenv import load_dotenv
from pydantic import BaseModel

from kraken_llm.client.adaptive import AdaptiveLLMClient, AdaptiveConfig, ModelCapability
from kraken_llm.client.factory import ClientFactory
from kraken_llm.config.settings import LLMConfig
from kraken_llm.client.standard import StandardLLMClient
from kraken_llm.client.streaming import StreamingLLMClient
from kraken_llm.client.structured import StructuredLLMClient
from kraken_llm.client.reasoning import ReasoningLLMClient
from kraken_llm.client.multimodal import MultimodalLLMClient
from kraken_llm.client.embeddings import EmbeddingsLLMClient
from kraken_llm.client.completion import CompletionLLMClient

load_dotenv()


@dataclass
class ModelConfig:
    """Конфигурация модели для тестирования"""
    name: str
    endpoint: str
    token: str
    model: str
    description: str
    model_type: str  # 'chat', 'completion', 'embedding', etc


@dataclass
class TestResult:
    """Результат тестирования возможности"""
    capability: str
    client_type: str
    model_config: str
    success: bool
    response_time: float
    error: Optional[str] = None
    response_preview: Optional[str] = None


class TestModel(BaseModel):
    """Тестовая модель для structured output"""
    name: str
    value: int
    description: str


def get_model_configs() -> List[ModelConfig]:
    """Получить все конфигурации моделей из .env"""
    configs = []

    # Chat модель (использует /v1/chat/completions)
    if all([os.getenv("CHAT_ENDPOINT"), os.getenv("CHAT_TOKEN"), os.getenv("CHAT_MODEL")]):
        configs.append(ModelConfig(
            name="chat_model",
            endpoint=os.getenv("CHAT_ENDPOINT"),
            token=os.getenv("CHAT_TOKEN"),
            model=os.getenv("CHAT_MODEL"),
            description="Chat модель (/v1/chat/completions)",
            model_type="chat"
        ))

    # Completion модель (использует /v1/completions)
    if all([os.getenv("COMPLETION_ENDPOINT"), os.getenv("COMPLETION_TOKEN"), os.getenv("COMPLETION_MODEL")]):
        configs.append(ModelConfig(
            name="completion_model",
            endpoint=os.getenv("COMPLETION_ENDPOINT"),
            token=os.getenv("COMPLETION_TOKEN"),
            model=os.getenv("COMPLETION_MODEL"),
            description="Completion модель (/v1/completions)",
            model_type="completion"
        ))

    # Модель для эмбеддингов
    if all([os.getenv("EMBEDDING_ENDPOINT"), os.getenv("EMBEDDING_TOKEN"), os.getenv("EMBEDDING_MODEL")]):
        configs.append(ModelConfig(
            name="embedding_model",
            endpoint=os.getenv("EMBEDDING_ENDPOINT"),
            token=os.getenv("EMBEDDING_TOKEN"),
            model=os.getenv("EMBEDDING_MODEL"),
            description="Модель для эмбеддингов",
            model_type="embedding"
        ))

    return configs


async def test_basic_chat_completion(client, model_name: str) -> TestResult:
    """Тестирование базового chat completion"""
    start_time = time.time()

    messages = [
        {"role": "user", "content": "Привет! Ответь одним словом: 'Работает'"}
    ]

    try:
        response = await client.chat_completion(messages, max_tokens=10)
        response_time = time.time() - start_time

        return TestResult(
            capability="chat_completion",
            client_type=type(client).__name__,
            model_config=model_name,
            success=True,
            response_time=response_time,
            response_preview=str(response)[:100] if response else None
        )
    except Exception as e:
        return TestResult(
            capability="chat_completion",
            client_type=type(client).__name__,
            model_config=model_name,
            success=False,
            response_time=time.time() - start_time,
            error=str(e)
        )


async def test_streaming_completion(client, model_name: str) -> TestResult:
    """Тестирование streaming completion с правильной обработкой async generator"""
    start_time = time.time()

    messages = [
        {"role": "user", "content": "Считай от 1 до 3"}
    ]

    try:
        # Проверяем, поддерживает ли клиент streaming
        if not hasattr(client, 'chat_completion_stream'):
            return TestResult(
                capability="streaming",
                client_type=type(client).__name__,
                model_config=model_name,
                success=False,
                response_time=time.time() - start_time,
                error=f"{type(client).__name__} не поддерживает streaming"
            )

        # Получаем async generator
        response_stream = client.chat_completion_stream(
            messages, max_tokens=50)

        # Проверяем, что это действительно async generator
        if not hasattr(response_stream, '__aiter__'):
            return TestResult(
                capability="streaming",
                client_type=type(client).__name__,
                model_config=model_name,
                success=False,
                response_time=time.time() - start_time,
                error="Метод не возвращает async generator"
            )

        # Собираем streaming ответ
        full_response = ""
        chunk_count = 0

        async for chunk in response_stream:
            chunk_count += 1

            # Обработка разных форматов ответов
            if hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0:
                choice = chunk.choices[0]

                # OpenAI формат
                if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                    content = choice.delta.content
                    if content:
                        full_response += content

                # Альтернативный формат
                elif hasattr(choice, 'text'):
                    full_response += choice.text

            # Простой текстовый формат
            elif isinstance(chunk, str):
                full_response += chunk

            # Ограничиваем количество чанков для тестирования
            if chunk_count > 20:
                break

        response_time = time.time() - start_time

        return TestResult(
            capability="streaming",
            client_type=type(client).__name__,
            model_config=model_name,
            success=True,
            response_time=response_time,
            response_preview=f"Streaming: {chunk_count} чанков, {len(full_response)} символов"
        )

    except Exception as e:
        return TestResult(
            capability="streaming",
            client_type=type(client).__name__,
            model_config=model_name,
            success=False,
            response_time=time.time() - start_time,
            error=str(e)
        )


async def test_structured_output_native(client, model_name: str) -> TestResult:
    """Тестирование нативного structured output (OpenAI формат)"""
    start_time = time.time()

    messages = [
        {"role": "user", "content": "Верни JSON с name='test', value=42, description='тестовый объект'"}
    ]

    try:
        # Для StructuredLLMClient используем специальный метод
        if isinstance(client, StructuredLLMClient):
            response = await client.chat_completion_structured(
                messages,
                response_model=TestModel,
                max_tokens=100,
                stream=False  # Нативный режим
            )
        # Для других клиентов с поддержкой structured output
        elif hasattr(client, 'chat_completion_structured'):
            response = await client.chat_completion_structured(
                messages,
                response_model=TestModel,
                max_tokens=100
            )
        else:
            # Для клиентов без structured support - не тестируем
            return TestResult(
                capability="structured_output_native",
                client_type=type(client).__name__,
                model_config=model_name,
                success=False,
                response_time=time.time() - start_time,
                error=f"{type(client).__name__} не поддерживает structured output"
            )

        response_time = time.time() - start_time

        return TestResult(
            capability="structured_output_native",
            client_type=type(client).__name__,
            model_config=model_name,
            success=True,
            response_time=response_time,
            response_preview=f"Native SO: {str(response)[:80]}"
        )

    except Exception as e:
        return TestResult(
            capability="structured_output_native",
            client_type=type(client).__name__,
            model_config=model_name,
            success=False,
            response_time=time.time() - start_time,
            error=str(e)
        )


async def test_structured_output_outlines(client, model_name: str) -> TestResult:
    """Тестирование structured output через Outlines"""
    start_time = time.time()

    messages = [
        {"role": "user", "content": "Верни JSON с name='outlines_test', value=99, description='тест через outlines'"}
    ]

    try:
        # Только для StructuredLLMClient
        if isinstance(client, StructuredLLMClient):
            # Принудительно используем outlines режим
            response = await client._structured_stream_outlines(
                messages,
                response_model=TestModel,
                max_tokens=100
            )
        else:
            return TestResult(
                capability="structured_output_outlines",
                client_type=type(client).__name__,
                model_config=model_name,
                success=False,
                response_time=time.time() - start_time,
                error=f"{type(client).__name__} не поддерживает Outlines режим"
            )

        response_time = time.time() - start_time

        return TestResult(
            capability="structured_output_outlines",
            client_type=type(client).__name__,
            model_config=model_name,
            success=True,
            response_time=response_time,
            response_preview=f"Outlines SO: {str(response)[:80]}"
        )

    except Exception as e:
        return TestResult(
            capability="structured_output_outlines",
            client_type=type(client).__name__,
            model_config=model_name,
            success=False,
            response_time=time.time() - start_time,
            error=str(e)
        )


async def test_function_calling(client, model_name: str) -> TestResult:
    """Тестирование function calling с реальной проверкой вызова функций"""
    start_time = time.time()

    # Простая функция для тестирования
    def get_weather(city: str) -> str:
        return f"Погода в {city}: солнечно, +20°C"

    messages = [
        {"role": "user", "content": "Какая погода в Москве? Используй функцию get_weather."}
    ]

    functions = [{
        "name": "get_weather",
        "description": "Получить информацию о погоде в городе",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "Название города"
                }
            },
            "required": ["city"]
        }
    }]

    try:
        if hasattr(client, 'register_function'):
            client.register_function(
                "get_weather", get_weather, "Получить погоду")

        response = await client.chat_completion(
            messages,
            functions=functions,
            function_call="auto",  # Явно указываем auto
            max_tokens=200
        )
        response_time = time.time() - start_time

        # Проверяем, действительно ли была вызвана функция
        response_str = str(response).lower()
        
        # Более строгая проверка function calling
        has_function_call_structure = any(indicator in str(response) for indicator in [
            '"function_call"',
            '"name": "get_weather"',
            '"arguments"'
        ])
        
        # Проверяем результат выполнения функции
        has_weather_result = any(indicator in response_str for indicator in [
            'солнечно',
            '+20°c',
            '20°c',
            'погода в москве'
        ])
        
        # Проверяем, что это НЕ обычный текстовый ответ
        is_regular_text = any(indicator in response_str for indicator in [
            'к сожалению',
            'не могу использовать',
            'не имею доступа',
            'не могу вызвать',
            'функции недоступны'
        ])

        success_indicators = []
        if has_function_call_structure:
            success_indicators.append("структура function call")
        if has_weather_result:
            success_indicators.append("результат функции")
        if not is_regular_text:
            success_indicators.append("не обычный текст")

        # Считаем успешным, если есть структура function call ИЛИ результат функции, но НЕ обычный текст
        is_real_function_calling = (has_function_call_structure or has_weather_result) and not is_regular_text

        return TestResult(
            capability="function_calling",
            client_type=type(client).__name__,
            model_config=model_name,
            success=is_real_function_calling,
            response_time=response_time,
            response_preview=f"FC {'✓' if is_real_function_calling else '✗'} ({', '.join(success_indicators) if success_indicators else 'обычный ответ'}): {str(response)[:60]}"
        )
    except Exception as e:
        return TestResult(
            capability="function_calling",
            client_type=type(client).__name__,
            model_config=model_name,
            success=False,
            response_time=time.time() - start_time,
            error=str(e)
        )


async def test_embeddings(client, model_name: str) -> TestResult:
    """Тестирование embeddings"""
    start_time = time.time()

    texts = ["Привет мир", "Hello world", "Тестовый текст"]

    try:
        if hasattr(client, 'get_embeddings'):
            response = await client.get_embeddings(texts)
        elif hasattr(client, 'create_embeddings'):
            response = await client.create_embeddings(texts)
        else:
            raise Exception("Embeddings не поддерживаются этим клиентом")

        response_time = time.time() - start_time

        return TestResult(
            capability="embeddings",
            client_type=type(client).__name__,
            model_config=model_name,
            success=True,
            response_time=response_time,
            response_preview=f"Embeddings для {len(texts)} текстов получены"
        )
    except Exception as e:
        return TestResult(
            capability="embeddings",
            client_type=type(client).__name__,
            model_config=model_name,
            success=False,
            response_time=time.time() - start_time,
            error=str(e)
        )


async def test_tool_calling(client, model_name: str) -> TestResult:
    """Тестирование tool calling с реальной проверкой вызова инструментов"""
    start_time = time.time()

    # Простой инструмент для тестирования
    def calculate_sum(a: int, b: int) -> int:
        return a + b

    messages = [
        {"role": "user", "content": "Вычисли сумму 15 и 27 используя инструмент calculate_sum."}
    ]

    tools = [{
        "type": "function",
        "function": {
            "name": "calculate_sum",
            "description": "Вычисляет сумму двух чисел",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "integer",
                        "description": "Первое число"
                    },
                    "b": {
                        "type": "integer",
                        "description": "Второе число"
                    }
                },
                "required": ["a", "b"]
            }
        }
    }]

    try:
        if hasattr(client, 'register_tool'):
            client.register_tool(
                "calculate_sum", calculate_sum, "Вычисляет сумму двух чисел")

        response = await client.chat_completion(
            messages,
            tools=tools,
            tool_choice="auto",  # Явно указываем auto
            max_tokens=200
        )
        response_time = time.time() - start_time

        # Проверяем, действительно ли был вызван инструмент
        response_str = str(response).lower()
        
        # Более строгая проверка tool calling
        has_tool_call_structure = any(indicator in str(response) for indicator in [
            '"tool_calls"',
            '"type": "function"',
            '"name": "calculate_sum"',
            '"arguments"'
        ])

        # Проверяем результат выполнения инструмента
        has_calculation_result = '42' in response_str  # Правильный ответ 15 + 27 = 42
        
        # Проверяем, что это НЕ обычный текстовый ответ
        is_regular_text = any(indicator in response_str for indicator in [
            'к сожалению',
            'не могу использовать',
            'не имею доступа',
            'не могу вызвать',
            'инструменты недоступны'
        ])

        success_indicators = []
        if has_tool_call_structure:
            success_indicators.append("структура tool call")
        if has_calculation_result:
            success_indicators.append("правильный результат")
        if not is_regular_text:
            success_indicators.append("не обычный текст")

        # Считаем успешным, если есть структура tool call ИЛИ правильный результат, но НЕ обычный текст
        is_real_tool_calling = (has_tool_call_structure or has_calculation_result) and not is_regular_text

        return TestResult(
            capability="tool_calling",
            client_type=type(client).__name__,
            model_config=model_name,
            success=is_real_tool_calling,
            response_time=response_time,
            response_preview=f"TC {'✓' if is_real_tool_calling else '✗'} ({', '.join(success_indicators) if success_indicators else 'обычный ответ'}): {str(response)[:60]}"
        )
    except Exception as e:
        return TestResult(
            capability="tool_calling",
            client_type=type(client).__name__,
            model_config=model_name,
            success=False,
            response_time=time.time() - start_time,
            error=str(e)
        )


async def test_reasoning_native_thinking(client, model_name: str) -> TestResult:
    """Тестирование нативного reasoning через встроенные thinking токены"""
    start_time = time.time()

    messages = [
        {"role": "user", "content": "Реши задачу пошагово: если у меня есть 10 яблок и я съел 3, сколько осталось?"}
    ]

    try:
        if hasattr(client, 'reasoning_completion'):
            # Пытаемся использовать нативный thinking режим
            response = await client.reasoning_completion(
                messages,
                problem_type="math",
                max_tokens=200,
                enable_streaming=False
            )

            # Проверяем, есть ли thinking блоки в ответе
            has_thinking = False
            has_steps = False
            has_correct_answer = False
            
            response_str = str(response)
            
            # Проверяем наличие thinking структур
            if hasattr(response, 'steps') and response.steps:
                has_thinking = True
                has_steps = True
            elif '<thinking>' in response_str or 'думаю' in response_str.lower():
                has_thinking = True
                
            # Проверяем правильность ответа (10 - 3 = 7)
            if '7' in response_str and any(word in response_str.lower() for word in ['остал', 'получ', 'равн', 'ответ']):
                has_correct_answer = True

        else:
            # Для обычных клиентов пробуем через промпт
            thinking_messages = messages + [
                {"role": "system", "content": "Думай пошагово, используй формат: <thinking>твои размышления</thinking>"}
            ]
            response = await client.chat_completion(thinking_messages, max_tokens=200)
            response_str = str(response)
            
            has_thinking = '<thinking>' in response_str if response else False
            has_steps = any(word in response_str.lower() for word in ['шаг', 'сначала', 'затем', 'потом'])
            has_correct_answer = '7' in response_str and any(word in response_str.lower() for word in ['остал', 'получ', 'равн', 'ответ'])

        # Подсчитываем качество reasoning
        reasoning_indicators = []
        if has_thinking:
            reasoning_indicators.append("thinking")
        if has_steps:
            reasoning_indicators.append("пошаговость")
        if has_correct_answer:
            reasoning_indicators.append("правильный ответ")
            
        reasoning_quality = len(reasoning_indicators)
        is_real_reasoning = reasoning_quality >= 2  # Минимум 2 индикатора для "настоящего" reasoning

        response_time = time.time() - start_time

        return TestResult(
            capability="reasoning_native_thinking",
            client_type=type(client).__name__,
            model_config=model_name,
            success=is_real_reasoning,  # Успех только если реально работает
            response_time=response_time,
            response_preview=f"Native thinking ({'✓' if is_real_reasoning else '✗'}) {reasoning_quality}/3 ({', '.join(reasoning_indicators)}): {response_str[:60]}"
        )
    except Exception as e:
        return TestResult(
            capability="reasoning_native_thinking",
            client_type=type(client).__name__,
            model_config=model_name,
            success=False,
            response_time=time.time() - start_time,
            error=str(e)
        )


async def test_reasoning_cot(client, model_name: str) -> TestResult:
    """Тестирование Chain-of-Thought reasoning"""
    start_time = time.time()

    messages = [
        {"role": "system", "content": "Ты эксперт по решению математических задач. Всегда решай задачи пошагово, объясняя каждый шаг."},
        {"role": "user", "content": "Реши задачу пошагово: В магазине было 50 яблок. Утром продали 15, днем еще 12. Сколько яблок осталось?"}
    ]

    try:
        if hasattr(client, 'reasoning_completion'):
            response = await client.reasoning_completion(
                messages,
                problem_type="math",
                max_tokens=200
            )
        else:
            response = await client.chat_completion(messages, max_tokens=200)

        response_time = time.time() - start_time

        # Проверяем качество CoT рассуждений
        response_str = str(response)
        has_steps = any(word in response_str.lower() for word in [
                        'шаг', 'сначала', 'затем', 'итак', 'следовательно'])
        has_calculation = any(char in response_str for char in [
                              '=', '+', '-', '*', '/'])

        quality_score = sum([has_steps, has_calculation])

        return TestResult(
            capability="reasoning_cot",
            client_type=type(client).__name__,
            model_config=model_name,
            success=True,
            response_time=response_time,
            response_preview=f"CoT quality {quality_score}/2: {response_str[:80]}"
        )
    except Exception as e:
        return TestResult(
            capability="reasoning_cot",
            client_type=type(client).__name__,
            model_config=model_name,
            success=False,
            response_time=time.time() - start_time,
            error=str(e)
        )


def get_suitable_tests_for_client_and_model(client_name: str, model_config: ModelConfig) -> List[tuple]:
    """Получить подходящие тесты для конкретного типа клиента и модели"""

    # Для embedding моделей
    if model_config.model_type == "embedding":
        if client_name == "embeddings":
            return [("embeddings", test_embeddings)]
        else:
            # Другие клиенты не должны тестироваться на embedding моделях
            return []

    # Для chat и completion моделей
    base_tests = [
        ("chat_completion", test_basic_chat_completion),
        ("streaming", test_streaming_completion),
        ("function_calling", test_function_calling),
        ("tool_calling", test_tool_calling),
        ("reasoning_native_thinking", test_reasoning_native_thinking),
        ("reasoning_cot", test_reasoning_cot),
    ]

    # Специфичные тесты для разных клиентов
    if client_name == "structured":
        # StructuredLLMClient - тестируем оба режима SO
        return [
            ("structured_output_native", test_structured_output_native),
            ("structured_output_outlines", test_structured_output_outlines)
        ]

    elif client_name == "embeddings":
        # EmbeddingsLLMClient не должен тестироваться на chat/completion моделях
        return []

    elif client_name == "completion":
        # CompletionLLMClient - только для completion моделей, не для chat
        if model_config.model_type == "completion":
            return [
                ("chat_completion", test_basic_chat_completion),
                ("streaming", test_streaming_completion),
                ("structured_output_native", test_structured_output_native),
            ]
        else:
            # Не тестируем CompletionLLMClient на chat моделях
            return []

    elif client_name == "multimodal":
        # MultimodalLLMClient не поддерживает streaming
        return [
            ("chat_completion", test_basic_chat_completion),
            ("function_calling", test_function_calling),
            ("tool_calling", test_tool_calling),
            ("reasoning_native_thinking", test_reasoning_native_thinking),
            ("reasoning_cot", test_reasoning_cot),
        ]
    
    elif client_name in ["standard", "streaming", "reasoning"]:
        # Эти клиенты поддерживают базовые операции
        return base_tests

    return base_tests


async def test_adaptive_capabilities(model_config: ModelConfig) -> List[TestResult]:
    """Тестирование AdaptiveLLMClient возможностей для конкретной модели"""
    print(
        f"\n=== Тестирование {model_config.name} ({model_config.description}) ===")

    config = LLMConfig(
        endpoint=model_config.endpoint,
        api_key=model_config.token,
        model=model_config.model,
        temperature=0.7
    )

    adaptive_config = AdaptiveConfig(
        capability_detection_timeout=5.0,
        enable_performance_tracking=True
    )

    results = []

    # Для embedding моделей не тестируем AdaptiveLLMClient
    if model_config.model_type == "embedding":
        print("   AdaptiveLLMClient не подходит для embedding моделей")
        return results

    try:
        async with AdaptiveLLMClient(config, adaptive_config) as client:
            print("1. Определение возможностей модели:")

            # Принудительно обновляем возможности
            capabilities = await client.get_model_capabilities(force_refresh=True)
            print(
                f"   Обнаруженные возможности: {[cap.value for cap in capabilities]}")

            # Получаем детальный отчет
            capabilities_dict = await client.detect_model_capabilities()

            print("2. Детальный отчет о возможностях:")
            for capability, supported in capabilities_dict.items():
                status = "✓" if supported else "✗"
                print(f"   {status} {capability}: {supported}")

            # Тестируем базовые возможности
            print("3. Тестирование базовых возможностей:")

            # Chat completion
            result = await test_basic_chat_completion(client, model_config.name)
            results.append(result)
            print(
                f"   Chat completion: {'✓' if result.success else '✗'} ({result.response_time:.2f}s)")

            # Smart completion
            messages = [
                {"role": "user", "content": "Привет! Расскажи короткую шутку."}]
            try:
                response = await client.smart_completion(messages, max_tokens=100)
                print(
                    f"   Smart completion: ✓ ({len(str(response))} символов)")
            except Exception as e:
                print(f"   Smart completion: ✗ ({str(e)[:50]}...)")
            
            # Реальная проверка возможностей (независимо от detect_model_capabilities)
            print("4. Реальная проверка возможностей:")
            
            # Всегда проверяем function calling
            fc_result = await test_function_calling(client, model_config.name)
            results.append(fc_result)
            print(f"   Function calling: {'✓' if fc_result.success else '✗'} - {fc_result.response_preview[:80]}")
            
            # Всегда проверяем tool calling
            tc_result = await test_tool_calling(client, model_config.name)
            results.append(tc_result)
            print(f"   Tool calling: {'✓' if tc_result.success else '✗'} - {tc_result.response_preview[:80]}")
            
            # Всегда проверяем reasoning
            reasoning_result = await test_reasoning_native_thinking(client, model_config.name)
            results.append(reasoning_result)
            print(f"   Reasoning: {'✓' if reasoning_result.success else '✗'} - {reasoning_result.response_preview[:80]}")
            
            # Всегда проверяем structured output
            so_result = await test_structured_output_native(client, model_config.name)
            results.append(so_result)
            print(f"   Structured Output: {'✓' if so_result.success else '✗'} - {so_result.response_preview[:80] if so_result.response_preview else so_result.error[:80]}")
            
            # Обновляем capabilities_dict на основе реальных результатов
            print("5. Обновленные возможности на основе реальных тестов:")
            real_capabilities = {
                'function_calling': fc_result.success,
                'tool_calling': tc_result.success,
                'reasoning': reasoning_result.success,
                'structured_output': so_result.success,
                'streaming': capabilities_dict.get('streaming', False),
                'chat_completion': capabilities_dict.get('chat_completion', True),  # Обычно работает
            }
            
            for capability, supported in real_capabilities.items():
                status = "✓" if supported else "✗"
                print(f"   {status} {capability}: {supported}")
                
            # Обновляем кэш возможностей AdaptiveLLMClient на основе реальных результатов
            if hasattr(client, '_model_info') and client._model_info:
                from kraken_llm.client.adaptive import ModelCapability
                updated_capabilities = set()
                
                if real_capabilities['chat_completion']:
                    updated_capabilities.add(ModelCapability.CHAT_COMPLETION)
                if real_capabilities['streaming']:
                    updated_capabilities.add(ModelCapability.STREAMING)
                if real_capabilities['function_calling']:
                    updated_capabilities.add(ModelCapability.FUNCTION_CALLING)
                if real_capabilities['tool_calling']:
                    updated_capabilities.add(ModelCapability.TOOL_CALLING)
                if real_capabilities['structured_output']:
                    updated_capabilities.add(ModelCapability.STRUCTURED_OUTPUT)
                if real_capabilities['reasoning']:
                    updated_capabilities.add(ModelCapability.REASONING)
                    
                client._model_info.capabilities = updated_capabilities
                print(f"   Обновлен кэш AdaptiveLLMClient: {[cap.value for cap in updated_capabilities]}")

            # Получаем отчет о производительности
            performance_report = client.get_performance_report()

            if performance_report["model_info"]:
                model_info = performance_report["model_info"]
                print(f"   Модель: {model_info['name']}")
                print(f"   Провайдер: {model_info['provider']}")
                print(f"   Возможности: {len(model_info['capabilities'])}")

    except Exception as e:
        print(f"   Ошибка тестирования AdaptiveLLMClient: {e}")
        results.append(TestResult(
            capability="adaptive_client",
            client_type="AdaptiveLLMClient",
            model_config=model_config.name,
            success=False,
            response_time=0.0,
            error=str(e)
        ))

    return results


async def test_all_client_types(model_config: ModelConfig) -> List[TestResult]:
    """Тестирование всех типов клиентов для конкретной модели"""
    print(f"\n=== Тестирование всех клиентов для {model_config.name} ===")

    config = LLMConfig(
        endpoint=model_config.endpoint,
        api_key=model_config.token,
        model=model_config.model,
        temperature=0.7
    )

    all_results = []

    # Список клиентов для тестирования
    client_types = [
        ("standard", StandardLLMClient),
        ("streaming", StreamingLLMClient),
        ("structured", StructuredLLMClient),
        ("reasoning", ReasoningLLMClient),
        ("multimodal", MultimodalLLMClient),
        ("embeddings", EmbeddingsLLMClient),
        ("completion", CompletionLLMClient),
    ]

    for client_name, client_class in client_types:
        print(f"\n--- Тестирование {client_name} клиента ---")

        # Получаем подходящие тесты для этого клиента и модели
        suitable_tests = get_suitable_tests_for_client_and_model(
            client_name, model_config)

        if not suitable_tests:
            print(
                f"   Нет подходящих тестов для {client_name} клиента с моделью {model_config.name}")
            continue

        try:
            async with client_class(config) as client:

                for test_name, test_func in suitable_tests:
                    try:
                        result = await test_func(client, model_config.name)
                        all_results.append(result)

                        status = "✓" if result.success else "✗"
                        time_str = f"{result.response_time:.2f}s"

                        if result.success:
                            preview = result.response_preview[:30] + "..." if result.response_preview and len(
                                result.response_preview) > 30 else result.response_preview
                            print(
                                f"   {status} {test_name}: {time_str} - {preview}")
                        else:
                            error_preview = result.error[:50] + "..." if result.error and len(
                                result.error) > 50 else result.error
                            print(
                                f"   {status} {test_name}: {time_str} - {error_preview}")

                    except Exception as e:
                        print(
                            f"   ✗ {test_name}: Ошибка теста - {str(e)[:50]}...")
                        all_results.append(TestResult(
                            capability=test_name,
                            client_type=client_name,
                            model_config=model_config.name,
                            success=False,
                            response_time=0.0,
                            error=str(e)
                        ))

        except Exception as e:
            print(
                f"   Не удалось создать {client_name} клиент: {str(e)[:100]}...")
            all_results.append(TestResult(
                capability="client_creation",
                client_type=client_name,
                model_config=model_config.name,
                success=False,
                response_time=0.0,
                error=str(e)
            ))

    return all_results


async def test_client_factory(model_config: ModelConfig) -> List[TestResult]:
    """Тестирование ClientFactory"""
    print(f"\n=== Тестирование ClientFactory для {model_config.name} ===")

    config = LLMConfig(
        endpoint=model_config.endpoint,
        api_key=model_config.token,
        model=model_config.model,
        temperature=0.7
    )

    results = []

    # Тестируем создание разных типов клиентов через фабрику
    factory_methods = [
        ("adaptive", ClientFactory.create_adaptive_client),
        ("standard", ClientFactory.create_standard_client),
        ("streaming", ClientFactory.create_streaming_client),
        ("structured", ClientFactory.create_structured_client),
        ("reasoning", ClientFactory.create_reasoning_client),
        ("multimodal", ClientFactory.create_multimodal_client),
        ("embeddings", ClientFactory.create_embeddings_client),
    ]

    for method_name, factory_method in factory_methods:
        start_time = time.time()

        # Для embedding моделей тестируем только embeddings клиент
        if model_config.model_type == "embedding" and method_name != "embeddings":
            continue

        # Для chat/completion моделей не тестируем embeddings клиент
        if model_config.model_type != "embedding" and method_name == "embeddings":
            continue

        try:
            client = factory_method(config)

            # Простой тест в зависимости от типа модели и клиента
            if model_config.model_type == "embedding" and method_name == "embeddings":
                # Тест embeddings
                texts = ["Тест"]
                response = await client.get_embeddings(texts)

                results.append(TestResult(
                    capability="factory_creation",
                    client_type=f"Factory_{method_name}",
                    model_config=model_config.name,
                    success=True,
                    response_time=time.time() - start_time,
                    response_preview="Embeddings клиент создан и работает"
                ))
                print(f"   ✓ {method_name}: Создан и протестирован")

            elif method_name == "structured":
                # Для structured клиента нужен response_model
                messages = [
                    {"role": "user", "content": "Верни JSON с полями: name='test', value=1, description='тестовый объект'"}]
                response = await client.chat_completion_structured(
                    messages,
                    response_model=TestModel,
                    max_tokens=100  # Увеличиваем лимит для полного ответа
                )

                results.append(TestResult(
                    capability="factory_creation",
                    client_type=f"Factory_{method_name}",
                    model_config=model_config.name,
                    success=True,
                    response_time=time.time() - start_time,
                    response_preview="Structured клиент создан и работает"
                ))
                print(f"   ✓ {method_name}: Создан и протестирован")

            elif hasattr(client, 'chat_completion'):
                # Тест chat completion
                messages = [{"role": "user", "content": "Тест"}]
                response = await client.chat_completion(messages, max_tokens=10)

                results.append(TestResult(
                    capability="factory_creation",
                    client_type=f"Factory_{method_name}",
                    model_config=model_config.name,
                    success=True,
                    response_time=time.time() - start_time,
                    response_preview="Клиент создан и работает"
                ))
                print(f"   ✓ {method_name}: Создан и протестирован")
            else:
                results.append(TestResult(
                    capability="factory_creation",
                    client_type=f"Factory_{method_name}",
                    model_config=model_config.name,
                    success=False,
                    response_time=time.time() - start_time,
                    error="Нет подходящего метода для тестирования"
                ))
                print(
                    f"   ✗ {method_name}: Нет подходящего метода для тестирования")

            await client.close()

        except Exception as e:
            results.append(TestResult(
                capability="factory_creation",
                client_type=f"Factory_{method_name}",
                model_config=model_config.name,
                success=False,
                response_time=time.time() - start_time,
                error=str(e)
            ))
            print(f"   ✗ {method_name}: {str(e)[:50]}...")

    return results


def print_summary_report(all_results: List[TestResult]):
    """Печать итогового отчета"""
    print("\n" + "="*80)
    print("ИТОГОВЫЙ ОТЧЕТ ТЕСТИРОВАНИЯ")
    print("="*80)

    # Группируем результаты по моделям
    by_model = {}
    for result in all_results:
        if result.model_config not in by_model:
            by_model[result.model_config] = []
        by_model[result.model_config].append(result)

    for model_name, model_results in by_model.items():
        print(f"\nМодель: {model_name}")
        print("-" * 50)

        # Статистика по возможностям
        capabilities_stats = {}
        for result in model_results:
            if result.capability not in capabilities_stats:
                capabilities_stats[result.capability] = {
                    "success": 0, "total": 0}
            capabilities_stats[result.capability]["total"] += 1
            if result.success:
                capabilities_stats[result.capability]["success"] += 1

        print("Возможности:")
        for capability, stats in capabilities_stats.items():
            success_rate = stats["success"] / stats["total"] * 100
            status = "✓" if success_rate > 50 else "⚠" if success_rate > 0 else "✗"
            print(
                f"  {status} {capability}: {stats['success']}/{stats['total']} ({success_rate:.0f}%)")

        # Статистика по клиентам
        client_stats = {}
        for result in model_results:
            if result.client_type not in client_stats:
                client_stats[result.client_type] = {"success": 0, "total": 0}
            client_stats[result.client_type]["total"] += 1
            if result.success:
                client_stats[result.client_type]["success"] += 1

        print("\nКлиенты:")
        for client_type, stats in client_stats.items():
            success_rate = stats["success"] / stats["total"] * 100
            status = "✓" if success_rate > 50 else "⚠" if success_rate > 0 else "✗"
            print(
                f"  {status} {client_type}: {stats['success']}/{stats['total']} ({success_rate:.0f}%)")

    # Общая статистика
    total_tests = len(all_results)
    successful_tests = sum(1 for r in all_results if r.success)
    success_rate = successful_tests / total_tests * 100 if total_tests > 0 else 0

    print(f"\nОБЩАЯ СТАТИСТИКА:")
    print(f"   Всего тестов: {total_tests}")
    print(f"   Успешных: {successful_tests}")
    print(f"   Процент успеха: {success_rate:.1f}%")

    # Топ ошибок
    errors = [r.error for r in all_results if r.error]
    if errors:
        print(f"\nЧастые ошибки:")
        error_counts = {}
        for error in errors:
            short_error = error[:100] + "..." if len(error) > 100 else error
            error_counts[short_error] = error_counts.get(short_error, 0) + 1

        for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   {count}x: {error}")


def generate_recommendations(all_results: List[TestResult]) -> Dict[str, List[str]]:
    """Динамическая генерация рекомендаций на основе результатов тестирования"""
    recommendations = {}

    # Анализируем результаты по моделям
    by_model = {}
    for result in all_results:
        if result.model_config not in by_model:
            by_model[result.model_config] = []
        by_model[result.model_config].append(result)

    # Анализируем частые ошибки
    error_counts = {}
    for result in all_results:
        if result.error:
            # Сокращаем ошибку для группировки
            short_error = result.error[:100]
            error_counts[short_error] = error_counts.get(short_error, 0) + 1

    for model_name, model_results in by_model.items():
        model_recs = []

        # Анализируем успешные и неуспешные клиенты
        client_stats = {}
        capability_stats = {}

        for result in model_results:
            # Статистика по клиентам
            if result.client_type not in client_stats:
                client_stats[result.client_type] = {
                    "success": 0, "total": 0, "errors": []}
            client_stats[result.client_type]["total"] += 1
            if result.success:
                client_stats[result.client_type]["success"] += 1
            else:
                client_stats[result.client_type]["errors"].append(result.error)

            # Статистика по возможностям
            if result.capability not in capability_stats:
                capability_stats[result.capability] = {
                    "success": 0, "total": 0, "errors": []}
            capability_stats[result.capability]["total"] += 1
            if result.success:
                capability_stats[result.capability]["success"] += 1
            else:
                capability_stats[result.capability]["errors"].append(
                    result.error)

        # Генерируем рекомендации на основе статистики

        # Рекомендуемые клиенты (успешность > 75%)
        excellent_clients = [name for name, stats in client_stats.items()
                             if stats["success"] / stats["total"] > 0.75 and stats["total"] > 1]

        for client in excellent_clients:
            success_rate = client_stats[client]["success"] / \
                client_stats[client]["total"]
            model_recs.append(
                f"{client} показывает ({success_rate:.0%})")

        # Проблемные клиенты (успешность < 50%)
        problematic_clients = [name for name, stats in client_stats.items()
                               if stats["success"] / stats["total"] < 0.5 and stats["total"] > 1]

        for client in problematic_clients:
            success_rate = client_stats[client]["success"] / \
                client_stats[client]["total"]
            common_errors = set(client_stats[client]["errors"])
            if common_errors:
                main_error = list(common_errors)[
                    0][:50] + "..." if len(list(common_errors)[0]) > 50 else list(common_errors)[0]
                model_recs.append(
                    f"Избегайте {client} ({success_rate:.0%} успеха): {main_error}")

        # Анализ возможностей
        working_capabilities = [cap for cap, stats in capability_stats.items()
                                if stats["success"] / stats["total"] > 0.5]

        if working_capabilities:
            model_recs.append(
                f"Поддерживаемые возможности: {', '.join(working_capabilities)}")

        # Проблемные возможности
        broken_capabilities = [cap for cap, stats in capability_stats.items()
                               if stats["success"] / stats["total"] == 0 and stats["total"] > 1]

        if broken_capabilities:
            model_recs.append(
                f"Не работают: {', '.join(broken_capabilities)}")

        # Специфичные рекомендации для типов моделей
        if model_name.endswith("_model"):
            model_type = model_name.replace("_model", "")
            if model_type == "embedding":
                model_recs.append(
                    "Специализированная модель только для векторных представлений")
            elif model_type == "chat":
                if any("structured_output" in cap for cap in working_capabilities):
                    model_recs.append(
                        "Поддерживает структурированные ответы")
                if any("streaming" in cap for cap in working_capabilities):
                    model_recs.append("🌊 Поддерживает потоковые запросы")
            elif model_type == "completion":
                model_recs.append(
                    "Completion API - может требовать специальной обработки")

        recommendations[model_name] = model_recs

    # Общие рекомендации на основе частых ошибок
    general_recs = []

    for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
        if count > 1:
            if "streaming" in error.lower():
                general_recs.append(
                    "Для streaming используйте StreamingLLMClient")
            elif "async generator" in error.lower():
                general_recs.append(
                    "Проблемы с async generators - проверьте реализацию")
            elif "html" in error.lower() or "404" in error:
                general_recs.append("🔗 Проверьте правильность URL endpoints")
            elif "response_model" in error.lower():
                general_recs.append(
                    "Для structured output всегда указывайте response_model")

    recommendations["general"] = general_recs

    return recommendations


def print_recommendations(recommendations: Dict[str, List[str]]):
    """Печать рекомендаций"""
    print("\n" + "="*80)
    print("РЕКОМЕНДАЦИИ ПО ИСПОЛЬЗОВАНИЮ")
    print("="*80)

    for model_name, recs in recommendations.items():
        if recs:
            print(f"\n  {model_name.upper()}:")
            for rec in recs:
                print(f"   {rec}")

    # Общие рекомендации
    print(f"\nОБЩИЕ РЕКОМЕНДАЦИИ:")
    print(f"   Используйте специализированные клиенты для конкретных задач")
    print(f"   Для StructuredLLMClient всегда указывайте response_model")
    print(f"   Обратите внимание на ошибки 404 - проверьте правильность endpoint'ов")


def save_results_to_json(all_results: List[TestResult], filename: str = "model_capabilities.json"):
    """Сохранение результатов в JSON файл"""
    results_data = []
    for result in all_results:
        results_data.append({
            "capability": result.capability,
            "client_type": result.client_type,
            "model_config": result.model_config,
            "success": result.success,
            "response_time": result.response_time,
            "error": result.error,
            "response_preview": result.response_preview
        })

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)

    print(f"\nРезультаты сохранены в {filename}")


async def main():
    """Основная функция тестирования"""
    print("\nТЕСТИРОВАНИЕ ВОЗМОЖНОСТЕЙ МОДЕЛЕЙ")

    # Получаем конфигурации моделей
    model_configs = get_model_configs()

    if not model_configs:
        print("❌ Не найдено конфигураций моделей в .env файле")
        return

    print(f"\nНайдено {len(model_configs)} конфигураций моделей:")
    for config in model_configs:
        print(f"• {config.name}: {config.description}")
        print(f"  Endpoint: {config.endpoint}")
        print(f"  Model: {config.model}")

    all_results = []

    # Тестируем каждую модель
    for model_config in model_configs:
        print(f"\nНачинаем тестирование модели: {model_config.name}")

        # Тестируем AdaptiveLLMClient (только для chat/completion моделей)
        adaptive_results = await test_adaptive_capabilities(model_config)
        all_results.extend(adaptive_results)

        # Тестируем все типы клиентов
        client_results = await test_all_client_types(model_config)
        all_results.extend(client_results)

        # Тестируем ClientFactory
        factory_results = await test_client_factory(model_config)
        all_results.extend(factory_results)

        print(f"✅ Завершено тестирование модели: {model_config.name}")

    # Печатаем итоговый отчет
    print_summary_report(all_results)

    # Генерируем и печатаем рекомендации
    recommendations = generate_recommendations(all_results)
    print_recommendations(recommendations)

    # Сохраняем результаты
    save_results_to_json(all_results)

if __name__ == "__main__":
    asyncio.run(main())
