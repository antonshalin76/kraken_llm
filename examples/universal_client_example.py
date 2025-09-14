#!/usr/bin/env python3
"""
Пример использования UniversalLLMClient

Демонстрирует различные способы создания и использования
универсального клиента Kraken LLM, который объединяет
все возможности в едином интерфейсе.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel

# Добавляем путь к модулям Kraken LLM
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kraken_llm.config.settings import LLMConfig
from kraken_llm.client.universal import (
    UniversalLLMClient,
    UniversalClientConfig,
    UniversalCapability,
    create_universal_client,
    create_universal_client_from_report,
    create_basic_client,
    create_advanced_client,
    create_full_client
)
from kraken_llm.tools import register_function, register_tool


# Модели для structured output
class TaskResponse(BaseModel):
    task: str
    priority: int
    estimated_time: str
    tags: List[str] = []


class WeatherInfo(BaseModel):
    city: str
    temperature: int
    condition: str
    humidity: int


# Функции для демонстрации function/tool calling
@register_function(description="Получить информацию о погоде")
def get_weather(city: str) -> str:
    """Получить информацию о погоде в городе"""
    return f"В городе {city}: солнечно, +22°C, влажность 65%"


@register_tool(description="Вычислить сумму чисел")
def calculate_sum(a: int, b: int) -> int:
    """Вычислить сумму двух чисел"""
    return a + b


async def demo_basic_usage():
    """Демонстрация базового использования"""
    print("🔧 Демонстрация базового использования UniversalLLMClient")
    print("=" * 60)
    
    # Создание базового клиента
    async with create_basic_client() as client:
        print(f"Доступные возможности: {client.get_available_capabilities()}")
        print(f"Активные клиенты: {client.get_active_clients()}")
        
        # Базовый chat completion
        response = await client.chat_completion([
            {"role": "user", "content": "Привет! Как дела?"}
        ], max_tokens=50)
        
        print(f"Ответ: {response}")
        
        # Streaming (если доступен)
        if UniversalCapability.STREAMING in client.universal_config.capabilities:
            print("\nStreaming ответ:")
            async for chunk in client.chat_completion_stream([
                {"role": "user", "content": "Считай от 1 до 5"}
            ]):
                print(chunk, end="", flush=True)
            print()


async def demo_advanced_usage():
    """Демонстрация продвинутого использования"""
    print("\n🚀 Демонстрация продвинутого использования")
    print("=" * 60)
    
    # Создание продвинутого клиента
    async with create_advanced_client() as client:
        print(f"Доступные возможности: {client.get_available_capabilities()}")
        
        # Регистрируем функции и инструменты
        client.register_function("get_weather", get_weather, "Получить погоду")
        client.register_tool("calculate_sum", calculate_sum, "Вычислить сумму")
        
        # Structured output
        if UniversalCapability.STRUCTURED_OUTPUT in client.universal_config.capabilities:
            try:
                task = await client.chat_completion_structured([
                    {"role": "user", "content": "Создай задачу: написать отчет по проекту"}
                ], response_model=TaskResponse)
                
                print(f"Структурированная задача: {task}")
            except Exception as e:
                print(f"Structured output недоступен: {e}")
        
        # Function calling
        if UniversalCapability.FUNCTION_CALLING in client.universal_config.capabilities:
            try:
                response = await client.chat_completion([
                    {"role": "user", "content": "Какая погода в Москве?"}
                ], functions=[{
                    "name": "get_weather",
                    "description": "Получить погоду",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"]
                    }
                }])
                
                print(f"Function calling ответ: {response}")
            except Exception as e:
                print(f"Function calling недоступен: {e}")
        
        # Reasoning
        if UniversalCapability.REASONING in client.universal_config.capabilities:
            try:
                response = await client.reasoning_completion([
                    {"role": "user", "content": "Реши задачу: 15 + 27 = ?"}
                ], problem_type="math")
                
                print(f"Reasoning ответ: {response}")
            except Exception as e:
                print(f"Reasoning недоступен: {e}")


async def demo_custom_configuration():
    """Демонстрация кастомной конфигурации"""
    print("\n⚙️ Демонстрация кастомной конфигурации")
    print("=" * 60)
    
    # Создание клиента с кастомными возможностями
    custom_capabilities = {
        UniversalCapability.CHAT_COMPLETION,
        UniversalCapability.STREAMING,
        UniversalCapability.STRUCTURED_OUTPUT,
        UniversalCapability.EMBEDDINGS,
    }
    
    config = LLMConfig()
    universal_config = UniversalClientConfig(
        capabilities=custom_capabilities,
        prefer_streaming=True,
        auto_fallback=True
    )
    
    async with UniversalLLMClient(config, universal_config) as client:
        print(f"Кастомные возможности: {client.get_available_capabilities()}")
        
        # Тестируем возможности
        test_results = await client.test_capabilities()
        print("Результаты тестирования:")
        for capability, result in test_results.items():
            status = "✅" if result else "❌"
            print(f"  {status} {capability}")
        
        # Embeddings (если доступны)
        if UniversalCapability.EMBEDDINGS in client.universal_config.capabilities:
            try:
                embeddings = await client.get_embeddings([
                    "Первый текст для векторизации",
                    "Второй текст для векторизации"
                ])
                print(f"Embeddings получены: {type(embeddings)}")
            except Exception as e:
                print(f"Embeddings недоступны: {e}")


async def demo_from_capabilities_report():
    """Демонстрация создания клиента из отчета анализатора"""
    print("\n📊 Демонстрация создания из отчета анализатора")
    print("=" * 60)
    
    # Симулируем отчет анализатора возможностей
    mock_report = {
        "model_summaries": {
            "test_model": {
                "confirmed_capabilities": [
                    {"capability": "chat_completion", "success_rate": 1.0},
                    {"capability": "streaming", "success_rate": 0.9},
                    {"capability": "structured_output_native", "success_rate": 0.8},
                    {"capability": "function_calling", "success_rate": 0.7},
                ]
            }
        }
    }
    
    # Создаем клиент на основе отчета
    async with create_universal_client_from_report(mock_report, model_name="test_model") as client:
        print(f"Возможности из отчета: {client.get_available_capabilities()}")
        
        # Информация о клиенте
        info = client.get_client_info()
        print("Информация о клиенте:")
        for key, value in info.items():
            print(f"  {key}: {value}")


async def demo_convenience_functions():
    """Демонстрация удобных функций создания"""
    print("\n🎯 Демонстрация удобных функций создания")
    print("=" * 60)
    
    # Базовый клиент
    print("Базовый клиент:")
    async with create_basic_client() as client:
        capabilities = client.get_available_capabilities()
        print(f"  Возможности: {capabilities}")
    
    # Продвинутый клиент
    print("\nПродвинутый клиент:")
    async with create_advanced_client() as client:
        capabilities = client.get_available_capabilities()
        print(f"  Возможности: {capabilities}")
    
    # Полнофункциональный клиент
    print("\nПолнофункциональный клиент:")
    try:
        async with create_full_client() as client:
            capabilities = client.get_available_capabilities()
            print(f"  Возможности: {capabilities}")
    except Exception as e:
        print(f"  Не удалось создать полнофункциональный клиент: {e}")
    
    # Кастомный клиент через create_universal_client
    print("\nКастомный клиент:")
    custom_caps = {
        UniversalCapability.CHAT_COMPLETION,
        UniversalCapability.STRUCTURED_OUTPUT
    }
    
    async with create_universal_client(capabilities=custom_caps) as client:
        capabilities = client.get_available_capabilities()
        print(f"  Возможности: {capabilities}")


async def demo_error_handling():
    """Демонстрация обработки ошибок"""
    print("\n⚠️ Демонстрация обработки ошибок")
    print("=" * 60)
    
    # Клиент только с базовыми возможностями
    basic_config = UniversalClientConfig(
        capabilities={UniversalCapability.CHAT_COMPLETION},
        auto_fallback=False
    )
    
    async with UniversalLLMClient(LLMConfig(), basic_config) as client:
        # Попытка использовать недоступную возможность
        try:
            await client.chat_completion_structured([
                {"role": "user", "content": "Тест"}
            ], response_model=TaskResponse)
        except Exception as e:
            print(f"Ожидаемая ошибка structured output: {e}")
        
        # Попытка использовать embeddings
        try:
            await client.get_embeddings(["тест"])
        except Exception as e:
            print(f"Ожидаемая ошибка embeddings: {e}")
        
        # Базовый chat completion должен работать
        try:
            response = await client.chat_completion([
                {"role": "user", "content": "Привет!"}
            ], max_tokens=10)
            print(f"Базовый chat completion работает: {bool(response)}")
        except Exception as e:
            print(f"Неожиданная ошибка chat completion: {e}")


async def demo_performance_comparison():
    """Демонстрация сравнения производительности"""
    print("\n⚡ Демонстрация сравнения производительности")
    print("=" * 60)
    
    import time
    
    messages = [{"role": "user", "content": "Напиши короткое стихотворение"}]
    
    # Тест с базовым клиентом
    start_time = time.time()
    async with create_basic_client() as client:
        response1 = await client.chat_completion(messages, max_tokens=50)
    basic_time = time.time() - start_time
    
    # Тест с продвинутым клиентом
    start_time = time.time()
    async with create_advanced_client() as client:
        response2 = await client.chat_completion(messages, max_tokens=50)
    advanced_time = time.time() - start_time
    
    print(f"Базовый клиент: {basic_time:.3f}s")
    print(f"Продвинутый клиент: {advanced_time:.3f}s")
    print(f"Разница: {abs(advanced_time - basic_time):.3f}s")


async def main():
    """Главная функция с демонстрацией всех возможностей"""
    print("🚀 Kraken LLM Universal Client Examples")
    print("=" * 80)
    
    try:
        # Базовое использование
        await demo_basic_usage()
        
        # Продвинутое использование
        await demo_advanced_usage()
        
        # Кастомная конфигурация
        await demo_custom_configuration()
        
        # Создание из отчета анализатора
        await demo_from_capabilities_report()
        
        # Удобные функции
        await demo_convenience_functions()
        
        # Обработка ошибок
        await demo_error_handling()
        
        # Сравнение производительности
        await demo_performance_comparison()
        
        print("\n✅ Все демонстрации завершены успешно!")
        
    except KeyboardInterrupt:
        print("\n❌ Демонстрация прервана пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка во время демонстрации: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())