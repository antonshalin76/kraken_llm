#!/usr/bin/env python3
"""
Пример использования фабрики клиентов Kraken LLM фреймворка.

Демонстрирует создание различных типов клиентов через фабрику
и их базовое использование.
"""

import asyncio
import os
from dotenv import load_dotenv
from kraken_llm import (
    create_client,
    create_standard_client,
    create_streaming_client,
    create_structured_client,
    create_reasoning_client,
    create_multimodal_client,
    create_adaptive_client,
    create_asr_client,
    create_embeddings_client,
    LLMConfig,
    ClientFactory
)
from pydantic import BaseModel

load_dotenv()

class PersonInfo(BaseModel):
    """Модель для демонстрации structured output."""
    name: str
    age: int
    occupation: str


async def demo_client_factory():
    """Демонстрация использования фабрики клиентов."""
    print("Демонстрация фабрики клиентов Kraken LLM")
    print("=" * 50)

    # Конфигурация для демонстрации
    config = LLMConfig(
        endpoint="http://localhost:8080",
        api_key="demo_key",
        model="demo_model",
        temperature=0.7
    )

    print("\n1. 📋 Доступные типы клиентов:")
    available_types = ClientFactory.get_available_client_types()
    for client_type, client_class in available_types.items():
        print(f"   • {client_type}: {client_class.__name__}")

    print("\n2. 🏭 Создание клиентов через фабрику:")

    # Создание различных типов клиентов
    clients = {
        "standard": create_standard_client(config),
        "streaming": create_streaming_client(config),
        "structured": create_structured_client(config),
        "reasoning": create_reasoning_client(config),
        "multimodal": create_multimodal_client(config),
        "adaptive": create_adaptive_client(config),
        "asr": create_asr_client(config),
        "embeddings": create_embeddings_client(config),
    }

    for client_type, client in clients.items():
        print(f"   ✅ {client_type}: {client.__class__.__name__}")
        print(f"      Endpoint: {client.config.endpoint}")
        print(f"      Model: {client.config.model}")
        print(f"      Temperature: {client.config.temperature}")

    print("\n3. 🎯 Автоматическое определение типа клиента:")

    # Автоматическое определение на основе параметров
    auto_clients = [
        ("structured", {"response_model": PersonInfo}),
        ("streaming", {"stream": True}),
        ("reasoning", {"reasoning_mode": True}),
        ("multimodal", {"media_input": True}),
        ("asr", {"audio_file": "test.wav"}),
        ("embeddings", {"embeddings": True}),
        ("adaptive", {}),  # По умолчанию
    ]

    for expected_type, params in auto_clients:
        client = create_client(config=config, **params)
        actual_type = client.__class__.__name__
        print(f"   🔍 Параметры: {params}")
        print(f"      Ожидаемый: {expected_type} → Получен: {actual_type}")

    print("\n4. ⚙️ Создание с параметрами:")

    # Создание клиента с переопределением параметров
    custom_client = create_client(
        client_type="standard",
        endpoint="http://custom-server:8080",
        model="custom-model",
        temperature=0.9,
        max_tokens=2000
    )

    print(f"   📝 Кастомный клиент: {custom_client.__class__.__name__}")
    print(f"      Endpoint: {custom_client.config.endpoint}")
    print(f"      Model: {custom_client.config.model}")
    print(f"      Temperature: {custom_client.config.temperature}")
    print(f"      Max tokens: {custom_client.config.max_tokens}")

    print("\n5. 🔧 Регистрация кастомного типа клиента:")

    # Создаем кастомный клиент
    class CustomDemoClient(clients["standard"].__class__):
        """Кастомный демо-клиент."""
        pass

    # Регистрируем новый тип
    ClientFactory.register_client_type("demo", CustomDemoClient)

    # Создаем клиент нового типа
    demo_client = create_client("demo", config)
    print(f"   🎨 Кастомный тип: {demo_client.__class__.__name__}")

    # Проверяем, что тип зарегистрирован
    updated_types = ClientFactory.get_available_client_types()
    print(f"   📊 Всего типов клиентов: {len(updated_types)}")

    print("\n6. 🔄 Независимость клиентов:")

    # Создаем несколько независимых клиентов
    client1 = create_client("standard", endpoint="http://server1:8080", model="model1")
    client2 = create_client("streaming", endpoint="http://server2:8080", model="model2")

    print(f"   🖥️  Клиент 1: {client1.config.endpoint} / {client1.config.model}")
    print(f"   🖥️  Клиент 2: {client2.config.endpoint} / {client2.config.model}")
    print(f"   ✅ Независимость: {client1.config is not client2.config}")

    print("\n7. 📊 Статистика:")
    print(f"   • Создано клиентов: {len(clients) + len(auto_clients) + 3}")
    print(f"   • Типов клиентов: {len(updated_types)}")
    print(f"   • Кастомных типов: 1")

    print("\n✨ Демонстрация завершена!")


def demo_simple_usage():
    """Простой пример использования."""
    print("\n🎯 Простое использование:")
    print("-" * 30)

    # Самый простой способ создания клиента
    client = create_client()
    print(f"Создан клиент: {client.__class__.__name__}")

    # С параметрами
    client_with_params = create_client(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL"),
        temperature=0.8
    )
    print(f"Клиент с параметрами: {client_with_params.config.model}")


if __name__ == "__main__":
    print("🔥 Kraken LLM Framework - Демонстрация фабрики клиентов")
    print("=" * 60)

    # Запуск демонстрации
    asyncio.run(demo_client_factory())

    # Простое использование
    demo_simple_usage()

    print("\nДля использования в реальных проектах:")
    print("1. Настройте правильный endpoint и API ключ")
    print("2. Выберите подходящий тип клиента для вашей задачи")
    print("3. Используйте фабрику для создания клиентов")
    print("4. Обрабатывайте ошибки соответствующим образом")
