#!/usr/bin/env python3
"""
Тест интеграции универсального клиента

Простой тест для проверки корректности интеграции UniversalLLMClient
в основную кодовую базу Kraken LLM.

Использует реальное подключение к LLM API если доступна конфигурация.
"""

import sys
import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

# Добавляем путь к модулям Kraken LLM
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_imports():
    """Тест импортов универсального клиента"""
    print("🔍 Тестирование импортов...")

    try:
        # Основные импорты
        from kraken_llm import (
            UniversalLLMClient,
            UniversalClientConfig,
            UniversalCapability,
            create_universal_client,
            create_universal_client_from_report,
            create_basic_client,
            create_advanced_client,
            create_full_client
        )
        print("✅ Основные импорты успешны")

        # Импорты из фабрики
        from kraken_llm.client.factory import ClientFactory
        print("✅ Импорт фабрики успешен")

        # Проверка регистрации в фабрике
        available_types = ClientFactory.get_available_client_types()
        assert 'universal' in available_types, "Universal client не зарегистрирован в фабрике"
        print("✅ Universal client зарегистрирован в фабрике")

        return True

    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        return False
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")
        return False


def test_client_creation():
    """Тест создания клиентов"""
    print("\n🏗️ Тестирование создания клиентов...")

    try:
        from kraken_llm import (
            create_universal_client,
            create_basic_client,
            create_advanced_client,
            UniversalCapability
        )

        # Тест создания базового клиента
        basic_client = create_basic_client()
        assert basic_client is not None
        print("✅ Базовый клиент создан")

        # Тест создания продвинутого клиента
        advanced_client = create_advanced_client()
        assert advanced_client is not None
        print("✅ Продвинутый клиент создан")

        # Тест создания кастомного клиента
        capabilities = {
            UniversalCapability.CHAT_COMPLETION,
            UniversalCapability.STREAMING
        }
        custom_client = create_universal_client(capabilities=capabilities)
        assert custom_client is not None
        print("✅ Кастомный клиент создан")

        return True

    except Exception as e:
        print(f"❌ Ошибка создания клиента: {e}")
        return False


def test_capabilities():
    """Тест возможностей"""
    print("\n⚙️ Тестирование возможностей...")

    try:
        from kraken_llm import UniversalCapability, create_basic_client

        # Проверка перечисления возможностей
        capabilities = list(UniversalCapability)
        assert len(capabilities) > 0
        print(f"✅ Найдено {len(capabilities)} возможностей")

        # Проверка базового клиента
        client = create_basic_client()
        available_caps = client.get_available_capabilities()
        assert isinstance(available_caps, list)
        assert len(available_caps) > 0
        print(f"✅ Базовый клиент имеет {len(available_caps)} возможностей")

        return True

    except Exception as e:
        print(f"❌ Ошибка тестирования возможностей: {e}")
        return False


def test_factory_integration():
    """Тест интеграции с фабрикой"""
    print("\n🏭 Тестирование интеграции с фабрикой...")

    try:
        from kraken_llm.client.factory import ClientFactory, create_client

        # Тест создания через фабрику
        client = ClientFactory.create_client(client_type='universal')
        assert client is not None
        print("✅ Создание через фабрику успешно")

        # Тест автоопределения
        auto_client = create_client()  # Должен создать adaptive или universal
        assert auto_client is not None
        print("✅ Автоопределение типа клиента работает")

        # Проверка доступных типов
        types = ClientFactory.get_available_client_types()
        expected_types = ['standard', 'streaming', 'structured', 'universal']
        for expected_type in expected_types:
            assert expected_type in types, f"Тип {expected_type} не найден"
        print(f"✅ Все ожидаемые типы клиентов доступны: {len(types)}")

        return True

    except Exception as e:
        print(f"❌ Ошибка интеграции с фабрикой: {e}")
        return False


async def test_real_connection():
    """Тест реального подключения к LLM API"""
    print("\n🌐 Тестирование реального подключения...")

    if not os.getenv('LLM_ENDPOINT'):
        print("⚠️ Конфигурация LLM не найдена в .env - пропускаем тест подключения")
        return True

    try:
        from kraken_llm import create_basic_client, LLMConfig

        config = LLMConfig()
        print(f"🔗 Подключение к: {config.endpoint}")
        print(f"📝 Модель: {config.model}")

        async with create_basic_client(config=config) as client:
            # Простой тест подключения
            response = await client.chat_completion([
                {"role": "user", "content": "Тест подключения. Ответь одним словом: OK"}
            ], max_tokens=10)

            assert response is not None
            assert len(response.strip()) > 0
            print(f"✅ Реальное подключение работает: {response.strip()}")

            return True

    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        print("   Проверьте настройки в .env файле и доступность сервера")
        return False


async def main():
    """Главная функция тестирования"""
    print("🧪 Тест интеграции UniversalLLMClient")
    print("=" * 50)

    # Синхронные тесты
    sync_tests = [
        test_imports,
        test_client_creation,
        test_capabilities,
        test_factory_integration
    ]

    # Асинхронные тесты
    async_tests = [
        test_real_connection
    ]

    passed = 0
    total = len(sync_tests) + len(async_tests)

    # Запускаем синхронные тесты
    for test in sync_tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Критическая ошибка в тесте {test.__name__}: {e}")

    # Запускаем асинхронные тесты
    for test in async_tests:
        try:
            if await test():
                passed += 1
        except Exception as e:
            print(f"❌ Критическая ошибка в тесте {test.__name__}: {e}")

    print("\n" + "=" * 50)
    print(f"📊 Результаты: {passed}/{total} тестов пройдено")

    if passed == total:
        print("🎉 Все тесты пройдены! Интеграция успешна.")
        return True
    else:
        print("⚠️ Некоторые тесты не пройдены. Проверьте интеграцию.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
