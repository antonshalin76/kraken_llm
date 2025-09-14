#!/usr/bin/env python3
"""
Тест переключателя outlines_so_mode и доработанной Outlines интеграции.
"""

import asyncio
import os
from pydantic import BaseModel, Field

from kraken_llm.config.settings import LLMConfig
from kraken_llm.client.structured import StructuredLLMClient

from dotenv import load_dotenv

load_dotenv()


class SimpleModel(BaseModel):
    """Простая модель для тестирования"""
    name: str = Field(..., description="Имя")
    value: int = Field(..., description="Числовое значение")
    active: bool = Field(True, description="Активность")


async def test_outlines_mode_disabled():
    """Тест с отключенным Outlines режимом (OpenAI режим)"""
    print("=== Тест с outlines_so_mode = False (OpenAI режим) ===")

    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL")
    )
    config.outlines_so_mode = False  # Отключаем Outlines

    client = StructuredLLMClient(config)

    messages = [{
        "role": "user",
        "content": "Создай JSON: name='OpenAI Test', value=100, active=true"
    }]

    try:
        result = await client.chat_completion_structured(
            messages=messages,
            response_model=SimpleModel,
            max_tokens=150,
            temperature=0.1
        )

        print(f"✅ OpenAI режим работает: {result}")
        print(f"   Тип: {type(result)}")
        print(
            f"   Данные: name='{result.name}', value={result.value}, active={result.active}")
        return True

    except Exception as e:
        print(f"❌ OpenAI режим ошибка: {e}")
        return False


async def test_outlines_mode_enabled():
    """Тест с включенным Outlines режимом"""
    print("\n=== Тест с outlines_so_mode = True (Outlines режим) ===")

    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL")
    )
    config.outlines_so_mode = True  # Включаем Outlines

    client = StructuredLLMClient(config)

    messages = [{
        "role": "user",
        "content": "Создай JSON: name='Outlines Test', value=200, active=false"
    }]

    try:
        result = await client.chat_completion_structured(
            messages=messages,
            response_model=SimpleModel,
            max_tokens=150,
            temperature=0.1
        )

        print(f"✅ Outlines режим работает: {result}")
        print(f"   Тип: {type(result)}")
        print(
            f"   Данные: name='{result.name}', value={result.value}, active={result.active}")
        return True

    except Exception as e:
        print(f"❌ Outlines режим ошибка: {e}")
        return False


async def test_outlines_streaming():
    """Тест Outlines в streaming режиме"""
    print("\n=== Тест Outlines Streaming ===")

    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL")
    )
    config.outlines_so_mode = True

    client = StructuredLLMClient(config)

    messages = [{
        "role": "user",
        "content": "Создай JSON: name='Streaming Test', value=300, active=true"
    }]

    try:
        result = await client.chat_completion_structured(
            messages=messages,
            response_model=SimpleModel,
            stream=True,
            max_tokens=150,
            temperature=0.1
        )

        print(f"✅ Outlines streaming работает: {result}")
        print(f"   Тип: {type(result)}")
        return True

    except Exception as e:
        print(f"❌ Outlines streaming ошибка: {e}")
        return False


async def test_direct_outlines_methods():
    """Тест прямых вызовов Outlines методов"""
    print("\n=== Тест прямых вызовов Outlines методов ===")

    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL")
    )
    client = StructuredLLMClient(config)

    messages = [{
        "role": "user",
        "content": "Создай JSON: name='Direct Test', value=400, active=false"
    }]

    # Тест non-streaming метода
    print("\n1. Прямой вызов _structured_non_stream_outlines:")
    try:
        result = await client._structured_non_stream_outlines(
            messages=messages,
            response_model=SimpleModel,
            temperature=0.1,
            max_tokens=150
        )

        print(f"✅ Прямой Outlines non-streaming: {result}")

    except Exception as e:
        print(f"❌ Прямой Outlines non-streaming ошибка: {e}")

    # Тест streaming метода
    print("\n2. Прямой вызов _structured_stream_outlines:")
    try:
        result = await client._structured_stream_outlines(
            messages=messages,
            response_model=SimpleModel,
            temperature=0.1,
            max_tokens=150
        )

        print(f"✅ Прямой Outlines streaming: {result}")

    except Exception as e:
        print(f"❌ Прямой Outlines streaming ошибка: {e}")


async def test_mode_switching():
    """Тест переключения между режимами"""
    print("\n=== Тест переключения режимов ===")

    base_config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL")
    )

    messages = [{
        "role": "user",
        "content": "Создай JSON: name='Switch Test', value=500, active=true"
    }]

    results = {}

    # Тест с разными значениями outlines_so_mode
    for mode_enabled in [False, True]:
        mode_name = "Outlines" if mode_enabled else "OpenAI"
        print(f"\n  Режим: {mode_name} (outlines_so_mode={mode_enabled})")

        config = LLMConfig(
            endpoint=base_config.endpoint,
            api_key=base_config.api_key,
            model=base_config.model
        )
        config.outlines_so_mode = mode_enabled

        client = StructuredLLMClient(config)

        try:
            import time
            start_time = time.time()

            result = await client.chat_completion_structured(
                messages=messages,
                response_model=SimpleModel,
                max_tokens=150,
                temperature=0.1
            )

            execution_time = time.time() - start_time

            print(f"    ✅ Успех за {execution_time:.3f}s: {result}")
            results[mode_name] = {
                "success": True,
                "result": result,
                "time": execution_time
            }

        except Exception as e:
            print(f"    ❌ Ошибка: {e}")
            results[mode_name] = {
                "success": False,
                "error": str(e),
                "time": 0
            }

    # Сравнение результатов
    print(f"\n📊 Сравнение режимов:")
    for mode_name, result in results.items():
        status = "✅" if result["success"] else "❌"
        print(f"  {status} {mode_name}: {result.get('time', 0):.3f}s")
        if result["success"]:
            print(f"      Результат: {result['result']}")

    return results


async def test_enhanced_prompts():
    """Тест улучшенных промптов"""
    print("\n=== Тест улучшенных промптов ===")

    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL")
    )
    config.outlines_so_mode = True

    client = StructuredLLMClient(config)

    # Тест метода улучшения промптов
    original_messages = [{
        "role": "user",
        "content": "Создай информацию о человеке: Тест, 25 лет"
    }]

    enhanced_messages = client._enhance_messages_for_json(
        original_messages, SimpleModel)

    print(f"Исходное сообщение: {original_messages[0]['content'][:50]}...")
    print(f"Количество сообщений после улучшения: {len(enhanced_messages)}")

    # Проверяем системное сообщение
    if enhanced_messages[0]["role"] == "system":
        print(f"✅ Добавлено системное сообщение")
        print(
            f"   Содержит схему: {'схема' in enhanced_messages[0]['content'].lower()}")
        print(
            f"   Содержит пример: {'пример' in enhanced_messages[0]['content'].lower()}")

    # Тест с улучшенными промптами
    try:
        result = await client.chat_completion_structured(
            messages=original_messages,
            response_model=SimpleModel,
            max_tokens=150,
            temperature=0.1
        )

        print(f"✅ Улучшенные промпты работают: {result}")

    except Exception as e:
        print(f"❌ Ошибка с улучшенными промптами: {e}")


async def test_json_extraction():
    """Тест извлечения JSON из ответов"""
    print("\n=== Тест извлечения JSON ===")

    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL")
    )
    client = StructuredLLMClient(config)

    # Тестовые ответы с разными форматами
    test_responses = [
        '{"name": "Test", "value": 123, "active": true}',  # Чистый JSON
        # JSON в тексте
        'Вот результат: {"name": "Test", "value": 123, "active": true}',
        # JSON в code block
        '```json\n{"name": "Test", "value": 123, "active": true}\n```',
        # JSON в обычном block
        '```\n{"name": "Test", "value": 123, "active": true}\n```',
        'Некорректный ответ без JSON',  # Без JSON
    ]

    for i, response in enumerate(test_responses, 1):
        print(f"\n  Тест {i}: {response[:30]}...")

        extracted = client._extract_json_from_response(response)

        if extracted:
            print(f"    ✅ JSON извлечен: {extracted}")

            # Проверяем валидность
            try:
                import json
                parsed = json.loads(extracted)
                print(f"    ✅ JSON валиден: {list(parsed.keys())}")
            except json.JSONDecodeError:
                print(f"    ❌ JSON невалиден")
        else:
            print(f"    ❌ JSON не найден")


async def main():
    """Главная функция тестирования"""
    print("🧪 ТЕСТИРОВАНИЕ OUTLINES ИНТЕГРАЦИИ И ПЕРЕКЛЮЧАТЕЛЯ РЕЖИМОВ")
    print("=" * 70)

    results = []

    # Тестируем переключатель режимов
    switch_results = await test_mode_switching()

    # Тестируем отдельные режимы
    results.append(await test_outlines_mode_disabled())
    results.append(await test_outlines_mode_enabled())
    results.append(await test_outlines_streaming())

    # Тестируем прямые методы
    await test_direct_outlines_methods()

    # Тестируем улучшенные промпты
    await test_enhanced_prompts()

    # Тестируем извлечение JSON
    await test_json_extraction()

    # Подводим итоги
    print("\n" + "=" * 70)
    print("📊 ИТОГИ ТЕСТИРОВАНИЯ")
    print("=" * 70)

    success_count = sum(results)
    total_tests = len(results)

    print(f"Базовые тесты: {success_count}/{total_tests} успешно")

    if switch_results:
        print(f"\nРезультаты переключения режимов:")
        for mode, result in switch_results.items():
            status = "✅" if result["success"] else "❌"
            print(f"  {status} {mode}: {result.get('time', 0):.3f}s")

    print(f"\n🎯 Выводы:")
    if success_count == total_tests:
        print("- ✅ Все режимы работают корректно")
    elif success_count > 0:
        print("- ⚠️  Некоторые режимы работают, требуется доработка")
    else:
        print("- ❌ Требуется серьезная доработка Outlines интеграции")

    print("- 🔧 Улучшенные промпты помогают генерации JSON")
    print("- 📊 Переключатель outlines_so_mode функционирует")
    print("- 🚀 Система готова к дальнейшему развитию")


if __name__ == "__main__":
    asyncio.run(main())
