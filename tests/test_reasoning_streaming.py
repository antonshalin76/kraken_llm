#!/usr/bin/env python3
"""
Тест исправлений для потокового режима ReasoningLLMClient.
"""

import asyncio
import os

from kraken_llm.client.reasoning import ReasoningLLMClient, ReasoningConfig
from kraken_llm.config.settings import LLMConfig

from dotenv import load_dotenv

load_dotenv()


async def test_streaming_fix():
    """Тест исправленного потокового режима"""
    print("=== Тест исправленного потокового режима ===")

    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL")
    )

    client = ReasoningLLMClient(config)

    messages = [
        {
            "role": "user",
            "content": "Объясни пошагово, как сварить яйцо вкрутую. Нужно 3-4 простых шага."
        }
    ]

    try:
        print("Получение шагов в потоковом режиме:")
        print()

        step_count = 0
        async for step in client.reasoning_completion(
            messages=messages,
            problem_type="general",
            enable_streaming=True
        ):
            step_count += 1
            print(f"Шаг {step.step_number}: {step.thought[:100]}...")
            if step.action:
                print(f"  Действие: {step.action[:50]}...")
            if step.confidence:
                print(f"  Уверенность: {step.confidence:.2f}")
            print()

            # Ограничиваем количество шагов для теста
            if step_count >= 5:
                break

        print(f"✓ Потоковый режим работает! Получено шагов: {step_count}")
        return True

    except Exception as e:
        print(f"✗ Ошибка в потоковом режиме: {e}")
        return False


async def test_parsing_improvements():
    """Тест улучшенного парсинга"""
    print("=== Тест улучшенного парсинга ===")

    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL")
    )

    client = ReasoningLLMClient(config)

    messages = [
        {
            "role": "user",
            "content": "Реши простую задачу: 15 + 27 = ? Покажи вычисления пошагово."
        }
    ]

    try:
        chain = await client.reasoning_completion(
            messages=messages,
            problem_type="math",
            enable_streaming=False
        )

        print(f"✓ Парсинг выполнен")
        print(f"✓ Количество шагов: {len(chain.steps)}")
        print(f"✓ Финальный ответ: {chain.final_answer}")

        # Проверяем качество парсинга
        for i, step in enumerate(chain.steps):
            print(f"Шаг {step.step_number}: {step.thought}")
            if step.action:
                print(f"  Действие: {step.action}")
            if step.observation:
                print(f"  Результат: {step.observation}")
            if step.confidence:
                print(f"  Уверенность: {step.confidence}")
            print()

        return True

    except Exception as e:
        print(f"✗ Ошибка в парсинге: {e}")
        return False


async def main():
    """Запуск тестов исправлений"""
    print("Тестирование исправлений ReasoningLLMClient")
    print("=" * 50)
    print()

    tests = [
        ("Потоковый режим", test_streaming_fix),
        ("Улучшенный парсинг", test_parsing_improvements)
    ]

    results = []

    for test_name, test_func in tests:
        print(f"Запуск теста: {test_name}")
        try:
            result = await test_func()
            results.append((test_name, result))
            status = "✓ ПРОЙДЕН" if result else "✗ ПРОВАЛЕН"
            print(f"Результат: {status}")
        except Exception as e:
            results.append((test_name, False))
            print(f"✗ ОШИБКА: {e}")

        print("-" * 40)
        print()

    # Итоговый отчет
    passed = sum(1 for _, result in results if result)
    total = len(results)

    print(f"Пройдено тестов: {passed}/{total}")

if __name__ == "__main__":
    asyncio.run(main())
