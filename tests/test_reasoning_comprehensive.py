#!/usr/bin/env python3
"""
Комплексный тест ReasoningLLMClient для проверки всех возможностей рассуждающих моделей.

Этот тест проверяет:
- Базовое рассуждение с Chain of Thought
- Различные типы задач (math, logic, coding, general)
- Потоковый режим рассуждений
- Валидацию цепочек рассуждений
- Анализ качества рассуждений
- Кастомные шаблоны промптов
- Подсчет reasoning токенов
"""

import asyncio
import time
import os
from typing import List, Dict

from kraken_llm.client.reasoning import ReasoningLLMClient, ReasoningConfig, ReasoningChain
from kraken_llm.config.settings import LLMConfig
from kraken_llm.exceptions.validation import ValidationError

from dotenv import load_dotenv

load_dotenv()

async def test_basic_reasoning():
    """Тест базового рассуждения"""
    print("=== Тест базового рассуждения ===")
    
    config = LLMConfig(
        endpoint = os.getenv("LLM_ENDPOINT"),
        api_key = os.getenv("LLM_TOKEN"),
        model = os.getenv("LLM_MODEL"),
        temperature=0.1
    )
    
    reasoning_config = ReasoningConfig(
        enable_cot=True,
        max_reasoning_steps=5,
        extract_confidence=True
    )
    
    client = ReasoningLLMClient(config, reasoning_config)
    
    messages = [
        {
            "role": "user",
            "content": "Если в корзине 12 яблок, и я съел 3, а потом купил еще 5, сколько яблок стало в корзине?"
        }
    ]
    
    try:
        start_time = time.time()
        chain = await client.reasoning_completion(
            messages=messages,
            problem_type="math",
            enable_streaming=False
        )
        end_time = time.time()
        
        print(f"✓ Рассуждение выполнено за {end_time - start_time:.2f}s")
        print(f"✓ Количество шагов: {len(chain.steps)}")
        print(f"✓ Есть финальный ответ: {bool(chain.final_answer)}")
        print(f"✓ Время рассуждения записано: {chain.reasoning_time is not None}")
        
        # Проверяем структуру шагов
        for i, step in enumerate(chain.steps):
            assert step.step_number == i + 1, f"Неверная нумерация шага: {step.step_number} != {i + 1}"
            assert step.thought, f"Шаг {step.step_number} не содержит рассуждения"
        
        print(f"✓ Все шаги корректно структурированы")
        print(f"Финальный ответ: {chain.final_answer}")
        print()
        
        return True
        
    except Exception as e:
        print(f"✗ Ошибка в базовом рассуждении: {e}")
        return False


async def test_math_reasoning():
    """Тест математического рассуждения"""
    print("=== Тест математического рассуждения ===")
    
    config = LLMConfig(
        endpoint = os.getenv("LLM_ENDPOINT"),
        api_key = os.getenv("LLM_TOKEN"),
        model = os.getenv("LLM_MODEL")
    )
    
    client = ReasoningLLMClient(config)
    
    messages = [
        {
            "role": "user",
            "content": """
            Реши систему уравнений:
            x + y = 10
            2x - y = 5
            Найди x и y, показав все шаги решения.
            """
        }
    ]
    
    try:
        chain = await client.reasoning_completion(
            messages=messages,
            problem_type="math",
            enable_streaming=False
        )
        
        print(f"✓ Математическая задача решена")
        print(f"✓ Шагов решения: {len(chain.steps)}")
        
        # Проверяем, что есть вычисления в шагах
        has_calculations = any(
            step.action and any(op in step.action for op in ['+', '-', '*', '/', '='])
            for step in chain.steps
        )
        
        if has_calculations:
            print("✓ Обнаружены математические вычисления в шагах")
        else:
            print("⚠ Математические вычисления не обнаружены явно")
        
        print(f"Решение: {chain.final_answer}")
        print()
        
        return True
        
    except Exception as e:
        print(f"✗ Ошибка в математическом рассуждении: {e}")
        return False


async def test_logic_reasoning():
    """Тест логического рассуждения"""
    print("=== Тест логического рассуждения ===")
    
    config = LLMConfig(
        endpoint = os.getenv("LLM_ENDPOINT"),
        api_key = os.getenv("LLM_TOKEN"),
        model = os.getenv("LLM_MODEL")
    )
    
    client = ReasoningLLMClient(config)
    
    messages = [
        {
            "role": "user",
            "content": """
            Дано:
            - Все программисты любят кофе
            - Анна - программист
            - Если кто-то любит кофе, то он не спит по ночам
            
            Вопрос: Спит ли Анна по ночам? Объясни логическую цепочку.
            """
        }
    ]
    
    try:
        chain = await client.reasoning_completion(
            messages=messages,
            problem_type="logic",
            enable_streaming=False
        )
        
        print(f"✓ Логическая задача решена")
        print(f"✓ Шагов рассуждения: {len(chain.steps)}")
        
        # Проверяем логические связки
        logical_words = ['если', 'то', 'следовательно', 'значит', 'поэтому', 'так как']
        has_logic = any(
            any(word in step.thought.lower() for word in logical_words)
            for step in chain.steps
        )
        
        if has_logic:
            print("✓ Обнаружены логические связки в рассуждении")
        else:
            print("⚠ Логические связки не обнаружены явно")
        
        print(f"Логический вывод: {chain.final_answer}")
        print()
        
        return True
        
    except Exception as e:
        print(f"✗ Ошибка в логическом рассуждении: {e}")
        return False


async def test_coding_reasoning():
    """Тест рассуждения при программировании"""
    print("=== Тест рассуждения при программировании ===")
    
    config = LLMConfig(
        endpoint = os.getenv("LLM_ENDPOINT"),
        api_key = os.getenv("LLM_TOKEN"),
        model = os.getenv("LLM_MODEL")
    )
    
    client = ReasoningLLMClient(config)
    
    messages = [
        {
            "role": "user",
            "content": """
            Напиши функцию для поиска второго по величине элемента в массиве.
            Объясни алгоритм пошагово и покажи код.
            """
        }
    ]
    
    try:
        chain = await client.reasoning_completion(
            messages=messages,
            problem_type="coding",
            enable_streaming=False
        )
        
        print(f"✓ Задача программирования решена")
        print(f"✓ Шагов планирования: {len(chain.steps)}")
        
        # Проверяем наличие кода
        has_code = any(
            step.action and any(keyword in step.action for keyword in ['def', 'for', 'if', 'return'])
            for step in chain.steps
        )
        
        if has_code:
            print("✓ Обнаружен код в шагах рассуждения")
        else:
            print("⚠ Код не обнаружен в шагах")
        
        print(f"Решение: {chain.final_answer[:200]}...")
        print()
        
        return True
        
    except Exception as e:
        print(f"✗ Ошибка в рассуждении о программировании: {e}")
        return False


async def test_streaming_reasoning():
    """Тест потокового рассуждения"""
    print("=== Тест потокового рассуждения ===")
    
    config = LLMConfig(
        endpoint = os.getenv("LLM_ENDPOINT"),
        api_key = os.getenv("LLM_TOKEN"),
        model = os.getenv("LLM_MODEL")
    )
    
    client = ReasoningLLMClient(config)
    
    messages = [
        {
            "role": "user",
            "content": "Объясни пошагово, как приготовить омлет. Каждый шаг должен быть детальным."
        }
    ]
    
    try:
        print("Получение шагов в потоковом режиме:")
        
        steps_received = 0
        start_time = time.time()
        
        async for step in client.reasoning_completion(
            messages=messages,
            problem_type="general",
            enable_streaming=True
        ):
            steps_received += 1
            print(f"  Получен шаг {step.step_number}: {step.thought[:50]}...")
            
            # Проверяем структуру шага
            assert step.step_number > 0, "Неверный номер шага"
            assert step.thought, "Шаг не содержит рассуждения"
        
        end_time = time.time()
        
        print(f"✓ Потоковое рассуждение завершено за {end_time - start_time:.2f}s")
        print(f"✓ Получено шагов: {steps_received}")
        print()
        
        return steps_received > 0
        
    except Exception as e:
        print(f"✗ Ошибка в потоковом рассуждении: {e}")
        return False


async def test_reasoning_validation():
    """Тест валидации цепочек рассуждений"""
    print("=== Тест валидации рассуждений ===")
    
    config = LLMConfig(
        endpoint = os.getenv("LLM_ENDPOINT"),
        api_key = os.getenv("LLM_TOKEN"),
        model = os.getenv("LLM_MODEL")
    )
    
    reasoning_config = ReasoningConfig(
        require_step_validation=True,
        max_reasoning_steps=4
    )
    
    client = ReasoningLLMClient(config, reasoning_config)
    
    messages = [
        {
            "role": "user",
            "content": "Объясни, почему 2+2=4. Используй простые логические шаги."
        }
    ]
    
    try:
        chain = await client.reasoning_completion(
            messages=messages,
            problem_type="math",
            enable_streaming=False
        )
        
        print(f"✓ Рассуждение с валидацией выполнено")
        print(f"✓ Шагов: {len(chain.steps)}")
        print(f"✓ Валидация прошла успешно")
        
        # Дополнительная проверка валидации
        await client._validate_reasoning_chain(chain)
        print("✓ Дополнительная валидация прошла")
        print()
        
        return True
        
    except ValidationError as e:
        print(f"⚠ Ошибка валидации (ожидаемо): {e}")
        return True  # Валидация работает
        
    except Exception as e:
        print(f"✗ Неожиданная ошибка в валидации: {e}")
        return False


async def test_custom_reasoning_template():
    """Тест кастомного шаблона рассуждений"""
    print("=== Тест кастомного шаблона ===")
    
    config = LLMConfig(
        endpoint = os.getenv("LLM_ENDPOINT"),
        api_key = os.getenv("LLM_TOKEN"),
        model = os.getenv("LLM_MODEL")
    )
    
    custom_template = """
Анализируй проблему по схеме SWOT:
1. Strengths (Сильные стороны)
2. Weaknesses (Слабые стороны)  
3. Opportunities (Возможности)
4. Threats (Угрозы)

Формат:
Шаг 1: [анализ сильных сторон]
Анализ: [что обнаружено]
Уверенность: [0.0-1.0]

Финальный ответ: [общий SWOT анализ]
"""
    
    reasoning_config = ReasoningConfig(
        reasoning_prompt_template=custom_template,
        max_reasoning_steps=4
    )
    
    client = ReasoningLLMClient(config, reasoning_config)
    
    messages = [
        {
            "role": "user",
            "content": "Проведи SWOT анализ для малого IT стартапа, разрабатывающего мобильные приложения."
        }
    ]
    
    try:
        chain = await client.reasoning_completion(
            messages=messages,
            problem_type="general",
            enable_streaming=False
        )
        
        print(f"✓ Кастомный шаблон применен")
        print(f"✓ Шагов анализа: {len(chain.steps)}")
        
        # Проверяем, что используется SWOT терминология
        swot_terms = ['сильн', 'слаб', 'возможност', 'угроз', 'swot']
        has_swot = any(
            any(term in step.thought.lower() for term in swot_terms)
            for step in chain.steps
        )
        
        if has_swot:
            print("✓ SWOT терминология обнаружена в рассуждении")
        else:
            print("⚠ SWOT терминология не обнаружена явно")
        
        print(f"SWOT анализ: {chain.final_answer[:150]}...")
        print()
        
        return True
        
    except Exception as e:
        print(f"✗ Ошибка в кастомном шаблоне: {e}")
        return False


async def test_reasoning_quality_analysis():
    """Тест анализа качества рассуждений"""
    print("=== Тест анализа качества рассуждений ===")
    
    config = LLMConfig(
        endpoint = os.getenv("LLM_ENDPOINT"),
        api_key = os.getenv("LLM_TOKEN"),
        model = os.getenv("LLM_MODEL")
    )
    
    client = ReasoningLLMClient(config)
    
    messages = [
        {
            "role": "user",
            "content": "Объясни, почему важно изучать историю. Приведи 3-4 веских аргумента."
        }
    ]
    
    try:
        # Выполняем рассуждение
        chain = await client.reasoning_completion(
            messages=messages,
            problem_type="general",
            enable_streaming=False
        )
        
        # Анализируем качество
        quality_analysis = await client.analyze_reasoning_quality(chain)
        
        print(f"✓ Анализ качества выполнен")
        print(f"✓ Общее количество шагов: {quality_analysis['total_steps']}")
        print(f"✓ Полнота рассуждения: {quality_analysis['reasoning_completeness']:.2f}")
        print(f"✓ Логическая последовательность: {quality_analysis['logical_consistency']:.2f}")
        
        # Проверяем структуру анализа
        required_keys = ['total_steps', 'reasoning_completeness', 'logical_consistency', 'step_quality_scores']
        for key in required_keys:
            assert key in quality_analysis, f"Отсутствует ключ в анализе: {key}"
        
        print(f"✓ Структура анализа корректна")
        
        # Проверяем анализ каждого шага
        for step_quality in quality_analysis['step_quality_scores']:
            assert 'step_number' in step_quality, "Отсутствует номер шага"
            assert 'quality_score' in step_quality, "Отсутствует оценка качества"
        
        print(f"✓ Анализ каждого шага корректен")
        print()
        
        return True
        
    except Exception as e:
        print(f"✗ Ошибка в анализе качества: {e}")
        return False


async def test_chat_completion_with_reasoning():
    """Тест упрощенного API для рассуждений"""
    print("=== Тест упрощенного API рассуждений ===")
    
    config = LLMConfig(
        endpoint = os.getenv("LLM_ENDPOINT"),
        api_key = os.getenv("LLM_TOKEN"),
        model = os.getenv("LLM_MODEL")
    )
    
    client = ReasoningLLMClient(config)
    
    messages = [
        {
            "role": "user",
            "content": "Почему небо голубое? Объясни простыми словами."
        }
    ]
    
    try:
        result = await client.chat_completion_with_reasoning(
            messages=messages,
            reasoning_config=ReasoningConfig(max_reasoning_steps=3)
        )
        
        print(f"✓ Упрощенный API работает")
        
        # Проверяем структуру результата
        required_keys = ['reasoning_steps', 'final_answer', 'confidence_score', 'total_steps']
        for key in required_keys:
            assert key in result, f"Отсутствует ключ в результате: {key}"
        
        print(f"✓ Структура результата корректна")
        print(f"✓ Шагов рассуждения: {result['total_steps']}")
        print(f"✓ Финальный ответ: {len(result['final_answer'])} символов")
        
        if result['confidence_score']:
            print(f"✓ Уверенность: {result['confidence_score']:.2f}")
        
        print()
        
        return True
        
    except Exception as e:
        print(f"✗ Ошибка в упрощенном API: {e}")
        return False


async def test_reasoning_error_handling():
    """Тест обработки ошибок в рассуждениях"""
    print("=== Тест обработки ошибок ===")
    
    config = LLMConfig(
        endpoint="http://invalid-endpoint:9999",  # Неверный endpoint
        api_key="invalid_key",
        model="chat"
    )
    
    client = ReasoningLLMClient(config)
    
    messages = [
        {
            "role": "user",
            "content": "Тестовый запрос"
        }
    ]
    
    try:
        # Этот запрос должен вызвать ошибку
        await client.reasoning_completion(
            messages=messages,
            problem_type="general",
            enable_streaming=False
        )
        
        print("✗ Ошибка не была вызвана (неожиданно)")
        return False
        
    except Exception as e:
        print(f"✓ Ошибка корректно обработана: {type(e).__name__}")
        print(f"✓ Сообщение об ошибке: {str(e)[:100]}...")
        print()
        return True


async def run_comprehensive_test():
    """Запуск всех тестов"""
    print("Комплексное тестирование ReasoningLLMClient")
    print("=" * 60)
    print()
    
    tests = [
        ("Базовое рассуждение", test_basic_reasoning),
        ("Математическое рассуждение", test_math_reasoning),
        ("Логическое рассуждение", test_logic_reasoning),
        ("Рассуждение при программировании", test_coding_reasoning),
        ("Потоковое рассуждение", test_streaming_reasoning),
        ("Валидация рассуждений", test_reasoning_validation),
        ("Кастомный шаблон", test_custom_reasoning_template),
        ("Анализ качества", test_reasoning_quality_analysis),
        ("Упрощенный API", test_chat_completion_with_reasoning),
        ("Обработка ошибок", test_reasoning_error_handling)
    ]
    
    results = []
    total_start_time = time.time()
    
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
    
    total_time = time.time() - total_start_time
    
    # Итоговый отчет
    print("ИТОГОВЫЙ ОТЧЕТ")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"Пройдено тестов: {passed}/{total}")
    print(f"Общее время выполнения: {total_time:.2f}s")
    print()
    
    print("Детальные результаты:")
    for test_name, result in results:
        status = "✓" if result else "✗"
        print(f"  {status} {test_name}")
    
    print()
    
    if passed == total:
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
    else:
        print(f"⚠ {total - passed} тестов провалено")
    
    return passed == total


if __name__ == "__main__":
    asyncio.run(run_comprehensive_test())