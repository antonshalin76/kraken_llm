#!/usr/bin/env python3
"""
Пример использования ReasoningLLMClient для решения сложных задач.

Этот пример демонстрирует различные возможности рассуждающих моделей:
- Математические задачи с пошаговым решением
- Логические задачи с анализом фактов
- Задачи программирования с планированием алгоритма
- Потоковый режим рассуждений
- Анализ качества рассуждений
"""

import asyncio
import os
from dotenv import load_dotenv

from kraken_llm.client.reasoning import ReasoningLLMClient, ReasoningConfig
from kraken_llm.config.settings import LLMConfig

load_dotenv()

common_config = LLMConfig(
    endpoint=os.getenv("LLM_REASONING_ENDPOINT"),
    api_key=os.getenv("LLM_REASONING_TOKEN"),
    model=os.getenv("LLM_REASONING_MODEL")
)

async def math_reasoning_example():
    """Пример решения математической задачи с рассуждениями"""
    print("=== Пример математического рассуждения ===")
    
    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL"),
        temperature=0.1
    )

    reasoning_config = ReasoningConfig(
        enable_cot=False,
        max_reasoning_steps=8,
        reasoning_temperature=0.1,
        extract_confidence=True
    )
    
    client = ReasoningLLMClient(common_config, reasoning_config)
    
    # Математическая задача
    task = """
            В магазине есть 3 полки с книгами. На первой полке 24 книги, 
            на второй - в 2 раза больше чем на первой, а на третьей - на 8 книг меньше чем на второй.
            Сколько всего книг в магазине?
            """
    print(f"Задача: {task}")
    messages = [
        {
            "role": "user",
            "content": task + " Покажи подробные вычисления."
        }
    ]
    
    try:
        # Выполняем рассуждение
        chain = await client.reasoning_completion(
            messages=messages,
            problem_type="math",
            enable_streaming=False
        )
        
        print(f"Количество шагов рассуждения: {len(chain.steps)}")
        print(f"Время рассуждения: {chain.reasoning_time:.2f}s")
        print(f"Общая уверенность: {chain.confidence_score:.2f}")
        print()
        
        # Выводим каждый шаг
        for step in chain.steps:
            print(f"Шаг {step.step_number}: {step.thought}")
            if step.action:
                print(f"  Действие: {step.action}")
            if step.observation:
                print(f"  Результат: {step.observation}")
            if step.confidence:
                print(f"  Уверенность: {step.confidence:.2f}")
            print()
        
        print(f"Финальный ответ: {chain.final_answer}")
        print()
        
        # Анализируем качество рассуждений
        quality_analysis = await client.analyze_reasoning_quality(chain)
        print("Анализ качества рассуждений:")
        print(f"  Полнота рассуждения: {quality_analysis['reasoning_completeness']:.2f}")
        print(f"  Логическая последовательность: {quality_analysis['logical_consistency']:.2f}")
        
    except Exception as e:
        print(f"Ошибка в математическом рассуждении: {e}")


async def logic_reasoning_example():
    """Пример решения логической задачи"""
    print("=== Пример логического рассуждения ===")
    
    config = common_config
    
    reasoning_config = ReasoningConfig(
        max_reasoning_steps=6,
        require_step_validation=True
    )
    
    client = ReasoningLLMClient(config, reasoning_config)
    
    task = """
            У нас есть следующие факты:
            1. Все кошки - млекопитающие
            2. Все млекопитающие дышат воздухом
            3. Мурка - кошка
            4. Если животное дышит воздухом, то оно живое
            
            Вопрос: Является ли Мурка живой?
            """
    print(f"Логическая задача: {task}")

    messages = [
        {
            "role": "user",
            "content": task + " Объясни логическую цепочку."
        }
    ]
    
    try:
        chain = await client.reasoning_completion(
            messages=messages,
            problem_type="logic",
            enable_streaming=False
        )
        
        print(f"Логических шагов: {len(chain.steps)}")
        print()
        
        for step in chain.steps:
            print(f"Шаг {step.step_number}: {step.thought}")
            if step.observation:
                print(f"  Вывод: {step.observation}")
            print()
        
        print(f"Финальный ответ: {chain.final_answer}")
        
    except Exception as e:
        print(f"Ошибка в логическом рассуждении: {e}")


async def coding_reasoning_example():
    """Пример решения задачи программирования"""
    print("=== Пример рассуждения при программировании ===")
    
    config = common_config
    
    client = ReasoningLLMClient(config)
    
    task = """
            Напиши функцию на Python, которая находит все простые числа до N
            используя алгоритм "Решето Эратосфена".
            """
    print(f"Задача на программирование: {task}")

    messages = [
        {
            "role": "user",
            "content": task + " Объясни каждый шаг алгоритма."
        }
    ]
    
    try:
        chain = await client.reasoning_completion(
            messages=messages,
            problem_type="coding",
            enable_streaming=False
        )
        
        print(f"Шагов планирования: {len(chain.steps)}")
        print()
        
        for step in chain.steps:
            print(f"Шаг {step.step_number}: {step.thought}")
            if step.action:
                print(f"  Код:\n{step.action}")
            print()
        
        print("Финальное решение:")
        print(chain.final_answer)
        
    except Exception as e:
        print(f"Ошибка в рассуждении о программировании: {e}")


async def streaming_reasoning_example():
    """Пример потокового рассуждения"""
    print("=== Пример потокового рассуждения ===")
    
    config = common_config
    
    client = ReasoningLLMClient(config)
    
    task = """
            Объясни пошагово, как работает алгоритм быстрой сортировки (QuickSort).
            Покажи каждый этап алгоритма на примере массива [64, 34, 25, 12, 22, 11, 90].
            """

    print(f"Вопрос по программированию: {task}")

    messages = [
        {
            "role": "user",
            "content": task
        }
    ]
    
    try:
        print("Получаем шаги рассуждения в реальном времени:")
        print()
        
        step_count = 0
        async for step in client.reasoning_completion(
            messages=messages,
            problem_type="general",
            enable_streaming=True
        ):
            step_count += 1
            print(f"Шаг {step.step_number} (получен в реальном времени):")
            print(f"  {step.thought}")
            if step.action:
                print(f"  Действие: {step.action}")
            if step.confidence:
                print(f"  Уверенность: {step.confidence:.2f}")
            print()
        
        print(f"Всего получено шагов в потоке: {step_count}")
        
    except Exception as e:
        print(f"Ошибка в потоковом рассуждении: {e}")


async def custom_reasoning_template_example():
    """Пример использования кастомного шаблона рассуждений"""
    print("=== Пример кастомного шаблона рассуждений ===")
    
    config = common_config
    
    # Кастомный шаблон для анализа текста
    custom_template = """
Проанализируй данный текст по следующей схеме:
1. Определи основную тему
2. Найди ключевые аргументы
3. Оцени логическую структуру
4. Сделай выводы о качестве текста

Формат ответа:
Шаг 1: [анализ темы]
Анализ: [что обнаружено]
Оценка: [качественная оценка]
Уверенность: [0.0-1.0]

[повтори для каждого аспекта]

Финальный ответ: [общая оценка текста]
"""
    
    reasoning_config = ReasoningConfig(
        reasoning_prompt_template=custom_template,
        max_reasoning_steps=5
    )
    
    client = ReasoningLLMClient(config, reasoning_config)
    
    task = """Проанализируй этот текст:
            
            "Искусственный интеллект революционизирует современный мир. 
            Машинное обучение позволяет компьютерам учиться на данных без явного программирования.
            Это открывает новые возможности в медицине, транспорте и образовании.
            Однако важно учитывать этические аспекты развития ИИ."
            """

    print(task)

    messages = [
        {
            "role": "user",
            "content": task
        }
    ]
    
    try:
        chain = await client.reasoning_completion(
            messages=messages,
            problem_type="general",
            enable_streaming=False
        )
        
        print("Анализ текста с кастомным шаблоном:")
        print()
        
        for step in chain.steps:
            print(f"Шаг {step.step_number}: {step.thought}")
            if step.observation:
                print(f"  Оценка: {step.observation}")
            print()
        
        print(f"Финальная оценка: {chain.final_answer}")
        
    except Exception as e:
        print(f"Ошибка в кастомном рассуждении: {e}")


async def reasoning_quality_analysis_example():
    """Пример детального анализа качества рассуждений"""
    print("=== Пример анализа качества рассуждений ===")
    
    config = common_config
    
    client = ReasoningLLMClient(config)
    
    task = """
            Реши задачу: У фермера есть куры и кролики, всего 35 голов и 94 ноги.
            Сколько кур и сколько кроликов у фермера?
            """

    print(task)

    messages = [
        {
            "role": "user",
            "content": task
        }
    ]
    
    try:
        # Выполняем рассуждение
        chain = await client.reasoning_completion(
            messages=messages,
            problem_type="math",
            enable_streaming=False
        )
        
        # Детальный анализ качества
        quality_analysis = await client.analyze_reasoning_quality(chain)
        
        print("Детальный анализ качества рассуждений:")
        print(f"Общее количество шагов: {quality_analysis['total_steps']}")
        print(f"Средняя уверенность: {quality_analysis['avg_confidence']:.2f}")
        print(f"Есть финальный ответ: {quality_analysis['has_final_answer']}")
        print(f"Полнота рассуждения: {quality_analysis['reasoning_completeness']:.2f}")
        print(f"Логическая последовательность: {quality_analysis['logical_consistency']:.2f}")
        print()
        
        print("Качество каждого шага:")
        for step_quality in quality_analysis['step_quality_scores']:
            print(f"  Шаг {step_quality['step_number']}:")
            print(f"    Есть рассуждение: {step_quality['has_thought']}")
            print(f"    Есть действие: {step_quality['has_action']}")
            print(f"    Есть наблюдение: {step_quality['has_observation']}")
            print(f"    Есть уверенность: {step_quality['has_confidence']}")
            print(f"    Оценка качества: {step_quality['quality_score']:.2f}")
            print()
        
        print("Решение задачи:")
        print(chain.final_answer)
        
    except Exception as e:
        print(f"Ошибка в анализе качества: {e}")


async def main():
    """Главная функция с примерами использования"""
    print("Демонстрация возможностей ReasoningLLMClient")
    print("=" * 50)
    print()
    
    # Запускаем все примеры
    examples = [
        ("Математическое рассуждение", math_reasoning_example),
        ("Логическое рассуждение", logic_reasoning_example),
        ("Рассуждение при программировании", coding_reasoning_example),
        ("Потоковое рассуждение", streaming_reasoning_example),
        ("Кастомный шаблон", custom_reasoning_template_example),
        ("Анализ качества", reasoning_quality_analysis_example)
    ]
    
    for name, example_func in examples:
        try:
            print(f"Запуск примера: {name}")
            await example_func()
            print("-" * 50)
            print()
        except Exception as e:
            print(f"Ошибка в примере '{name}': {e}")
            print("-" * 50)
            print()


if __name__ == "__main__":
    asyncio.run(main())