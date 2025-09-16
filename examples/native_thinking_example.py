#!/usr/bin/env python3
"""
Пример использования ReasoningLLMClient с нативными рассуждающими моделями.

Этот пример демонстрирует работу с моделями, которые поддерживают
встроенные thinking токены и параметр enable_thinking.
"""

import asyncio
import os
from dotenv import load_dotenv

from kraken_llm.client.reasoning import (
    ReasoningLLMClient, 
    ReasoningConfig, 
    ReasoningModelType
)
from kraken_llm.config.settings import LLMConfig

load_dotenv()

async def native_thinking_basic_example():
    """Базовый пример работы с нативной рассуждающей моделью"""
    print("=== Базовый пример Native Thinking ===")
    
    # Настройка для нативной рассуждающей модели
    config = LLMConfig(
        endpoint=os.getenv("LLM_REASONING_ENDPOINT"),
        api_key=os.getenv("LLM_REASONING_TOKEN"),
        model=os.getenv("LLM_REASONING_MODEL"),
        temperature=0.1
    )
    
    # Конфигурация для нативного thinking режима
    reasoning_config = ReasoningConfig(
        model_type=ReasoningModelType.NATIVE_THINKING,
        enable_thinking=True,           # Включаем thinking режим
        thinking_max_tokens=1000,       # Максимум токенов для рассуждений
        thinking_temperature=0.05,      # Низкая температура для точности
        expose_thinking=True,           # Показываем thinking блоки
        max_reasoning_steps=10
    )
    
    client = ReasoningLLMClient(config, reasoning_config)
    
    # Математическая задача
    messages = [
        {
            "role": "user",
            "content": """
            Реши задачу пошагово:
            В классе 28 учеников. 3/4 из них изучают английский язык, 
            а 2/3 изучают французский. 5 учеников изучают оба языка.
            Сколько учеников не изучают ни одного из этих языков?
            """
        }
    ]
    
    try:
        # Выполняем рассуждение с нативной моделью
        chain = await client.reasoning_completion(
            messages=messages,
            problem_type="math",
            enable_streaming=False
        )
        
        print(f"Тип модели: {chain.model_type}")
        print(f"Время рассуждения: {chain.reasoning_time:.2f}s")
        print(f"Reasoning токены: {chain.total_reasoning_tokens}")
        print(f"Уверенность: {chain.confidence_score:.2f}")
        print()
        
        # Показываем thinking блоки
        if chain.thinking_blocks:
            print("=== THINKING БЛОКИ ===")
            for i, block in enumerate(chain.thinking_blocks, 1):
                print(f"Thinking блок {i} ({block.token_count} токенов):")
                print(block.content)
                print("-" * 40)
        
        # Показываем структурированные шаги
        print("=== ШАГИ РАССУЖДЕНИЯ ===")
        for step in chain.steps:
            print(f"Шаг {step.step_number}: {step.thought}")
            if step.action:
                print(f"  Действие: {step.action}")
            if step.observation:
                print(f"  Результат: {step.observation}")
            if step.confidence:
                print(f"  Уверенность: {step.confidence:.2f}")
            if step.thinking_block:
                print(f"  Связанный thinking блок: {step.thinking_block.token_count} токенов")
            print()
        
        print(f"ФИНАЛЬНЫЙ ОТВЕТ: {chain.final_answer}")
        
    except Exception as e:
        print(f"Ошибка: {e}")


async def thinking_vs_prompt_comparison():
    """Сравнение нативного thinking и prompt-based подходов"""
    print("=== Сравнение подходов ===")
    
    config = LLMConfig(
        endpoint="http://10.129.0.37:8082",
        api_key="auth_7313ff09b5b24e529786c48f1bfc068c",
        model="universal-model"
    )
    
    # Одна и та же задача
    messages = [
        {
            "role": "user",
            "content": "Объясни, почему 0.1 + 0.2 ≠ 0.3 в компьютерных вычислениях?"
        }
    ]
    
    print("--- Prompt-based подход ---")
    
    # Prompt-based конфигурация
    prompt_config = ReasoningConfig(
        model_type=ReasoningModelType.PROMPT_BASED,
        enable_cot=True,
        reasoning_temperature=0.1,
        max_reasoning_steps=5
    )
    
    prompt_client = ReasoningLLMClient(config, prompt_config)
    
    try:
        prompt_chain = await prompt_client.reasoning_completion(
            messages=messages,
            problem_type="general",
            enable_streaming=False
        )
        
        print(f"Тип: {prompt_chain.model_type}")
        print(f"Шагов: {len(prompt_chain.steps)}")
        print(f"Thinking блоков: {len(prompt_chain.thinking_blocks) if prompt_chain.thinking_blocks else 0}")
        print(f"Ответ: {prompt_chain.final_answer[:100]}...")
        
    except Exception as e:
        print(f"Ошибка в prompt-based: {e}")
    
    print("\n--- Native thinking подход ---")
    
    # Native thinking конфигурация
    thinking_config = ReasoningConfig(
        model_type=ReasoningModelType.NATIVE_THINKING,
        enable_thinking=True,
        thinking_max_tokens=800,
        expose_thinking=True
    )
    
    thinking_client = ReasoningLLMClient(config, thinking_config)
    
    try:
        thinking_chain = await thinking_client.reasoning_completion(
            messages=messages,
            problem_type="general",
            enable_streaming=False
        )
        
        print(f"Тип: {thinking_chain.model_type}")
        print(f"Шагов: {len(thinking_chain.steps)}")
        print(f"Thinking блоков: {len(thinking_chain.thinking_blocks) if thinking_chain.thinking_blocks else 0}")
        print(f"Reasoning токены: {thinking_chain.total_reasoning_tokens}")
        print(f"Ответ: {thinking_chain.final_answer[:100]}...")
        
        # Показываем thinking блоки
        if thinking_chain.thinking_blocks:
            print("\nThinking процесс:")
            for block in thinking_chain.thinking_blocks:
                print(f"- {block.content[:80]}... ({block.token_count} токенов)")
        
    except Exception as e:
        print(f"Ошибка в native thinking: {e}")


async def thinking_streaming_example():
    """Пример потокового режима с thinking моделью"""
    print("=== Потоковый thinking режим ===")
    
    config = LLMConfig(
        endpoint="http://10.129.0.37:8082",
        api_key="auth_7313ff09b5b24e529786c48f1bfc068c",
        model="thinking-model"
    )
    
    reasoning_config = ReasoningConfig(
        model_type=ReasoningModelType.NATIVE_THINKING,
        enable_thinking=True,
        expose_thinking=True
    )
    
    client = ReasoningLLMClient(config, reasoning_config)
    
    messages = [
        {
            "role": "user",
            "content": """
            Разработай алгоритм для поиска кратчайшего пути в графе.
            Объясни каждый шаг и приведи псевдокод.
            """
        }
    ]
    
    try:
        print("Получение thinking процесса в реальном времени:")
        print()
        
        step_count = 0
        async for step in client.reasoning_completion(
            messages=messages,
            problem_type="coding",
            enable_streaming=True
        ):
            step_count += 1
            print(f"Шаг {step.step_number} (получен в реальном времени):")
            print(f"  Мысль: {step.thought[:80]}...")
            
            if step.thinking_block:
                print(f"  Thinking: {step.thinking_block.content[:60]}...")
                print(f"  Токены: {step.thinking_block.token_count}")
            
            if step.action:
                print(f"  Действие: {step.action[:50]}...")
            
            if step.confidence:
                print(f"  Уверенность: {step.confidence:.2f}")
            
            print()
            
            # Ограничиваем для демонстрации
            if step_count >= 5:
                break
        
        print(f"Получено шагов в потоке: {step_count}")
        
    except Exception as e:
        print(f"Ошибка в потоковом режиме: {e}")


async def thinking_configuration_examples():
    """Примеры различных конфигураций thinking режима"""
    print("=== Примеры конфигураций thinking ===")
    
    config = LLMConfig(
        endpoint="http://10.129.0.37:8082",
        api_key="auth_7313ff09b5b24e529786c48f1bfc068c",
        model="thinking-model"
    )
    
    # Конфигурация 1: Максимальная детализация
    detailed_config = ReasoningConfig(
        model_type=ReasoningModelType.NATIVE_THINKING,
        enable_thinking=True,
        thinking_max_tokens=2000,       # Много токенов для детального анализа
        thinking_temperature=0.0,       # Детерминированность
        expose_thinking=True,           # Показываем все thinking
        max_reasoning_steps=15
    )
    
    print("Конфигурация 1: Максимальная детализация")
    print(f"- thinking_max_tokens: {detailed_config.thinking_max_tokens}")
    print(f"- thinking_temperature: {detailed_config.thinking_temperature}")
    print(f"- expose_thinking: {detailed_config.expose_thinking}")
    
    # Конфигурация 2: Быстрый режим
    fast_config = ReasoningConfig(
        model_type=ReasoningModelType.NATIVE_THINKING,
        enable_thinking=True,
        thinking_max_tokens=300,        # Ограниченные рассуждения
        thinking_temperature=0.3,       # Больше креативности
        expose_thinking=False,          # Скрываем thinking для скорости
        max_reasoning_steps=5
    )
    
    print("\nКонфигурация 2: Быстрый режим")
    print(f"- thinking_max_tokens: {fast_config.thinking_max_tokens}")
    print(f"- thinking_temperature: {fast_config.thinking_temperature}")
    print(f"- expose_thinking: {fast_config.expose_thinking}")
    
    # Конфигурация 3: Сбалансированный режим
    balanced_config = ReasoningConfig(
        model_type=ReasoningModelType.NATIVE_THINKING,
        enable_thinking=True,
        thinking_max_tokens=800,        # Умеренное количество токенов
        thinking_temperature=0.1,       # Низкая температура
        expose_thinking=True,           # Показываем thinking
        require_step_validation=True,   # Валидация шагов
        max_reasoning_steps=8
    )
    
    print("\nКонфигурация 3: Сбалансированный режим")
    print(f"- thinking_max_tokens: {balanced_config.thinking_max_tokens}")
    print(f"- thinking_temperature: {balanced_config.thinking_temperature}")
    print(f"- require_step_validation: {balanced_config.require_step_validation}")
    
    # Демонстрация использования разных конфигураций
    messages = [{"role": "user", "content": "Что такое машинное обучение?"}]
    
    for name, config_obj in [
        ("Детализированный", detailed_config),
        ("Быстрый", fast_config),
        ("Сбалансированный", balanced_config)
    ]:
        print(f"\n--- Тест конфигурации: {name} ---")
        
        client = ReasoningLLMClient(config, config_obj)
        
        # Показываем подготовленные параметры
        params = client._prepare_thinking_params(messages)
        print("Параметры API:")
        for key, value in params.items():
            if 'thinking' in key or key == 'enable_thinking':
                print(f"  {key}: {value}")


async def thinking_token_analysis():
    """Анализ использования thinking токенов"""
    print("=== Анализ thinking токенов ===")
    
    config = LLMConfig(
        endpoint="http://10.129.0.37:8082",
        api_key="auth_7313ff09b5b24e529786c48f1bfc068c",
        model="thinking-model"
    )
    
    reasoning_config = ReasoningConfig(
        model_type=ReasoningModelType.NATIVE_THINKING,
        enable_thinking=True,
        thinking_max_tokens=1000,
        expose_thinking=True
    )
    
    client = ReasoningLLMClient(config, reasoning_config)
    
    # Тестируем подсчет токенов для разных типов текста
    test_texts = [
        "Простой текст",
        "Более сложный текст с математическими формулами: E = mc²",
        """
        Длинный thinking блок с детальным анализом проблемы.
        Здесь модель может рассуждать о различных аспектах задачи,
        анализировать возможные подходы к решению и выбирать
        наиболее подходящий алгоритм для конкретной ситуации.
        """,
        "Код: def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
    ]
    
    print("Анализ подсчета токенов:")
    for i, text in enumerate(test_texts, 1):
        token_count = client._count_tokens(text)
        chars = len(text)
        ratio = chars / token_count if token_count > 0 else 0
        
        print(f"\nТекст {i}:")
        print(f"  Символов: {chars}")
        print(f"  Токенов: {token_count}")
        print(f"  Соотношение символы/токен: {ratio:.1f}")
        print(f"  Текст: {text[:50]}...")
    
    # Демонстрация экономии токенов
    print("\n=== Сравнение эффективности ===")
    
    # Prompt-based подход (много токенов в промпте)
    prompt_tokens_estimate = client._count_tokens("""
    Реши эту задачу пошагово. Для каждого шага:
    1. Объясни, что ты делаешь
    2. Покажи вычисления
    3. Проверь результат
    4. Оцени уверенность в шаге (0-1)
    
    Формат ответа:
    Шаг 1: [объяснение]
    Вычисление: [формула и расчет]
    Результат: [результат шага]
    Уверенность: [0.0-1.0]
    """)
    
    # Native thinking подход (токены только в thinking блоке)
    thinking_tokens_estimate = client._count_tokens(
        "Реши эту задачу, показав свои рассуждения."
    )
    
    print(f"Prompt-based промпт: ~{prompt_tokens_estimate} токенов")
    print(f"Native thinking промпт: ~{thinking_tokens_estimate} токенов")
    print(f"Экономия в промпте: {prompt_tokens_estimate - thinking_tokens_estimate} токенов")
    print("\nПреимущества native thinking:")
    print("- Более короткие промпты")
    print("- Thinking токены обрабатываются эффективнее")
    print("- Модель может рассуждать более естественно")
    print("- Лучший контроль над процессом рассуждения")


async def main():
    """Главная функция с примерами использования нативных thinking моделей"""
    print("Демонстрация Native Thinking Models")
    print("=" * 50)
    print()
    
    examples = [
        ("Базовый пример", native_thinking_basic_example),
        ("Сравнение подходов", thinking_vs_prompt_comparison),
        ("Потоковый режим", thinking_streaming_example),
        ("Конфигурации", thinking_configuration_examples),
        ("Анализ токенов", thinking_token_analysis)
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