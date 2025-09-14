#!/usr/bin/env python3
"""
Пример использования клиентов библиотеки Kraken LLM.

Демонстрирует:
1. Reasoning клиент для пошагового рассуждения
2. Multimodal клиент для работы с изображениями
3. Adaptive клиент с автоопределением возможностей
"""

import asyncio
from pathlib import Path
import os
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

from kraken_llm import (
    LLMConfig,
    ReasoningLLMClient,
    ReasoningConfig,
    MultimodalLLMClient,
    MultimodalConfig,
    AdaptiveLLMClient,
    AdaptiveConfig
)

load_dotenv()

# Модель для structured режима
class WeatherInfo(BaseModel):
    city: str
    temperature: int
    condition: str
    humidity: Optional[int] = None
    wind_speed: Optional[int] = None

async def demo_reasoning_client():
    """
    Демонстрация клиента рассуждений.

    ReasoningLLMClient поддерживает:
    - Chain of Thought (CoT) рассуждения
    - Пошаговый анализ задач
    - Streaming режим для рассуждений в реальном времени
    - Анализ качества рассуждений
    """
    print("=== Демонстрация ReasoningLLMClient ===")
    print("Клиент для пошагового рассуждения с поддержкой Chain of Thought")

    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL")
    )

    reasoning_config = ReasoningConfig(
        enable_cot=True,
        max_reasoning_steps=5,
        reasoning_temperature=0.1,
        extract_confidence=True
    )

    client = ReasoningLLMClient(config, reasoning_config)
    task1 = "В магазине было 150 яблок. Продали 60% от общего количества. Сколько яблок осталось?"
    # Математическая задача
    print(f"\n1. Решение математической задачи:\n{task1}\n")
    messages = [{
        "role": "user",
        "content": "Реши задачу: " + task1
    }]

    try:
        # Сначала пробуем streaming режим
        print("Рассуждение в реальном времени:")
        reasoning_steps = []

        # Проверяем, возвращает ли метод async generator
        result = client.reasoning_completion(
            messages=messages,
            problem_type="math",
            enable_streaming=True
        )

        # Если это async generator, обрабатываем как поток
        if hasattr(result, '__aiter__'):
            async for step in result:
                reasoning_steps.append(step)
                print(f"\nШаг {len(reasoning_steps)}:")
                print(f"  Мысль: {step.thought}")
                if hasattr(step, 'action') and step.action:
                    print(f"  Действие: {step.action}")
                if hasattr(step, 'observation') and step.observation:
                    print(f"  Результат: {step.observation}")
                if hasattr(step, 'confidence') and step.confidence:
                    print(f"  Уверенность: {step.confidence:.2f}")

                # Добавляем небольшую задержку для демонстрации streaming эффекта
                await asyncio.sleep(0.1)
        else:
            # Если это обычный объект, ждем его
            reasoning_chain = await result
            print(
                f"Получена цепочка рассуждений с {len(reasoning_chain.steps)} шагами")

            for i, step in enumerate(reasoning_chain.steps, 1):
                print(f"\nШаг {i}:")
                print(f"  Мысль: {step.thought}")
                if hasattr(step, 'action') and step.action:
                    print(f"  Действие: {step.action}")
                if hasattr(step, 'observation') and step.observation:
                    print(f"  Результат: {step.observation}")
                if hasattr(step, 'confidence') and step.confidence:
                    print(f"  Уверенность: {step.confidence:.2f}")

            print(f"\nФинальный ответ: {reasoning_chain.final_answer}")

    except Exception as e:
        print(f"Ошибка в reasoning completion: {e}")
        # Fallback к обычному режиму
        try:
            response = await client.chat_completion(messages)
            print(f"Обычный ответ: {response}")
        except Exception as fallback_error:
            print(f"Fallback также не сработал: {fallback_error}")

    # Логическая задача
    task2 = "Все кошки - млекопитающие. Все млекопитающие - животные. Мурка - кошка. Что можно сказать о Мурке?"
    print(f"\n2. Решение логической задачи:\n{task2}\n")
    logic_messages = [{
        "role": "user",
        "content": "Дано: " + task2
    }]

    try:
        print("Логическое рассуждение в реальном времени:")
        logic_steps = []

        # Проверяем тип возвращаемого значения
        result = client.reasoning_completion(
            messages=logic_messages,
            problem_type="logic",
            enable_streaming=True
        )

        if hasattr(result, '__aiter__'):
            async for step in result:
                logic_steps.append(step)
                print(f"\nШаг {len(logic_steps)}:")
                print(f"  Логический вывод: {step.thought}")
                if hasattr(step, 'confidence') and step.confidence:
                    print(f"  Уверенность: {step.confidence:.2f}")

                # Добавляем небольшую задержку для демонстрации streaming эффекта
                await asyncio.sleep(0.1)
        else:
            reasoning_chain = await result
            print(
                f"Получена цепочка логических рассуждений с {len(reasoning_chain.steps)} шагами")

            for i, step in enumerate(reasoning_chain.steps, 1):
                print(f"\nШаг {i}:")
                print(f"  Логический вывод: {step.thought}")
                if hasattr(step, 'confidence') and step.confidence:
                    print(f"  Уверенность: {step.confidence:.2f}")

            print(f"\nФинальный вывод: {reasoning_chain.final_answer}")

        print(
            f"\nЛогическое рассуждение завершено за {len(logic_steps) if logic_steps else len(reasoning_chain.steps)} шагов")

    except Exception as e:
        print(f"Ошибка в логическом рассуждении: {e}")
        # Fallback к обычному режиму
        try:
            response = await client.chat_completion(logic_messages)
            print(f"Обычный ответ: {response}")
        except Exception as fallback_error:
            print(f"Fallback также не сработал: {fallback_error}")


async def demo_multimodal_client():
    """
    Демонстрация мультимодального клиента.

    MultimodalLLMClient поддерживает:
    - Анализ изображений (vision completion)
    - Работу с аудио файлами
    - Автоматическое изменение размера изображений
    - Валидацию медиа файлов
    """
    print("\n=== Демонстрация MultimodalLLMClient ===")
    print("Клиент для работы с изображениями, аудио и видео")

    config = LLMConfig()
    multimodal_config = MultimodalConfig(
        max_image_size=10 * 1024 * 1024,  # 10MB
        auto_resize_images=True
    )

    client = MultimodalLLMClient(config, multimodal_config)

    # Создаем тестовое изображение (простой PNG 1x1 пиксель)
    test_image_path = Path("test_image.png")

    # Создаем минимальное PNG изображение
    png_data = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
        "890000000a4944415478da6300010000050001d72cc82f0000000049454e44ae426082"
    )

    try:
        # Сохраняем тестовое изображение
        with open(test_image_path, 'wb') as f:
            f.write(png_data)

        print("1. Анализ изображения:")

        try:
            response = await client.vision_completion(
                text_prompt="Опиши что ты видишь на этом изображении",
                images=[test_image_path],
                detail_level="high"
            )

            print(f"Ответ модели: {response}")

        except Exception as e:
            print(f"Vision completion не поддерживается: {e}")
            print(
                "Примечание: Для работы с изображениями нужна модель с поддержкой vision (например, Qwen2.5-VL)")

        # Демонстрация поддерживаемых форматов
        print("\n2. Поддерживаемые форматы:")
        formats = MultimodalLLMClient.get_supported_formats()
        for media_type, format_list in formats.items():
            print(f"  {media_type}: {', '.join(format_list)}")

        # Создание контента для изображения
        print("\n3. Создание контента изображения:")
        image_content = MultimodalLLMClient.create_image_url_content(
            test_image_path,
            detail="low"
        )
        print(f"Тип контента: {image_content['type']}")
        print(f"URL начинается с: {image_content['image_url']['url'][:50]}...")

    finally:
        # Удаляем тестовое изображение
        if test_image_path.exists():
            test_image_path.unlink()


async def demo_adaptive_client():
    """
    Демонстрация адаптивного клиента.

    AdaptiveLLMClient автоматически:
    - Определяет возможности модели
    - Выбирает оптимальный режим работы
    - Предоставляет fallback механизмы
    - Собирает метрики производительности
    """
    print("\n=== Демонстрация AdaptiveLLMClient ===")
    print("Клиент с автоопределением возможностей и выбором оптимального режима")

    config = LLMConfig()
    adaptive_config = AdaptiveConfig(
        auto_fallback=True,
        prefer_streaming=False,
        enable_performance_tracking=True
    )

    client = AdaptiveLLMClient(config, adaptive_config)

    # Определение возможностей модели
    print("1. Определение возможностей модели:")
    try:
        capabilities = await client.get_model_capabilities()
        print(
            f"Обнаруженные возможности: {[cap.value for cap in capabilities]}")

    except Exception as e:
        print(f"Ошибка определения возможностей: {e}")
        return

    # Умный completion
    print("\n2. Умный completion (автовыбор режима):")

    test_cases = [
        {
            "name": "Обычный вопрос",
            "messages": [{"role": "user", "content": "Привет! Как дела?"}],
            "expected_mode": "standard"
        },
        {
            "name": "Запрос на рассуждение",
            "messages": [{"role": "user", "content": "Объясни пошагово, как работает фотосинтез"}],
            "expected_mode": "reasoning"
        },
        {
            "name": "Запрос JSON",
            "messages": [{"role": "user", "content": "Верни JSON с информацией о погоде в Москве"}],
            "expected_mode": "structured",
            "response_model": WeatherInfo
        }
    ]

    for test_case in test_cases:
        print(f"\n  Тест: {test_case['name']}")
        try:
            # Подготавливаем параметры
            completion_params = {
                "messages": test_case["messages"],
                "max_tokens": 1000,
                "preferred_mode": test_case["expected_mode"]
            }

            # Для structured режима добавляем response_model
            if test_case["expected_mode"] == "structured" and "response_model" in test_case:
                completion_params["response_model"] = test_case["response_model"]

            response = await client.smart_completion(**completion_params)
            print(f"  Ответ: {response}")

        except Exception as e:
            print(f"  Ошибка: {e}")

    # Отчет о производительности
    print("\n3. Отчет о производительности:")
    performance_report = client.get_performance_report()

    if performance_report["model_info"]:
        model_info = performance_report["model_info"]
        print(f"  Модель: {model_info['name']}")
        print(f"  Провайдер: {model_info['provider']}")
        print(f"  Возможности: {len(model_info['capabilities'])}")

    if performance_report["performance_metrics"]:
        print("  Метрики производительности:")
        for mode, metrics in performance_report["performance_metrics"].items():
            print(f"    {mode}:")
            print(f"      Запросов: {metrics['total_requests']}")
            print(f"      Успешность: {metrics['success_rate']:.2%}")
            print(f"      Средняя задержка: {metrics['avg_latency']:.3f}s")

    # Тест fallback механизма
    print("\n4. Тест fallback механизма:")
    try:
        # Пробуем запрос с несуществующими параметрами
        response = await client.smart_completion(
            messages=[{"role": "user", "content": "Тест fallback"}],
            preferred_mode="nonexistent_mode"
        )
        print(f"  Fallback сработал: {response}")

    except Exception as e:
        print(f"  Fallback не сработал: {e}")


async def demo_streaming_comparison():
    """Сравнение обычного и streaming режимов"""
    print("\n=== Сравнение режимов ===")
    print("Сравнение производительности обычного и streaming режимов")

    config = LLMConfig()
    adaptive_client = AdaptiveLLMClient(config)

    test_prompt = "Расскажи короткую историю про кота"
    messages = [{"role": "user", "content": test_prompt}]

    # Обычный режим
    print("1. Обычный режим:")
    import time
    start_time = time.time()

    try:
        response = await adaptive_client.smart_completion(messages, max_tokens=100)
        normal_time = time.time() - start_time
        print(f"  Время: {normal_time:.2f}s")
        print(f"  Ответ: {response[:100]}...")

    except Exception as e:
        print(f"  Ошибка: {e}")
        normal_time = 0

    # Streaming режим
    print("\n2. Streaming режим:")
    print("  Ответ в реальном времени: ", end='', flush=True)
    start_time = time.time()

    try:
        chunks = []
        full_response = ""

        async for chunk in adaptive_client.chat_completion_stream(messages, max_tokens=100):
            chunks.append(chunk)
            full_response += chunk
            print(chunk, end='', flush=True)

        streaming_time = time.time() - start_time
        print(f"\n  Время: {streaming_time:.2f}s")
        print(f"  Чанков: {len(chunks)}")
        print(f"  Полный ответ: {len(full_response)} символов")

        if normal_time > 0:
            print(f"  Разница: {abs(streaming_time - normal_time):.2f}s")

    except Exception as e:
        print(f"  Ошибка: {e}")


async def main():
    """Главная функция демонстрации"""
    print("🚀 Демонстрация клиентов Kraken LLM")
    print("=" * 60)

    try:
        # Демонстрация reasoning клиента
        await demo_reasoning_client()

        # Демонстрация multimodal клиента
        await demo_multimodal_client()

        # Демонстрация adaptive клиента
        await demo_adaptive_client()

        # Сравнение режимов
        await demo_streaming_comparison()

    except Exception as e:
        print(f"Общая ошибка: {e}")
        import traceback
        traceback.print_exc()

    print("\n✅ Демонстрация завершена!")
    print("\nНовые возможности:")
    print("- 🧠 Reasoning клиент для пошагового рассуждения")
    print("- 🖼️  Multimodal клиент для работы с изображениями")
    print("- 🤖 Adaptive клиент с автоопределением возможностей")
    print("- 📊 Метрики производительности и fallback механизмы")
    print("- 🔄 Автоматический выбор оптимального режима")


if __name__ == "__main__":
    asyncio.run(main())
