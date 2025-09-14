"""
Демонстрация всех режимов structured output в Kraken LLM.

Этот скрипт демонстрирует:
1. OpenAI нативный structured output (non-streaming)
2. OpenAI нативный structured output (streaming)
3. Outlines structured output (non-streaming)
4. Outlines structured output (реальный streaming с инкрементальным парсингом)
5. Сравнение производительности всех режимов
6. Параллельное выполнение запросов
"""

import asyncio
import os
import time
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from kraken_llm.client.structured import StructuredLLMClient
from kraken_llm.config.settings import LLMConfig

load_dotenv()


class TaskResponse(BaseModel):
    """Модель ответа для задачи."""
    task_name: str = Field(description="Название задачи")
    priority: int = Field(ge=1, le=5, description="Приоритет от 1 до 5")
    estimated_hours: float = Field(
        ge=0.5, le=40.0, description="Оценка времени в часах")
    tags: List[str] = Field(description="Теги задачи")
    is_urgent: bool = Field(description="Флаг срочности")
    description: Optional[str] = Field(
        default=None, description="Описание задачи")


class ProjectSummary(BaseModel):
    """Модель резюме проекта."""
    project_name: str = Field(description="Название проекта")
    tasks: List[TaskResponse] = Field(description="Список задач")
    total_estimated_hours: float = Field(ge=0.0, description="Общее время")
    completion_percentage: float = Field(
        ge=0.0, le=100.0, description="Процент завершения")
    next_milestone: str = Field(description="Следующий этап")


async def demo_simple_task(client: StructuredLLMClient, mode_name: str):
    """Демонстрация простой задачи."""
    print(f"\n=== {mode_name} режим - Простая задача ===")

    messages = [
        {
            "role": "system",
            "content": "Ты помощник по управлению проектами. Создавай структурированные ответы о задачах."
        },
        {
            "role": "user",
            "content": "Создай задачу 'Написать документацию' с приоритетом 3, оценкой 8 часов, тегами ['docs', 'writing'] и пометь как срочную"
        }
    ]

    try:
        result = await client.chat_completion(
            messages=messages,
            response_model=TaskResponse,
            stream=False
        )

        print(f"Задача: {result.task_name}")
        print(f"Приоритет: {result.priority}")
        print(f"Время: {result.estimated_hours} часов")
        print(f"Теги: {result.tags}")
        print(f"Срочная: {'Да' if result.is_urgent else 'Нет'}")
        if result.description:
            print(f"Описание: {result.description}")

    except Exception as e:
        print(f"Ошибка в {mode_name} режиме: {e}")


async def demo_complex_project(client: StructuredLLMClient, mode_name: str):
    """Демонстрация сложного проекта."""
    print(f"\n=== {mode_name} режим - Сложный проект ===")

    messages = [
        {
            "role": "system",
            "content": "Ты помощник по управлению проектами. Создавай детальные резюме проектов."
        },
        {
            "role": "user",
            "content": """Создай резюме проекта 'Веб-приложение' с тремя задачами:
            1. Дизайн UI (приоритет 2, 12 часов, теги ['design', 'ui'])
            2. Backend API (приоритет 1, 20 часов, теги ['backend', 'api'], срочная)
            3. Тестирование (приоритет 3, 8 часов, теги ['testing', 'qa'])
            
            Общее время 40 часов, завершено 25%, следующий этап 'Разработка MVP'"""
        }
    ]

    try:
        result = await client.chat_completion(
            messages=messages,
            response_model=ProjectSummary,
            stream=False
        )

        print(f"Проект: {result.project_name}")
        print(f"Общее время: {result.total_estimated_hours} часов")
        print(f"Завершено: {result.completion_percentage}%")
        print(f"Следующий этап: {result.next_milestone}")
        print(f"Задачи ({len(result.tasks)}):")

        for i, task in enumerate(result.tasks, 1):
            print(f"  {i}. {task.task_name}")
            print(
                f"     Приоритет: {task.priority}, Время: {task.estimated_hours}ч")
            print(
                f"     Теги: {task.tags}, Срочная: {'Да' if task.is_urgent else 'Нет'}")

    except Exception as e:
        print(f"Ошибка в {mode_name} режиме: {e}")


async def demo_all_modes_comparison(openai_client: StructuredLLMClient, outlines_client: StructuredLLMClient):
    """Демонстрация всех режимов structured output."""
    print(f"\n=== Сравнение всех режимов Structured Output ===")

    messages = [
        {
            "role": "system",
            "content": "Создай задачу в структурированном формате с полной информацией."
        },
        {
            "role": "user",
            "content": "Создай задачу 'Код-ревью API' с приоритетом 2, оценкой 3 часа, тегами ['review', 'api'] и пометь как срочную"
        }
    ]

    results = {}

    # 1. OpenAI Non-streaming
    print("\n1) OpenAI Non-streaming:")
    try:
        start_time = time.time()
        result = await openai_client.chat_completion(
            messages=messages,
            response_model=TaskResponse,
            stream=False
        )
        execution_time = time.time() - start_time
        results["openai_non_stream"] = {
            "time": execution_time, "result": result, "success": True}
        print(f"   Успешно выполнено за {execution_time:.3f}s: {result.task_name}")
        print(
            f"      Приоритет: {result.priority}, Время: {result.estimated_hours}ч")
    except Exception as e:
        results["openai_non_stream"] = {
            "time": 0, "success": False, "error": str(e)}
        print(f"   ❌ Ошибка: {e}")

    # 2. OpenAI Streaming
    print("\n2) OpenAI Streaming:")
    try:
        start_time = time.time()
        result = await openai_client.chat_completion(
            messages=messages,
            response_model=TaskResponse,
            stream=True
        )
        execution_time = time.time() - start_time
        results["openai_stream"] = {
            "time": execution_time, "result": result, "success": True}
        print(f"   Успешно выполнено за {execution_time:.3f}s: {result.task_name}")
        print(
            f"      Приоритет: {result.priority}, Время: {result.estimated_hours}ч")
    except Exception as e:
        results["openai_stream"] = {"time": 0,
                                    "success": False, "error": str(e)}
        print(f"   ❌ Ошибка: {e}")

    # 3. Outlines Non-streaming
    print("\n3) Outlines Non-streaming:")
    try:
        start_time = time.time()
        result = await outlines_client.chat_completion(
            messages=messages,
            response_model=TaskResponse,
            stream=False
        )
        execution_time = time.time() - start_time
        results["outlines_non_stream"] = {
            "time": execution_time, "result": result, "success": True}
        print(f"   Успешно выполнено за {execution_time:.3f}s: {result.task_name}")
        print(
            f"      Приоритет: {result.priority}, Время: {result.estimated_hours}ч")
    except Exception as e:
        results["outlines_non_stream"] = {
            "time": 0, "success": False, "error": str(e)}
        print(f"   ❌ Ошибка: {e}")

    # 4. Outlines Real Streaming
    print("\n4) Outlines Real Streaming (инкрементальный парсинг):")
    try:
        start_time = time.time()
        result = await outlines_client.chat_completion(
            messages=messages,
            response_model=TaskResponse,
            stream=True  # Теперь это реальный streaming!
        )
        execution_time = time.time() - start_time
        results["outlines_stream"] = {
            "time": execution_time, "result": result, "success": True}
        print(f"   Успешно выполнено за {execution_time:.3f}s: {result.task_name}")
        print(
            f"      Приоритет: {result.priority}, Время: {result.estimated_hours}ч")
    except Exception as e:
        results["outlines_stream"] = {
            "time": 0, "success": False, "error": str(e)}
        print(f"   ❌ Ошибка: {e}")

    # Анализ результатов
    print(f"\n   Анализ производительности:")
    successful_results = {k: v for k, v in results.items() if v["success"]}

    if len(successful_results) > 1:
        times = [(k, v["time"]) for k, v in successful_results.items()]
        times.sort(key=lambda x: x[1])

        print(f"   Рейтинг по скорости:")
        for i, (mode, exec_time) in enumerate(times, 1):
            mode_names = {
                "openai_non_stream": "OpenAI Non-streaming",
                "openai_stream": "OpenAI Streaming",
                "outlines_non_stream": "Outlines Non-streaming",
                "outlines_stream": "Outlines Real Streaming"
            }
            print(f"   {i}. {mode_names[mode]}: {exec_time:.3f}s")

        # Сравнение streaming vs non-streaming
        if "openai_stream" in successful_results and "openai_non_stream" in successful_results:
            openai_improvement = successful_results["openai_non_stream"]["time"] - \
                successful_results["openai_stream"]["time"]
            print(
                f"\n   OpenAI Streaming vs Non-streaming: {openai_improvement:+.3f}s")

        if "outlines_stream" in successful_results and "outlines_non_stream" in successful_results:
            outlines_improvement = successful_results["outlines_non_stream"]["time"] - \
                successful_results["outlines_stream"]["time"]
            print(
                f"   Outlines Streaming vs Non-streaming: {outlines_improvement:+.3f}s")

    return results


async def demo_concurrent_modes():
    """Демонстрация параллельного выполнения в разных режимах."""
    print(f"\n=== Параллельное выполнение в разных режимах ===")

    # Создаем клиенты для разных режимов
    openai_config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL"),
        temperature=0.2,
        max_tokens=500,
        outlines_so_mode=False
    )

    outlines_config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL"),
        temperature=0.2,
        max_tokens=500,
        outlines_so_mode=True
    )

    openai_client = StructuredLLMClient(openai_config)
    outlines_client = StructuredLLMClient(outlines_config)

    # Разные запросы для параллельного выполнения
    requests = [
        {"client": openai_client, "mode": "OpenAI",
            "stream": False, "task": "Разработка UI"},
        {"client": openai_client, "mode": "OpenAI",
            "stream": True, "task": "Тестирование API"},
        {"client": outlines_client, "mode": "Outlines",
            "stream": False, "task": "Код-ревью"},
        {"client": outlines_client, "mode": "Outlines",
            "stream": True, "task": "Документация"}
    ]

    async def single_request(req, index):
        messages = [{
            "role": "user",
            "content": f"Создай задачу '{req['task']}' с приоритетом {index + 1} и оценкой {(index + 1) * 2} часа"
        }]

        start_time = time.time()
        try:
            result = await req["client"].chat_completion(
                messages=messages,
                response_model=TaskResponse,
                stream=req["stream"]
            )

            execution_time = time.time() - start_time
            return {
                "index": index + 1,
                "mode": req["mode"],
                "stream": req["stream"],
                "time": execution_time,
                "result": result,
                "success": True
            }
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "index": index + 1,
                "mode": req["mode"],
                "stream": req["stream"],
                "time": execution_time,
                "error": str(e),
                "success": False
            }

    print(f"Запускаем {len(requests)} параллельных запросов...")

    start_time = time.time()
    results = await asyncio.gather(*[single_request(req, i) for i, req in enumerate(requests)])
    total_time = time.time() - start_time

    print(f"Общее время выполнения: {total_time:.3f}s")

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"Успешных: {len(successful)}/{len(results)}")

    for result in successful:
        stream_label = "Streaming" if result["stream"] else "Non-streaming"
        print(
            f"   #{result['index']}: {result['mode']} {stream_label} - {result['time']:.3f}s")
        print(f"      Задача: {result['result'].task_name}")

    for result in failed:
        stream_label = "Streaming" if result["stream"] else "Non-streaming"
        print(
            f"   #{result['index']}: {result['mode']} {stream_label} - ❌ {result['error'][:50]}...")

    if successful:
        sequential_time = sum(r["time"] for r in successful)
        speedup = sequential_time / total_time
        print(f"\nУскорение от параллелизма: {speedup:.1f}x")

    # Закрываем клиенты
    await openai_client.close()
    await outlines_client.close()

    return len(successful) == len(results)


async def demo_advanced_features():
    """Демонстрация продвинутых возможностей."""
    print(f"\n=== Продвинутые возможности ===")

    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL"),
        temperature=0.3,
        max_tokens=800,
        outlines_so_mode=True  # Используем Outlines для демонстрации реального streaming
    )

    client = StructuredLLMClient(config)

    # Демонстрация сложной модели с вложенными объектами
    print("\n🔧 Сложная модель с вложенными объектами:")

    messages = [{
        "role": "user",
        "content": """Создай проект 'E-commerce Platform' с задачами:
        - Frontend (React, приоритет 2, 25 часов, срочная)
        - Backend (Node.js, приоритет 1, 30 часов, срочная) 
        - Database (PostgreSQL, приоритет 3, 15 часов)
        
        Общее время 70 часов, завершено 15%, следующий этап 'MVP разработка'"""
    }]

    try:
        start_time = time.time()
        result = await client.chat_completion(
            messages=messages,
            response_model=ProjectSummary,
            stream=True  # Реальный streaming для сложной модели!
        )
        execution_time = time.time() - start_time

        print(f"   Успешно выполнено за {execution_time:.3f}s")
        print(f"   Проект: {result.project_name}")
        print(f"   Общее время: {result.total_estimated_hours}ч")
        print(f"   Прогресс: {result.completion_percentage}%")
        print(f"   Следующий этап: {result.next_milestone}")
        print(f"   Задач: {len(result.tasks)}")

        for i, task in enumerate(result.tasks, 1):
            print(
                f"       {i}. {task.task_name} (P{task.priority}, {task.estimated_hours}ч)")

    except Exception as e:
        print(f"   ❌ Ошибка: {e}")

    await client.close()


async def main():
    """Основная функция демонстрации."""
    print("ПОЛНАЯ ДЕМОНСТРАЦИЯ РЕЖИМОВ STRUCTURED OUTPUT")
    print("=" * 65)
    print("Демонстрируем все возможности Kraken LLM structured output:")
    print("• OpenAI нативный (non-streaming & streaming)")
    print("• Outlines (non-streaming & реальный streaming)")
    print("• Сравнение производительности")
    print("• Параллельное выполнение")
    print("• Сложные модели с вложенными объектами")
    print()

    # Показываем конфигурацию
    print(f"Конфигурация:")
    print(f"Endpoint: {os.getenv('LLM_ENDPOINT')}")
    print(f"Model: {os.getenv('LLM_MODEL')}")
    print()

    # Конфигурации для разных режимов
    openai_config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL"),
        temperature=0.1,
        max_tokens=1000,
        outlines_so_mode=False  # Нативный OpenAI режим
    )

    outlines_config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL"),
        temperature=0.1,
        max_tokens=1000,
        outlines_so_mode=True   # Outlines режим с реальным streaming
    )

    # Создание клиентов
    openai_client = StructuredLLMClient(openai_config)
    outlines_client = StructuredLLMClient(outlines_config)

    try:
        # 1. Демонстрация простых задач в разных режимах
        await demo_simple_task(openai_client, "OpenAI")
        await demo_simple_task(outlines_client, "Outlines")

        # 2. Демонстрация сложных проектов
        await demo_complex_project(openai_client, "OpenAI")
        await demo_complex_project(outlines_client, "Outlines")

        # 3. Сравнение всех режимов
        comparison_results = await demo_all_modes_comparison(openai_client, outlines_client)

        # 4. Параллельное выполнение
        concurrent_success = await demo_concurrent_modes()

        # 5. Продвинутые возможности
        await demo_advanced_features()

        # Итоговая статистика
        print(f"\n" + "=" * 65)
        print(f"ИТОГОВАЯ СТАТИСТИКА")
        print(f"=" * 65)

        successful_modes = sum(
            1 for result in comparison_results.values() if result["success"])
        total_modes = len(comparison_results)

        print(f"Режимы structured output: {successful_modes}/{total_modes} работают")
        print(f"Параллельное выполнение: {'Работает' if concurrent_success else 'Проблемы'}")

        if successful_modes == total_modes:
            print(f"\nВСЕ РЕЖИМЫ РАБОТАЮТ")
            print(f"Особенности:")
            print(f"   • OpenAI нативный structured output")
            print(f"   • OpenAI streaming с агрегацией")
            print(f"   • Outlines с улучшенными промптами")
            print(f"   • Outlines с РЕАЛЬНЫМ streaming и инкрементальным парсингом")
            print(f"   • Автоматическое переключение между режимами")
            print(f"   • Поддержка сложных вложенных моделей")
            print(f"   • Параллельное выполнение запросов")
        else:
            print(f"\n⚠️  Некоторые режимы требуют внимания")

    except Exception as e:
        print(f"\n❌ Критическая ошибка во время демонстрации: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Закрытие клиентов
        await openai_client.close()
        await outlines_client.close()


if __name__ == "__main__":
    asyncio.run(main())
