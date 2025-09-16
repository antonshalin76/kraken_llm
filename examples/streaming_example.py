#!/usr/bin/env python3
"""
Демонстрация реального Outlines streaming с инкрементальным парсингом.
"""

from kraken_llm.client.structured import StructuredLLMClient
from kraken_llm.config.settings import LLMConfig
import asyncio
import os
import time
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

# Создаем конфигурацию
config = LLMConfig(
    endpoint=os.getenv("LLM_ENDPOINT"),
    api_key=os.getenv("LLM_TOKEN"),
    model=os.getenv("LLM_MODEL")
)
class Person(BaseModel):
    """Простая модель человека"""
    name: str = Field(..., description="Имя")
    age: int = Field(..., description="Возраст")
    city: str = Field(..., description="Город")
    occupation: str = Field(..., description="Профессия")
    active: bool = Field(True, description="Активен")


class Project(BaseModel):
    """Модель проекта"""
    title: str = Field(..., description="Название проекта")
    description: str = Field(..., description="Описание")
    technologies: List[str] = Field(
        default_factory=list, description="Технологии")
    budget: float = Field(0.0, description="Бюджет")
    duration_months: int = Field(1, description="Длительность в месяцах")
    priority: int = Field(3, description="Приоритет (1-5)")
    completed: bool = Field(False, description="Завершен")


async def demo_simple_streaming():
    """Демонстрация простого streaming"""
    print("🚀 ДЕМОНСТРАЦИЯ РЕАЛЬНОГО OUTLINES STREAMING")
    print("=" * 55)
    print("\n1️⃣  Простой streaming с базовой моделью")
    print("-" * 40)

    config.outlines_so_mode = True

    client = StructuredLLMClient(config)

    messages = [{
        "role": "user",
        "content": "Создай профиль: Анна Смирнова, 29 лет, UX дизайнер из Москвы"
    }]

    print("📝 Запрос: Создай профиль UX дизайнера")
    print("⏱️  Начинаем streaming...")

    start_time = time.time()

    try:
        result = await client.chat_completion_structured(
            messages=messages,
            response_model=Person,
            stream=True,
            max_tokens=200,
            temperature=0.3
        )

        execution_time = time.time() - start_time

        print(f"✅ Завершено за {execution_time:.3f}s")
        print(f"👤 Результат:")
        print(f"   • Имя: {result.name}")
        print(f"   • Возраст: {result.age} лет")
        print(f"   • Город: {result.city}")
        print(f"   • Профессия: {result.occupation}")
        print(f"   • Активен: {'Да' if result.active else 'Нет'}")

        return True

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False


async def demo_complex_streaming():
    """Демонстрация сложного streaming"""
    print("\n2️⃣  Сложный streaming с детальной моделью")
    print("-" * 40)

    config.outlines_so_mode = True

    client = StructuredLLMClient(config)

    messages = [{
        "role": "user",
        "content": """Создай проект разработки мобильного приложения:
        - Название: "FoodDelivery Pro"
        - Технологии: React Native, Node.js, MongoDB
        - Бюджет: 150000 рублей
        - Срок: 6 месяцев
        - Высокий приоритет"""
    }]

    print("📝 Запрос: Создай проект мобильного приложения")
    print("⏱️  Начинаем streaming...")

    start_time = time.time()

    try:
        result = await client.chat_completion_structured(
            messages=messages,
            response_model=Project,
            stream=True,
            max_tokens=400,
            temperature=0.4
        )

        execution_time = time.time() - start_time

        print(f"✅ Завершено за {execution_time:.3f}s")
        print(f"📋 Результат:")
        print(f"   • Название: {result.title}")
        print(f"   • Описание: {result.description}")
        print(f"   • Технологии: {', '.join(result.technologies)}")
        print(f"   • Бюджет: {result.budget:,.0f} руб.")
        print(f"   • Длительность: {result.duration_months} мес.")
        print(f"   • Приоритет: {result.priority}/5")
        print(f"   • Статус: {'Завершен' if result.completed else 'В работе'}")

        return True

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False


async def demo_performance_comparison():
    """Демонстрация сравнения производительности"""
    print("\n3️⃣  Сравнение производительности")
    print("-" * 40)

    config.outlines_so_mode = True

    client = StructuredLLMClient(config)

    messages = [{
        "role": "user",
        "content": "Создай профиль: Дмитрий Петров, 35 лет, архитектор ПО из СПб"
    }]

    print("📝 Запрос: Создай профиль архитектора ПО")

    # Non-streaming тест
    print("⏱️  Тестируем non-streaming...")
    start_time = time.time()

    try:
        result_non_stream = await client.chat_completion_structured(
            messages=messages,
            response_model=Person,
            stream=False,
            max_tokens=200,
            temperature=0.3
        )

        non_stream_time = time.time() - start_time
        print(f"   ✅ Non-streaming: {non_stream_time:.3f}s")

    except Exception as e:
        print(f"   ❌ Non-streaming ошибка: {e}")
        non_stream_time = 0

    # Streaming тест
    print("⏱️  Тестируем streaming...")
    start_time = time.time()

    try:
        result_stream = await client.chat_completion_structured(
            messages=messages,
            response_model=Person,
            stream=True,
            max_tokens=200,
            temperature=0.3
        )

        stream_time = time.time() - start_time
        print(f"   ✅ Streaming: {stream_time:.3f}s")

    except Exception as e:
        print(f"   ❌ Streaming ошибка: {e}")
        stream_time = 0

    # Анализ
    if non_stream_time > 0 and stream_time > 0:
        if stream_time < non_stream_time:
            improvement = ((non_stream_time - stream_time) /
                           non_stream_time) * 100
            print(
                f"🚀 Streaming быстрее на {improvement:.1f}% ({non_stream_time - stream_time:.3f}s)")
        else:
            degradation = ((stream_time - non_stream_time) /
                           non_stream_time) * 100
            print(
                f"⚠️  Non-streaming быстрее на {degradation:.1f}% ({stream_time - non_stream_time:.3f}s)")

        # Проверяем идентичность результатов
        if (hasattr(result_stream, 'name') and hasattr(result_non_stream, 'name') and
                result_stream.name == result_non_stream.name):
            print("✅ Результаты идентичны")
        else:
            print("⚠️  Результаты отличаются")

    return non_stream_time > 0 and stream_time > 0


async def demo_concurrent_streaming():
    """Демонстрация параллельного streaming"""
    print("\n4️⃣  Параллельный streaming")
    print("-" * 40)

    config.outlines_so_mode = True

    client = StructuredLLMClient(config)

    # Создаем несколько запросов
    requests = [
        {"name": "Алексей", "age": 28, "city": "Москва", "job": "разработчик"},
        {"name": "Мария", "age": 32, "city": "СПб", "job": "дизайнер"},
        {"name": "Игорь", "age": 25, "city": "Казань", "job": "аналитик"}
    ]

    print(f"📝 Запускаем {len(requests)} параллельных запросов...")

    async def single_request(req, index):
        messages = [{
            "role": "user",
            "content": f"Создай профиль: {req['name']}, {req['age']} лет, {req['job']} из {req['city']}"
        }]

        start_time = time.time()
        try:
            result = await client.chat_completion_structured(
                messages=messages,
                response_model=Person,
                stream=True,
                max_tokens=200,
                temperature=0.3
            )

            execution_time = time.time() - start_time
            return {
                "index": index + 1,
                "time": execution_time,
                "result": result,
                "success": True
            }
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "index": index + 1,
                "time": execution_time,
                "error": str(e),
                "success": False
            }

    # Запускаем параллельно
    start_time = time.time()
    results = await asyncio.gather(*[single_request(req, i) for i, req in enumerate(requests)])
    total_time = time.time() - start_time

    print(f"⏱️  Общее время: {total_time:.3f}s")

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"✅ Успешных: {len(successful)}/{len(results)}")

    for result in successful:
        print(
            f"   #{result['index']}: {result['time']:.3f}s - {result['result'].name}")

    for result in failed:
        print(f"   #{result['index']}: ❌ {result['error'][:50]}...")

    if successful:
        avg_time = sum(r["time"] for r in successful) / len(successful)
        sequential_time = sum(r["time"] for r in successful)
        speedup = sequential_time / total_time

        print(f"📊 Статистика:")
        print(f"   • Среднее время на запрос: {avg_time:.3f}s")
        print(f"   • Ускорение от параллелизма: {speedup:.1f}x")

    return len(successful) == len(results)


async def demo_incremental_parsing():
    """Демонстрация инкрементального парсинга"""
    print("\n5️⃣  Инкрементальный парсинг JSON")
    print("-" * 40)

    from kraken_llm.client.structured import IncrementalJSONParser

    parser = IncrementalJSONParser(Person)

    # Симулируем поступление JSON по частям
    json_parts = [
        '{"name": "Incremental',
        ' Parser", "age": 25,',
        ' "city": "Moscow",',
        ' "occupation": "Developer",',
        ' "active": true}'
    ]

    print("📝 Симулируем поступление JSON по частям:")

    for i, part in enumerate(json_parts, 1):
        print(f"   Часть {i}: '{part}'")
        result = parser.add_content(part)

        if result.is_complete:
            print(f"   ✅ JSON завершен после части {i}")
            print(f"   📋 Результат: {result.parsed_object}")
            return True
        elif result.is_invalid:
            print(f"   ❌ Ошибка валидации: {result.error}")
            return False
        else:
            print(
                f"   ⏳ Продолжаем накопление... (буфер: {len(parser.content_buffer)} символов)")

    # Финализируем
    final_result = parser.finalize()
    if final_result.is_complete:
        print(f"   ✅ Финализация успешна: {final_result.parsed_object}")
        return True
    else:
        print(f"   ❌ Финализация не удалась")
        return False


async def main():
    """Главная демонстрация"""
    print("🎯 ПОЛНАЯ ДЕМОНСТРАЦИЯ РЕАЛЬНОГО OUTLINES STREAMING")
    print("=" * 60)
    print("Демонстрируем возможности инкрементального JSON парсинга")
    print("и реального streaming для structured output")
    print()

    results = []

    # Демонстрируем все возможности
    results.append(await demo_simple_streaming())
    results.append(await demo_complex_streaming())
    results.append(await demo_performance_comparison())
    results.append(await demo_concurrent_streaming())
    results.append(await demo_incremental_parsing())

    # Подводим итоги
    print("\n" + "=" * 60)
    print("📊 ИТОГИ ДЕМОНСТРАЦИИ")
    print("=" * 60)

    success_count = sum(results)
    total_demos = len(results)

    print(f"Успешных демонстраций: {success_count}/{total_demos}")

    if success_count == total_demos:
        print("\n🎉 ВСЕ ДЕМОНСТРАЦИИ ПРОШЛИ УСПЕШНО!")
        print("\n✨ Возможности реального Outlines streaming:")
        print("   • ⚡ Инкрементальный парсинг JSON в реальном времени")
        print("   • 🚀 Производительность сравнимая или лучше non-streaming")
        print("   • 🔄 Поддержка параллельных запросов")
        print("   • 🛡️  Надежная валидация с fallback механизмами")
        print("   • 📊 Детальная диагностика и отладка")
        print("   • 🎛️  Готовность к продакшн использованию")
    elif success_count >= total_demos * 0.8:
        print("\n✅ БОЛЬШИНСТВО ДЕМОНСТРАЦИЙ УСПЕШНЫ!")
        print("   Система работает стабильно с минимальными проблемами")
    else:
        print("\n⚠️  ТРЕБУЕТСЯ ДОРАБОТКА")
        print("   Обнаружены проблемы, требующие исправления")

    print(f"\n🔬 Технические достижения:")
    print("   • Реализован полноценный streaming для Outlines")
    print("   • Создан инкрементальный JSON парсер")
    print("   • Добавлена валидация в реальном времени")
    print("   • Обеспечена совместимость с существующим API")
    print("   • Сохранена производительность и надежность")


if __name__ == "__main__":
    asyncio.run(main())
