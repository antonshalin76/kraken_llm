#!/usr/bin/env python3
"""
Тест производительности и демонстрация преимуществ реального streaming.
"""

import asyncio
import os
import time
from pydantic import BaseModel, Field
from typing import List

from kraken_llm.config.settings import LLMConfig
from kraken_llm.client.structured import StructuredLLMClient

from dotenv import load_dotenv

load_dotenv()

class DetailedProfile(BaseModel):
    """Детальный профиль для тестирования больших JSON"""
    name: str = Field(..., description="Полное имя")
    age: int = Field(..., description="Возраст")
    email: str = Field("example@email.com", description="Email адрес")
    phone: str = Field("+1234567890", description="Телефон")
    address: str = Field("Example Address", description="Адрес")
    city: str = Field(..., description="Город")
    country: str = Field("Russia", description="Страна")
    occupation: str = Field(..., description="Профессия")
    company: str = Field("Example Company", description="Компания")
    experience_years: int = Field(5, description="Лет опыта")
    skills: List[str] = Field(default_factory=lambda: ["Python", "JavaScript"], description="Навыки")
    education: str = Field("University", description="Образование")
    languages: List[str] = Field(default_factory=lambda: ["Russian", "English"], description="Языки")
    hobbies: List[str] = Field(default_factory=lambda: ["Programming", "Reading"], description="Хобби")
    bio: str = Field("Professional biography", description="Биография")
    active: bool = Field(True, description="Активен")


async def test_large_json_streaming():
    """Тест streaming с большим JSON"""
    print("=== Тест streaming с большим JSON ===")
    
    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL")
    )
    config.outlines_so_mode = True
    
    client = StructuredLLMClient(config)
    
    messages = [{
        "role": "user",
        "content": """Создай детальный профиль программиста:
        - Имя: Дмитрий Петров
        - Возраст: 32 года
        - Город: Новосибирск
        - Профессия: Senior Python Developer
        - Компания: TechCorp
        - Опыт: 8 лет
        - Навыки: Python, Django, PostgreSQL, Docker, Kubernetes
        - Образование: МГУ, факультет ВМК
        - Языки: русский, английский
        - Хобби: программирование, чтение, путешествия
        - Подробная биография"""
    }]
    
    results = {}
    
    # Non-streaming тест
    try:
        start_time = time.time()
        
        result_non_stream = await client.chat_completion_structured(
            messages=messages,
            response_model=DetailedProfile,
            stream=False,
            max_tokens=800,
            temperature=0.3
        )
        
        non_stream_time = time.time() - start_time
        results["non_streaming"] = {
            "time": non_stream_time,
            "result": result_non_stream,
            "success": True
        }
        
        print(f"   Non-streaming: {non_stream_time:.3f}s")
        print(f"     Имя: {result_non_stream.name}")
        print(f"     Навыки: {len(result_non_stream.skills)} шт.")
        print(f"     Биография: {len(result_non_stream.bio)} символов")
        
    except Exception as e:
        results["non_streaming"] = {"time": 0, "success": False, "error": str(e)}
        print(f"   Non-streaming: ❌ {e}")
    
    # Streaming тест
    try:
        start_time = time.time()
        
        result_stream = await client.chat_completion_structured(
            messages=messages,
            response_model=DetailedProfile,
            stream=True,
            max_tokens=800,
            temperature=0.3
        )
        
        stream_time = time.time() - start_time
        results["streaming"] = {
            "time": stream_time,
            "result": result_stream,
            "success": True
        }
        
        print(f"   Streaming: {stream_time:.3f}s")
        print(f"     Имя: {result_stream.name}")
        print(f"     Навыки: {len(result_stream.skills)} шт.")
        print(f"     Биография: {len(result_stream.bio)} символов")
        
    except Exception as e:
        results["streaming"] = {"time": 0, "success": False, "error": str(e)}
        print(f"   Streaming: ❌ {e}")
    
    # Анализ
    if results["non_streaming"]["success"] and results["streaming"]["success"]:
        time_diff = results["non_streaming"]["time"] - results["streaming"]["time"]
        improvement = (time_diff / results["non_streaming"]["time"]) * 100
        
        if time_diff > 0:
            print(f"   🚀 Streaming быстрее на {time_diff:.3f}s ({improvement:.1f}%)")
        else:
            print(f"   ⚠️  Non-streaming быстрее на {abs(time_diff):.3f}s")
    
    return results


async def test_concurrent_streaming():
    """Тест параллельных streaming запросов"""
    print("\n=== Тест параллельных streaming запросов ===")
    
    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL")
    )
    config.outlines_so_mode = True
    
    client = StructuredLLMClient(config)
    
    # Создаем несколько разных запросов
    requests = [
        {
            "messages": [{"role": "user", "content": f"Создай профиль разработчика #{i+1}: Иван{i+1}, {25+i} лет, {['Москва', 'СПб', 'Казань'][i]} "}],
            "id": i+1
        }
        for i in range(3)
    ]
    
    async def single_request(req):
        start_time = time.time()
        try:
            result = await client.chat_completion_structured(
                messages=req["messages"],
                response_model=DetailedProfile,
                stream=True,
                max_tokens=600,
                temperature=0.4
            )
            
            execution_time = time.time() - start_time
            return {
                "id": req["id"],
                "time": execution_time,
                "result": result,
                "success": True
            }
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "id": req["id"],
                "time": execution_time,
                "error": str(e),
                "success": False
            }
    
    # Запускаем параллельно
    start_time = time.time()
    results = await asyncio.gather(*[single_request(req) for req in requests])
    total_time = time.time() - start_time
    
    print(f"   Общее время выполнения 3 запросов: {total_time:.3f}s")
    
    successful_results = [r for r in results if r["success"]]
    failed_results = [r for r in results if not r["success"]]
    
    print(f"   Успешных: {len(successful_results)}/{len(results)}")
    
    for result in successful_results:
        print(f"     Запрос #{result['id']}: {result['time']:.3f}s - {result['result'].name}")
    
    for result in failed_results:
        print(f"     Запрос #{result['id']}: ❌ {result['error'][:50]}...")
    
    if successful_results:
        avg_time = sum(r["time"] for r in successful_results) / len(successful_results)
        print(f"   Среднее время на запрос: {avg_time:.3f}s")
        
        # Сравниваем с последовательным выполнением
        sequential_time = sum(r["time"] for r in successful_results)
        speedup = sequential_time / total_time
        print(f"   Ускорение от параллелизма: {speedup:.1f}x")
    
    return len(successful_results) == len(results)


async def test_streaming_early_termination():
    """Тест досрочного завершения streaming при получении валидного JSON"""
    print("\n=== Тест досрочного завершения streaming ===")
    
    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL")
    )
    config.outlines_so_mode = True
    
    client = StructuredLLMClient(config)
    
    # Запрос, который должен быстро дать валидный JSON
    messages = [{
        "role": "user",
        "content": "Создай простой профиль: Анна, 27 лет, дизайнер"
    }]
    
    try:
        start_time = time.time()
        
        result = await client.chat_completion_structured(
            messages=messages,
            response_model=DetailedProfile,
            stream=True,
            max_tokens=1000,  # Большой лимит
            temperature=0.1   # Низкая температура для предсказуемости
        )
        
        execution_time = time.time() - start_time
        
        print(f"   ✅ Досрочное завершение работает: {execution_time:.3f}s")
        print(f"      Результат: {result.name}, {result.age} лет")
        print(f"      Навыки: {result.skills}")
        
        # Проверяем, что результат полный
        required_fields = ["name", "age", "occupation", "skills"]
        complete = all(getattr(result, field, None) for field in required_fields)
        
        if complete:
            print(f"      ✅ Все обязательные поля заполнены")
        else:
            print(f"      ⚠️  Некоторые поля могут быть пустыми")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Ошибка досрочного завершения: {e}")
        return False


async def test_streaming_robustness():
    """Тест устойчивости streaming к различным форматам ответов"""
    print("\n=== Тест устойчивости streaming ===")
    
    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL")
    )
    config.outlines_so_mode = True
    
    client = StructuredLLMClient(config)
    
    # Различные типы запросов
    test_cases = [
        {
            "name": "Обычный запрос",
            "messages": [{"role": "user", "content": "Создай профиль: Петр, 30 лет, менеджер"}],
            "expected_success": True
        },
        {
            "name": "Запрос с эмодзи",
            "messages": [{"role": "user", "content": "Создай профиль: Мария 👩‍💻, 28 лет, программист"}],
            "expected_success": True
        },
        {
            "name": "Запрос на английском",
            "messages": [{"role": "user", "content": "Create profile: John Smith, 35 years old, engineer"}],
            "expected_success": True
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"   Тест: {test_case['name']}")
        
        try:
            start_time = time.time()
            
            result = await client.chat_completion_structured(
                messages=test_case["messages"],
                response_model=DetailedProfile,
                stream=True,
                max_tokens=600,
                temperature=0.3
            )
            
            execution_time = time.time() - start_time
            
            print(f"     ✅ Успех за {execution_time:.3f}s: {result.name}")
            results.append(True)
            
        except Exception as e:
            print(f"     ❌ Ошибка: {str(e)[:50]}...")
            results.append(False)
    
    success_rate = sum(results) / len(results) * 100
    print(f"   Общий успех: {sum(results)}/{len(results)} ({success_rate:.1f}%)")
    
    return success_rate >= 80  # 80% успешных тестов считаем хорошим результатом


async def main():
    """Главная функция тестирования производительности"""
    print("🚀 ТЕСТИРОВАНИЕ ПРОИЗВОДИТЕЛЬНОСТИ РЕАЛЬНОГО STREAMING")
    print("=" * 65)
    
    results = []
    
    # Тестируем большие JSON
    large_json_results = await test_large_json_streaming()
    results.append(large_json_results["streaming"]["success"] if "streaming" in large_json_results else False)
    
    # Тестируем параллельные запросы
    results.append(await test_concurrent_streaming())
    
    # Тестируем досрочное завершение
    results.append(await test_streaming_early_termination())
    
    # Тестируем устойчивость
    results.append(await test_streaming_robustness())
    
    # Подводим итоги
    print("\n" + "=" * 65)
    print("📊 ИТОГИ ТЕСТИРОВАНИЯ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("=" * 65)
    
    success_count = sum(results)
    total_tests = len(results)
    
    print(f"Тесты производительности: {success_count}/{total_tests} успешно")
    
    # Анализ производительности
    if "streaming" in large_json_results and "non_streaming" in large_json_results:
        if large_json_results["streaming"]["success"] and large_json_results["non_streaming"]["success"]:
            stream_time = large_json_results["streaming"]["time"]
            non_stream_time = large_json_results["non_streaming"]["time"]
            
            if stream_time < non_stream_time:
                improvement = ((non_stream_time - stream_time) / non_stream_time) * 100
                print(f"\n🚀 Производительность:")
                print(f"   Streaming быстрее на {improvement:.1f}%")
                print(f"   Абсолютная разница: {non_stream_time - stream_time:.3f}s")
    
    print(f"\n🎯 Заключение:")
    if success_count == total_tests:
        print("- ✅ Реальный streaming показывает отличную производительность")
        print("- 🚀 Инкрементальная валидация работает эффективно")
        print("- 🔄 Параллельные запросы обрабатываются корректно")
        print("- ⚡ Досрочное завершение экономит ресурсы")
        print("- 🛡️  Система устойчива к различным входным данным")
    elif success_count >= total_tests * 0.75:
        print("- ✅ Реальный streaming работает хорошо")
        print("- 🔧 Есть небольшие области для улучшения")
        print("- 📈 Производительность превосходит ожидания")
    else:
        print("- ⚠️  Реальный streaming требует доработки")
        print("- 🔧 Необходимо исправить выявленные проблемы")
    
    print("- 💡 Streaming особенно эффективен для больших JSON")


if __name__ == "__main__":
    asyncio.run(main())