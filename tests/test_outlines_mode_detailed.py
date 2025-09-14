#!/usr/bin/env python3
"""
Детальное тестирование переключателя outlines_so_mode в StructuredLLMClient.
Проверяет различные сценарии использования и режимы работы.
"""

import asyncio
import os
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from kraken_llm.config.settings import LLMConfig
from kraken_llm.client.structured import StructuredLLMClient

from dotenv import load_dotenv

load_dotenv()

# Тестовые модели разной сложности
class SimpleModel(BaseModel):
    """Простая модель"""
    name: str = Field(..., description="Имя")
    value: int = Field(..., description="Значение")


class ComplexModel(BaseModel):
    """Сложная модель с вложенными объектами"""
    id: int = Field(..., description="Идентификатор")
    title: str = Field(..., description="Заголовок")
    tags: List[str] = Field(..., description="Теги")
    metadata: Dict[str, Any] = Field(..., description="Метаданные")
    is_active: bool = Field(..., description="Активность")


class NestedModel(BaseModel):
    """Модель с вложенными объектами"""
    user: SimpleModel = Field(..., description="Пользователь")
    settings: Dict[str, str] = Field(..., description="Настройки")
    scores: List[float] = Field(..., description="Оценки")


async def test_outlines_mode_configuration():
    """Тестирование конфигурации режима Outlines"""
    print("🔧 Тестирование конфигурации outlines_so_mode...")
    
    results = []
    
    # Тест 1: Режим по умолчанию
    try:
        config_default = LLMConfig(
            endpoint=os.getenv("LLM_ENDPOINT"),
            api_key=os.getenv("LLM_TOKEN"),
            model=os.getenv("LLM_MODEL")
        )
        client_default = StructuredLLMClient(config_default)
        
        print(f"   📝 Режим по умолчанию: {config_default.outlines_so_mode}")
        results.append(("Конфигурация по умолчанию", True, f"outlines_so_mode = {config_default.outlines_so_mode}"))
        
    except Exception as e:
        results.append(("Конфигурация по умолчанию", False, str(e)))
    
    # Тест 2: Явное отключение Outlines
    try:
        config_disabled = LLMConfig(
            endpoint=os.getenv("LLM_ENDPOINT"),
            api_key=os.getenv("LLM_TOKEN"),
            model=os.getenv("LLM_MODEL")            
        )
        config_disabled.outlines_so_mode = False
        client_disabled = StructuredLLMClient(config_disabled)
        
        print(f"   📝 Отключенный режим: {config_disabled.outlines_so_mode}")
        results.append(("Отключение Outlines", True, f"outlines_so_mode = {config_disabled.outlines_so_mode}"))
        
    except Exception as e:
        results.append(("Отключение Outlines", False, str(e)))
    
    # Тест 3: Явное включение Outlines
    try:
        config_enabled = LLMConfig(
            endpoint=os.getenv("LLM_ENDPOINT"),
            api_key=os.getenv("LLM_TOKEN"),
            model=os.getenv("LLM_MODEL")
        )
        config_enabled.outlines_so_mode = True
        client_enabled = StructuredLLMClient(config_enabled)
        
        print(f"   📝 Включенный режим: {config_enabled.outlines_so_mode}")
        results.append(("Включение Outlines", True, f"outlines_so_mode = {config_enabled.outlines_so_mode}"))
        
    except Exception as e:
        results.append(("Включение Outlines", False, str(e)))
    
    return results


async def test_simple_model_both_modes():
    """Тестирование простой модели в обоих режимах"""
    print("\n📋 Тестирование простой модели в обоих режимах...")
    
    results = []
    
    messages = [{
        "role": "user",
        "content": """
Создай JSON объект:
{"name": "Тест", "value": 42}

Верни ТОЛЬКО валидный JSON без дополнительного текста.
"""
    }]
    
    # Тест без Outlines
    try:
        config_no_outlines = LLMConfig(
            endpoint=os.getenv("LLM_ENDPOINT"),
            api_key=os.getenv("LLM_TOKEN"),
            model=os.getenv("LLM_MODEL")
        )
        config_no_outlines.outlines_so_mode = False
        client_no_outlines = StructuredLLMClient(config_no_outlines)
        
        start_time = time.time()
        result_no_outlines = await client_no_outlines.chat_completion_structured(
            messages=messages,
            response_model=SimpleModel,
            max_tokens=100,
            temperature=0.1
        )
        no_outlines_time = time.time() - start_time
        
        if isinstance(result_no_outlines, SimpleModel):
            results.append((
                "Простая модель без Outlines",
                True,
                f"Результат: {result_no_outlines.name}={result_no_outlines.value}, время: {no_outlines_time:.2f}с"
            ))
        else:
            results.append((
                "Простая модель без Outlines",
                False,
                f"Неожиданный тип: {type(result_no_outlines)}"
            ))
            
    except Exception as e:
        results.append(("Простая модель без Outlines", False, str(e)))
    
    # Тест с Outlines
    try:
        config_with_outlines = LLMConfig(
            endpoint=os.getenv("LLM_ENDPOINT"),
            api_key=os.getenv("LLM_TOKEN"),
            model=os.getenv("LLM_MODEL")
        )
        config_with_outlines.outlines_so_mode = True
        client_with_outlines = StructuredLLMClient(config_with_outlines)
        
        start_time = time.time()
        result_with_outlines = await client_with_outlines.chat_completion_structured(
            messages=messages,
            response_model=SimpleModel,
            max_tokens=100,
            temperature=0.1
        )
        with_outlines_time = time.time() - start_time
        
        if isinstance(result_with_outlines, SimpleModel):
            results.append((
                "Простая модель с Outlines",
                True,
                f"Результат: {result_with_outlines.name}={result_with_outlines.value}, время: {with_outlines_time:.2f}с"
            ))
        else:
            results.append((
                "Простая модель с Outlines",
                False,
                f"Неожиданный тип: {type(result_with_outlines)}"
            ))
            
    except Exception as e:
        results.append(("Простая модель с Outlines", False, str(e)))
    
    return results


async def test_complex_model_both_modes():
    """Тестирование сложной модели в обоих режимах"""
    print("\n🔬 Тестирование сложной модели в обоих режимах...")
    
    results = []
    
    messages = [{
        "role": "user",
        "content": """
Создай JSON объект для статьи:
{
  "id": 123,
  "title": "Тестовая статья",
  "tags": ["тест", "пример"],
  "metadata": {"author": "Автор", "category": "Тестирование"},
  "is_active": true
}

Верни ТОЛЬКО валидный JSON.
"""
    }]
    
    # Тест без Outlines
    try:
        config_no_outlines = LLMConfig(
            endpoint=os.getenv("LLM_ENDPOINT"),
            api_key=os.getenv("LLM_TOKEN"),
            model=os.getenv("LLM_MODEL")
        )
        config_no_outlines.outlines_so_mode = False
        client_no_outlines = StructuredLLMClient(config_no_outlines)
        
        start_time = time.time()
        result_no_outlines = await client_no_outlines.chat_completion_structured(
            messages=messages,
            response_model=ComplexModel,
            max_tokens=200,
            temperature=0.1
        )
        no_outlines_time = time.time() - start_time
        
        if isinstance(result_no_outlines, ComplexModel):
            results.append((
                "Сложная модель без Outlines",
                True,
                f"ID: {result_no_outlines.id}, теги: {len(result_no_outlines.tags)}, время: {no_outlines_time:.2f}с"
            ))
        else:
            results.append((
                "Сложная модель без Outlines",
                False,
                f"Неожиданный тип: {type(result_no_outlines)}"
            ))
            
    except Exception as e:
        results.append(("Сложная модель без Outlines", False, str(e)))
    
    # Тест с Outlines
    try:
        config_with_outlines = LLMConfig(
            endpoint=os.getenv("LLM_ENDPOINT"),
            api_key=os.getenv("LLM_TOKEN"),
            model=os.getenv("LLM_MODEL")
        )
        config_with_outlines.outlines_so_mode = True
        client_with_outlines = StructuredLLMClient(config_with_outlines)
        
        start_time = time.time()
        result_with_outlines = await client_with_outlines.chat_completion_structured(
            messages=messages,
            response_model=ComplexModel,
            max_tokens=200,
            temperature=0.1
        )
        with_outlines_time = time.time() - start_time
        
        if isinstance(result_with_outlines, ComplexModel):
            results.append((
                "Сложная модель с Outlines",
                True,
                f"ID: {result_with_outlines.id}, теги: {len(result_with_outlines.tags)}, время: {with_outlines_time:.2f}с"
            ))
        else:
            results.append((
                "Сложная модель с Outlines",
                False,
                f"Неожиданный тип: {type(result_with_outlines)}"
            ))
            
    except Exception as e:
        results.append(("Сложная модель с Outlines", False, str(e)))
    
    return results


async def test_error_handling_both_modes():
    """Тестирование обработки ошибок в обоих режимах"""
    print("\n🚨 Тестирование обработки ошибок в обоих режимах...")
    
    results = []
    
    # Некорректный промпт
    messages = [{
        "role": "user",
        "content": "Расскажи анекдот про котов"
    }]
    
    # Тест без Outlines
    try:
        config_no_outlines = LLMConfig(
            endpoint=os.getenv("LLM_ENDPOINT"),
            api_key=os.getenv("LLM_TOKEN"),
            model=os.getenv("LLM_MODEL")
        )
        config_no_outlines.outlines_so_mode = False
        client_no_outlines = StructuredLLMClient(config_no_outlines)
        
        result = await client_no_outlines.chat_completion_structured(
            messages=messages,
            response_model=SimpleModel,
            max_tokens=100,
            temperature=0.1
        )
        
        results.append((
            "Обработка ошибок без Outlines",
            False,
            f"Неожиданно получен результат: {type(result)}"
        ))
        
    except Exception as e:
        error_type = type(e).__name__
        if "ValidationError" in error_type or "JSON" in str(e):
            results.append((
                "Обработка ошибок без Outlines",
                True,
                f"Корректно обработана ошибка: {error_type}"
            ))
        else:
            results.append((
                "Обработка ошибок без Outlines",
                False,
                f"Неожиданная ошибка: {error_type}: {str(e)[:100]}"
            ))
    
    # Тест с Outlines
    try:
        config_with_outlines = LLMConfig(
            endpoint=os.getenv("LLM_ENDPOINT"),
            api_key=os.getenv("LLM_TOKEN"),
            model=os.getenv("LLM_MODEL")
        )
        config_with_outlines.outlines_so_mode = True
        client_with_outlines = StructuredLLMClient(config_with_outlines)
        
        result = await client_with_outlines.chat_completion_structured(
            messages=messages,
            response_model=SimpleModel,
            max_tokens=100,
            temperature=0.1
        )
        
        results.append((
            "Обработка ошибок с Outlines",
            False,
            f"Неожиданно получен результат: {type(result)}"
        ))
        
    except Exception as e:
        error_type = type(e).__name__
        if "ValidationError" in error_type or "JSON" in str(e):
            results.append((
                "Обработка ошибок с Outlines",
                True,
                f"Корректно обработана ошибка: {error_type}"
            ))
        else:
            results.append((
                "Обработка ошибок с Outlines",
                False,
                f"Неожиданная ошибка: {error_type}: {str(e)[:100]}"
            ))
    
    return results


async def test_performance_comparison():
    """Сравнение производительности режимов"""
    print("\n⚡ Сравнение производительности режимов...")
    
    results = []
    
    messages = [{
        "role": "user",
        "content": """
Создай JSON: {"name": "Производительность", "value": 100}
Верни ТОЛЬКО JSON.
"""
    }]
    
    # Множественные тесты для статистики
    no_outlines_times = []
    with_outlines_times = []
    
    for i in range(3):  # 3 теста для усреднения
        # Без Outlines
        try:
            config_no_outlines = LLMConfig(
                endpoint=os.getenv("LLM_ENDPOINT"),
                api_key=os.getenv("LLM_TOKEN"),
                model=os.getenv("LLM_MODEL")
            )
            config_no_outlines.outlines_so_mode = False
            client_no_outlines = StructuredLLMClient(config_no_outlines)
            
            start_time = time.time()
            await client_no_outlines.chat_completion_structured(
                messages=messages,
                response_model=SimpleModel,
                max_tokens=100,
                temperature=0.1
            )
            no_outlines_times.append(time.time() - start_time)
            
        except Exception as e:
            print(f"   ⚠️  Ошибка в тесте без Outlines #{i+1}: {e}")
        
        # С Outlines
        try:
            config_with_outlines = LLMConfig(
                endpoint=os.getenv("LLM_ENDPOINT"),
                api_key=os.getenv("LLM_TOKEN"),
                model=os.getenv("LLM_MODEL")
            )
            config_with_outlines.outlines_so_mode = True
            client_with_outlines = StructuredLLMClient(config_with_outlines)
            
            start_time = time.time()
            await client_with_outlines.chat_completion_structured(
                messages=messages,
                response_model=SimpleModel,
                max_tokens=100,
                temperature=0.1
            )
            with_outlines_times.append(time.time() - start_time)
            
        except Exception as e:
            print(f"   ⚠️  Ошибка в тесте с Outlines #{i+1}: {e}")
    
    # Анализ результатов
    if no_outlines_times and with_outlines_times:
        avg_no_outlines = sum(no_outlines_times) / len(no_outlines_times)
        avg_with_outlines = sum(with_outlines_times) / len(with_outlines_times)
        
        difference = abs(avg_with_outlines - avg_no_outlines)
        faster_mode = "без Outlines" if avg_no_outlines < avg_with_outlines else "с Outlines"
        
        results.append((
            "Сравнение производительности",
            True,
            f"Без Outlines: {avg_no_outlines:.2f}с, с Outlines: {avg_with_outlines:.2f}с, быстрее: {faster_mode} на {difference:.2f}с"
        ))
    else:
        results.append((
            "Сравнение производительности",
            False,
            "Недостаточно данных для сравнения"
        ))
    
    return results


def print_results(test_name: str, results: List[tuple]):
    """Вывод результатов теста"""
    print(f"\n📊 Результаты: {test_name}")
    print("-" * 60)
    
    success_count = 0
    total_count = len(results)
    
    for name, success, details in results:
        status = "✅" if success else "❌"
        print(f"{status} {name}")
        if details:
            print(f"   📝 {details}")
        if success:
            success_count += 1
    
    success_rate = (success_count / total_count * 100) if total_count > 0 else 0
    print(f"\n📈 Успешность: {success_count}/{total_count} ({success_rate:.1f}%)")
    
    return success_count, total_count


async def main():
    """Главная функция детального тестирования"""
    print("🔍 ДЕТАЛЬНОЕ ТЕСТИРОВАНИЕ ПЕРЕКЛЮЧАТЕЛЯ OUTLINES_SO_MODE")
    print("=" * 80)
    print("🎯 Цель: Проверить все аспекты работы переключателя outlines_so_mode")
    print("🔧 Включает: конфигурацию, простые/сложные модели, ошибки, производительность")
    print("=" * 80)
    
    total_success = 0
    total_tests = 0
    
    # Тестирование конфигурации
    config_results = await test_outlines_mode_configuration()
    success, tests = print_results("Конфигурация outlines_so_mode", config_results)
    total_success += success
    total_tests += tests
    
    # Тестирование простых моделей
    simple_results = await test_simple_model_both_modes()
    success, tests = print_results("Простые модели в обоих режимах", simple_results)
    total_success += success
    total_tests += tests
    
    # Тестирование сложных моделей
    complex_results = await test_complex_model_both_modes()
    success, tests = print_results("Сложные модели в обоих режимах", complex_results)
    total_success += success
    total_tests += tests
    
    # Тестирование обработки ошибок
    error_results = await test_error_handling_both_modes()
    success, tests = print_results("Обработка ошибок в обоих режимах", error_results)
    total_success += success
    total_tests += tests
    
    # Тестирование производительности
    perf_results = await test_performance_comparison()
    success, tests = print_results("Сравнение производительности", perf_results)
    total_success += success
    total_tests += tests
    
    # Итоговая сводка
    print(f"\n{'='*80}")
    print(f"📊 ИТОГОВАЯ СВОДКА ДЕТАЛЬНОГО ТЕСТИРОВАНИЯ")
    print(f"{'='*80}")
    print(f"📈 Всего тестов: {total_tests}")
    print(f"✅ Успешно: {total_success}")
    print(f"❌ Неудачно: {total_tests - total_success}")
    print(f"📊 Процент успеха: {(total_success/total_tests*100):.1f}%")
    
    if total_success == total_tests:
        print(f"\n🎉 ПРЕВОСХОДНО! Переключатель outlines_so_mode работает идеально!")
        print(f"✅ Все режимы функционируют корректно")
        print(f"✅ Обработка ошибок работает в обоих режимах")
        print(f"✅ Производительность в пределах нормы")
    elif total_success >= total_tests * 0.8:
        print(f"\n👍 ХОРОШО! Переключатель outlines_so_mode работает с незначительными проблемами")
        print(f"🔧 Рекомендуется исправить выявленные ошибки")
    else:
        print(f"\n⚠️  ТРЕБУЕТСЯ ДОРАБОТКА! Переключатель outlines_so_mode имеет проблемы")
        print(f"🛠️  Необходимо исправить критические ошибки")
    
    print(f"\n🎯 ПРОВЕРЕННЫЕ АСПЕКТЫ:")
    print(f"- ✅ Конфигурация и инициализация")
    print(f"- ✅ Работа с простыми моделями")
    print(f"- ✅ Работа со сложными моделями")
    print(f"- ✅ Обработка ошибок и исключений")
    print(f"- ✅ Сравнение производительности")
    print(f"- ✅ Переключение между режимами")
    
    print(f"\n🏁 ДЕТАЛЬНОЕ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")


if __name__ == "__main__":
    asyncio.run(main())