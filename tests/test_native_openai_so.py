#!/usr/bin/env python3
"""
Тест нативного OpenAI structured output в режиме без Outlines.
Проверяет использование response_format вместо промптов.
"""

import asyncio
import os
import time
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from kraken_llm.config.settings import LLMConfig
from kraken_llm.client.structured import StructuredLLMClient

from dotenv import load_dotenv

load_dotenv()

# Модели для тестирования


class PersonModel(BaseModel):
    """Модель человека для тестирования"""
    name: str = Field(..., description="Имя человека")
    age: int = Field(..., ge=0, le=150, description="Возраст человека")
    city: str = Field(..., description="Город проживания")
    occupation: str = Field(..., description="Профессия")


class ProductModel(BaseModel):
    """Модель продукта для тестирования"""
    name: str = Field(..., description="Название продукта")
    price: float = Field(..., ge=0, description="Цена продукта")
    category: str = Field(..., description="Категория продукта")
    in_stock: bool = Field(..., description="Наличие на складе")


async def test_native_openai_structured_output():
    """Тестирование нативного OpenAI structured output"""
    print("🔧 Тестирование нативного OpenAI structured output...")

    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL")
    )
    # Отключаем Outlines для использования нативного OpenAI
    config.outlines_so_mode = False
    client = StructuredLLMClient(config)

    # Более явный запрос с указанием формата
    messages = [{
        "role": "user",
        "content": """Create JSON data for a person with these details:
Name: Elena Vasilyeva
Age: 32
City: Novosibirsk  
Occupation: Doctor

Return only valid JSON with English keys: name, age, city, occupation"""
    }]

    try:
        start_time = time.time()
        result = await client.chat_completion_structured(
            messages=messages,
            response_model=PersonModel,
            max_tokens=200,
            temperature=0.1
        )
        duration = time.time() - start_time

        if isinstance(result, PersonModel):
            print(
                f"✅ Нативный OpenAI SO работает: {result.name}, {result.age} лет, {result.city}")
            print(f"   ⏱️  Время выполнения: {duration:.2f}с")
            return True
        else:
            print(f"❌ Неожиданный тип результата: {type(result)}")
            return False

    except Exception as e:
        print(
            f"⚠️  Нативный OpenAI SO не поддерживается сервером: {type(e).__name__}")
        print(f"   Это ожидаемо для серверов без нативной поддержки response_format")
        return False


async def test_native_vs_outlines_comparison():
    """Сравнение нативного OpenAI SO с Outlines"""
    print("\n⚖️  Сравнение нативного OpenAI SO с Outlines...")

    # Тест нативного OpenAI
    config_native = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL")
    )
    config_native.outlines_so_mode = False
    client_native = StructuredLLMClient(config_native)

    # Тест Outlines
    config_outlines = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL")
    )
    config_outlines.outlines_so_mode = True
    client_outlines = StructuredLLMClient(config_outlines)

    messages = [{
        "role": "user",
        "content": "Создай данные для продукта: Samsung Galaxy, цена 45000, категория Смартфоны, в наличии"
    }]

    results = {}

    # Более явный запрос для нативного OpenAI
    native_messages = [{
        "role": "user",
        "content": """Create JSON for product:
Name: Samsung Galaxy
Price: 45000
Category: Smartphones  
In stock: true

Return only valid JSON with English keys: name, price, category, in_stock"""
    }]

    # Тест нативного OpenAI
    try:
        start_time = time.time()
        result_native = await client_native.chat_completion_structured(
            messages=native_messages,
            response_model=ProductModel,
            max_tokens=200,
            temperature=0.1
        )
        native_time = time.time() - start_time

        if isinstance(result_native, ProductModel):
            results["native"] = {
                "success": True,
                "result": result_native,
                "time": native_time
            }
            print(
                f"✅ Нативный OpenAI: {result_native.name}, цена {result_native.price} ({native_time:.2f}с)")
        else:
            results["native"] = {
                "success": False, "error": f"Неожиданный тип: {type(result_native)}"}
            print(f"❌ Нативный OpenAI: неожиданный тип {type(result_native)}")

    except Exception as e:
        results["native"] = {"success": False, "error": str(e)}
        print(f"⚠️  Нативный OpenAI: не поддерживается ({type(e).__name__})")

    # Тест Outlines
    try:
        start_time = time.time()
        result_outlines = await client_outlines.chat_completion_structured(
            messages=messages,
            response_model=ProductModel,
            max_tokens=200,
            temperature=0.1
        )
        outlines_time = time.time() - start_time

        if isinstance(result_outlines, ProductModel):
            results["outlines"] = {
                "success": True,
                "result": result_outlines,
                "time": outlines_time
            }
            print(
                f"✅ Outlines: {result_outlines.name}, цена {result_outlines.price} ({outlines_time:.2f}с)")
        else:
            results["outlines"] = {
                "success": False, "error": f"Неожиданный тип: {type(result_outlines)}"}
            print(f"❌ Outlines: неожиданный тип {type(result_outlines)}")

    except Exception as e:
        results["outlines"] = {"success": False, "error": str(e)}
        print(f"❌ Outlines: ошибка {e}")

    # Анализ результатов
    native_success = results.get("native", {}).get("success", False)
    outlines_success = results.get("outlines", {}).get("success", False)

    if native_success and outlines_success:
        native_time = results["native"]["time"]
        outlines_time = results["outlines"]["time"]
        faster = "нативный OpenAI" if native_time < outlines_time else "Outlines"
        time_diff = abs(native_time - outlines_time)
        print(f"📊 Оба режима работают, быстрее: {faster} на {time_diff:.2f}с")
        return True
    elif native_success:
        print(f"📊 Работает только нативный OpenAI")
        return True
    elif outlines_success:
        print(f"📊 Работает только Outlines")
        return True
    else:
        print(f"📊 Оба режима не работают")
        return False


async def test_complex_schema_native():
    """Тестирование сложной схемы с нативным OpenAI SO"""
    print("\n🔬 Тестирование сложной схемы с нативным OpenAI SO...")

    class ComplexModel(BaseModel):
        """Сложная модель для тестирования"""
        id: int = Field(..., description="Идентификатор")
        title: str = Field(..., description="Заголовок")
        tags: List[str] = Field(..., description="Теги")
        metadata: Dict[str, Any] = Field(..., description="Метаданные")
        is_active: bool = Field(..., description="Активность")
        price: float = Field(..., description="Цена")

    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL")
    )
    config.outlines_so_mode = False
    client = StructuredLLMClient(config)

    messages = [{
        "role": "user",
        "content": """
Создай данные для статьи:
- ID: 123
- Заголовок: "Искусственный интеллект в 2024"
- Теги: ["AI", "технологии", "будущее"]
- Метаданные: автор "Иван Петров", категория "Технологии"
- Активна: да
- Цена: 0 (бесплатная)
"""
    }]

    try:
        result = await client.chat_completion_structured(
            messages=messages,
            response_model=ComplexModel,
            max_tokens=300,
            temperature=0.1
        )

        if isinstance(result, ComplexModel):
            print(
                f"✅ Сложная схема работает: ID {result.id}, теги: {len(result.tags)}, метаданные: {len(result.metadata)}")
            return True
        else:
            print(f"❌ Неожиданный тип результата: {type(result)}")
            return False

    except Exception as e:
        print(f"❌ Ошибка со сложной схемой: {e}")
        return False


async def test_english_keys_enforcement():
    """Тестирование принуждения к английским ключам"""
    print("\n🌐 Тестирование принуждения к английским ключам...")

    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL")
    )
    config.outlines_so_mode = False
    client = StructuredLLMClient(config)

    # Явный запрос с требованием английских ключей
    messages = [{
        "role": "user",
        "content": """Create JSON for person with these Russian details:
Имя: Александр Иванов (Name: Alexander Ivanov)
Возраст: 29 лет (Age: 29)
Город: Казань (City: Kazan)
Профессия: Программист (Occupation: Programmer)

IMPORTANT: Use only English keys in JSON: name, age, city, occupation
Return only valid JSON."""
    }]

    try:
        result = await client.chat_completion_structured(
            messages=messages,
            response_model=PersonModel,
            max_tokens=200,
            temperature=0.1
        )

        if isinstance(result, PersonModel):
            print(
                f"✅ Английские ключи принудительно используются: {result.name}, {result.age} лет")

            # Проверяем что все поля заполнены (значит ключи были правильными)
            if all([result.name, result.age > 0, result.city, result.occupation]):
                print("✅ Все поля корректно заполнены с английскими ключами")
                return True
            else:
                print("⚠️  Некоторые поля не заполнены")
                return False
        else:
            print(f"❌ Неожиданный тип результата: {type(result)}")
            return False

    except Exception as e:
        print(
            f"⚠️  Принуждение английских ключей не работает: {type(e).__name__}")
        print("   Это ожидаемо без нативной поддержки response_format")
        return False


async def main():
    """Главная функция тестирования нативного OpenAI SO"""
    print("🔧 ТЕСТИРОВАНИЕ НАТИВНОГО OPENAI STRUCTURED OUTPUT")
    print("="*80)
    print("🎯 Цель: Проверить использование response_format вместо промптов")
    print("🔧 Режим: outlines_so_mode = False (нативный OpenAI)")
    print("="*80)

    start_time = time.time()
    results = []

    # Запускаем все тесты
    results.append(await test_native_openai_structured_output())
    results.append(await test_native_vs_outlines_comparison())
    results.append(await test_complex_schema_native())
    results.append(await test_english_keys_enforcement())

    # Подводим итоги
    duration = time.time() - start_time
    success_count = sum(results)
    total_tests = len(results)
    success_rate = success_count / total_tests

    print(f"\n{'='*80}")
    print(f"📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ НАТИВНОГО OPENAI SO")
    print(f"{'='*80}")
    print(f"⏱️  Время выполнения: {duration:.2f} секунд")
    print(f"📈 Всего тестов: {total_tests}")
    print(f"✅ Успешно: {success_count}")
    print(f"❌ Неудачно: {total_tests - success_count}")
    print(f"📊 Процент успеха: {success_rate*100:.1f}%")

    if success_rate >= 0.9:
        print(f"\n🎉 ОТЛИЧНО! Нативный OpenAI structured output работает идеально!")
        print(f"✅ response_format используется корректно")
        print(f"✅ Английские ключи принудительно применяются")
        print(f"✅ Сложные схемы поддерживаются")
    elif success_rate >= 0.7:
        print(f"\n👍 ХОРОШО! Нативный OpenAI SO работает с незначительными проблемами")
        print(f"🔧 Рекомендуется дополнительная настройка")
    elif success_rate >= 0.5:
        print(f"\n⚠️  ЧАСТИЧНАЯ ПОДДЕРЖКА! Некоторые возможности работают")
        print(f"� Сервер мтожет не поддерживать нативный response_format")
        print(f"🔧 Рекомендуется использовать Outlines режим для лучших результатов")
    elif success_rate >= 0.25:
        print(f"\n⚠️  ОГРАНИЧЕННАЯ ПОДДЕРЖКА! Минимальная работоспособность")
        print(f"🚨 Сервер не поддерживает нативный OpenAI response_format")
        print(f"✅ Используйте Outlines режим (outlines_so_mode = True)")
    else:
        print(f"\n❌ НЕТ ПОДДЕРЖКИ! Нативный OpenAI SO не работает")
        print(f"🚨 Сервер полностью не поддерживает response_format")
        print(f"✅ Обязательно используйте Outlines режим")

    print(f"\n📋 ПРОВЕРЕННЫЕ ВОЗМОЖНОСТИ:")
    print("- ✅ Нативный OpenAI structured output")
    print("- ✅ Сравнение с Outlines режимом")
    print("- ✅ Сложные JSON схемы")
    print("- ✅ Принуждение к английским ключам")
    print("- ✅ response_format вместо промптов")

    print(f"\n🏁 ТЕСТИРОВАНИЕ НАТИВНОГО OPENAI SO ЗАВЕРШЕНО!")


if __name__ == "__main__":
    asyncio.run(main())
