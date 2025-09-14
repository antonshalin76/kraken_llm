#!/usr/bin/env python3
"""
Быстрый старт с UniversalLLMClient

Простой пример использования универсального клиента Kraken LLM
для быстрого начала работы с различными возможностями.

Использует реальное подключение к LLM API с конфигурацией из .env файла.
"""

import asyncio
import sys
import os
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

# Добавляем путь к модулям Kraken LLM
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kraken_llm import (
    create_basic_client,
    create_advanced_client,
    create_universal_client,
    UniversalCapability,
    LLMConfig
)


class SimpleTask(BaseModel):
    """Простая модель задачи"""
    title: str
    priority: int
    completed: bool = False


async def quick_start_basic():
    """Быстрый старт с базовыми возможностями"""
    print("🚀 Быстрый старт - Базовые возможности")
    print("-" * 50)
    
    # Проверяем наличие конфигурации
    if not os.getenv('LLM_ENDPOINT'):
        print("❌ Не найдена конфигурация LLM_ENDPOINT в .env файле")
        print("   Пожалуйста, настройте .env файл по примеру .env.example")
        return
    
    # Создаем конфигурацию из переменных окружения
    config = LLMConfig()
    print(f"🔗 Подключение к: {config.endpoint}")
    print(f"📝 Модель: {config.model}")
    
    # Создаем базовый клиент (только chat + streaming)
    try:
        async with create_basic_client(config=config) as client:
            # Простой чат
            print("\n💬 Тестируем простой чат...")
            response = await client.chat_completion([
                {"role": "user", "content": "Привет! Ответь кратко - как дела?"}
            ], max_tokens=50)
            print(f"Ответ: {response}")
            
            # Streaming
            print("\n🌊 Тестируем streaming...")
            print("Ответ по частям: ", end="")
            async for chunk in client.chat_completion_stream([
                {"role": "user", "content": "Считай от 1 до 5, каждое число с новой строки"}
            ], max_tokens=30):
                print(chunk, end="", flush=True)
            print("\n")
            
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        print("   Проверьте настройки в .env файле и доступность сервера")


async def quick_start_advanced():
    """Быстрый старт с продвинутыми возможностями"""
    print("⚡ Быстрый старт - Продвинутые возможности")
    print("-" * 50)
    
    if not os.getenv('LLM_ENDPOINT'):
        print("❌ Не найдена конфигурация в .env файле")
        return
    
    config = LLMConfig()
    print(f"🔗 Подключение к: {config.endpoint}")
    
    # Создаем продвинутый клиент
    try:
        async with create_advanced_client(config=config) as client:
            print(f"📋 Доступные возможности: {client.get_available_capabilities()}")
            
            # Structured output с автоматическим fallback
            try:
                print("\n🏗️ Тестируем structured output (с автоматическим fallback)...")
                task = await client.chat_completion_structured([
                    {"role": "system", "content": "Создавай задачи в JSON формате согласно схеме."},
                    {"role": "user", "content": "Создай задачу: изучить основы Python программирования"}
                ], response_model=SimpleTask, max_tokens=100)
                
                print(f"✅ Структурированная задача получена:")
                print(f"   Название: {task.title}")
                print(f"   Приоритет: {task.priority}")
                print(f"   Завершена: {task.completed}")
                
            except Exception as e:
                print(f"❌ Structured output недоступен: {e}")
                print("   Возможные причины:")
                print("   - Модель не поддерживает JSON форматирование")
                print("   - Не установлена библиотека outlines")
                print("   - Проблемы с парсингом ответа")
            
            # Function calling
            print("\n🔧 Тестируем function calling...")
            
            def get_current_time():
                """Получить текущее время"""
                from datetime import datetime
                return datetime.now().strftime("%H:%M:%S")
            
            client.register_function(
                name="get_current_time",
                function=get_current_time,
                description="Получить текущее время"
            )
            
            try:
                response = await client.chat_completion([
                    {"role": "user", "content": "Сколько сейчас времени?"}
                ], max_tokens=50)
                print(f"✅ Function calling: {response}")
            except Exception as e:
                print(f"❌ Function calling недоступен: {e}")
            
            # Reasoning
            print("\n🧠 Тестируем reasoning...")
            try:
                response = await client.reasoning_completion([
                    {"role": "user", "content": "Реши простую задачу: 15 + 27 = ?"}
                ], max_tokens=100)
                print(f"✅ Reasoning: {response}")
            except Exception as e:
                print(f"❌ Reasoning недоступен: {e}")
                
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        print("   Проверьте настройки в .env файле и доступность сервера")


async def quick_start_custom():
    """Быстрый старт с кастомной конфигурацией"""
    print("⚙️ Быстрый старт - Кастомная конфигурация")
    print("-" * 50)
    
    if not os.getenv('LLM_ENDPOINT'):
        print("❌ Не найдена конфигурация в .env файле")
        return
    
    config = LLMConfig()
    
    # Создаем клиент с выбранными возможностями
    capabilities = {
        UniversalCapability.CHAT_COMPLETION,
        UniversalCapability.STREAMING,
        UniversalCapability.STRUCTURED_OUTPUT,
    }
    
    try:
        async with create_universal_client(
            config=config, 
            capabilities=capabilities
        ) as client:
            print(f"🎯 Кастомные возможности: {client.get_available_capabilities()}")
            
            # Тестируем возможности
            print("\n🧪 Тестируем доступные возможности...")
            test_results = await client.test_capabilities()
            print("Результаты тестирования:")
            for capability, result in test_results.items():
                status = "✅" if result else "❌"
                print(f"  {status} {capability}")
                
    except Exception as e:
        print(f"❌ Ошибка: {e}")


async def quick_start_from_analyzer():
    """Быстрый старт с использованием результатов анализатора"""
    print("📊 Быстрый старт - Из результатов анализатора")
    print("-" * 50)
    
    if not os.getenv('LLM_ENDPOINT'):
        print("❌ Не найдена конфигурация в .env файле")
        return
    
    # Запускаем реальный анализ возможностей модели
    print("🔍 Запускаем анализ возможностей модели...")
    
    try:
        # Импортируем анализатор
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from model_capabilities_analyzer import ModelCapabilitiesAnalyzer
        
        # Создаем анализатор и запускаем быстрый анализ
        analyzer = ModelCapabilitiesAnalyzer()
        print("⏳ Анализируем модель (это может занять несколько секунд)...")
        
        capabilities_report = await analyzer.analyze_all_models(quick_mode=True)
        
        if not capabilities_report.get('model_summaries'):
            print("❌ Не удалось получить результаты анализа")
            return
        
        # Создаем клиент на основе результатов анализа
        from kraken_llm import create_universal_client_from_report
        
        # Берем первую доступную модель из отчета
        model_name = next(iter(capabilities_report['model_summaries'].keys()))
        print(f"📋 Используем модель: {model_name}")
        
        async with create_universal_client_from_report(
            capabilities_report, 
            model_name=model_name
        ) as client:
            print(f"🎯 Возможности из анализа: {client.get_available_capabilities()}")
            
            # Используем подтвержденные возможности
            response = await client.chat_completion([
                {"role": "user", "content": "Тест подтвержденных возможностей модели"}
            ], max_tokens=50)
            print(f"✅ Ответ: {response}")
            
    except ImportError as e:
        print(f"❌ Не удалось импортировать анализатор: {e}")
        print("   Убедитесь, что model_capabilities_analyzer.py доступен")
    except Exception as e:
        print(f"❌ Ошибка анализа: {e}")
        print("   Проверьте подключение к модели и настройки .env")


async def main():
    """Главная функция демонстрации"""
    print("🎯 Kraken LLM Universal Client - Быстрый старт")
    print("=" * 60)
    
    try:
        # Базовые возможности
        await quick_start_basic()
        
        # Продвинутые возможности  
        await quick_start_advanced()
        
        # Кастомная конфигурация
        await quick_start_custom()
        
        # Из результатов анализатора
        await quick_start_from_analyzer()
        
        print("✅ Быстрый старт завершен!")
        print("\n💡 Следующие шаги:")
        print("1. Настройте .env файл с вашими моделями (см. .env.example)")
        print("2. Запустите python3 model_capabilities_analyzer.py для полного анализа")
        print("3. Используйте create_universal_client_from_report() с результатами")
        print("4. Изучите universal_client_example.py для подробных примеров")
        print("\n🔧 Текущая конфигурация:")
        if os.getenv('LLM_ENDPOINT'):
            print(f"   Endpoint: {os.getenv('LLM_ENDPOINT')}")
            print(f"   Model: {os.getenv('LLM_MODEL', 'не указана')}")
        else:
            print("   ❌ Конфигурация не найдена - настройте .env файл")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())