#!/usr/bin/env python3
"""
Comprehensive Universal Client Example

Демонстрация всех возможностей UniversalLLMClient с практическими примерами
использования различных режимов и автоматического fallback.

Использует реальное подключение к LLM API с конфигурацией из .env файла.
"""

import asyncio
import sys
import os
from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

# Добавляем путь к модулям Kraken LLM
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kraken_llm import (
    create_universal_client,
    create_basic_client,
    create_advanced_client,
    create_full_client,
    create_universal_client_from_report,
    UniversalCapability,
    UniversalClientConfig,
    LLMConfig
)


# Модели данных для демонстрации
class Task(BaseModel):
    """Модель задачи"""
    title: str
    description: str
    priority: int
    estimated_hours: float
    tags: List[str] = []
    completed: bool = False


class ProjectPlan(BaseModel):
    """Модель плана проекта"""
    name: str
    description: str
    tasks: List[Task]
    total_hours: float
    deadline: str


class WeatherInfo(BaseModel):
    """Модель информации о погоде"""
    city: str
    temperature: int
    condition: str
    humidity: int
    wind_speed: float


async def demo_basic_universal_client():
    """Демонстрация базового универсального клиента"""
    print("🔧 Базовый универсальный клиент")
    print("-" * 50)
    
    if not os.getenv('LLM_ENDPOINT'):
        print("❌ Не найдена конфигурация LLM_ENDPOINT в .env файле")
        return
    
    config = LLMConfig()
    print(f"🔗 Подключение к: {config.endpoint}")
    print(f"📝 Модель: {config.model}")
    
    try:
        async with create_basic_client(config=config) as client:
            print(f"📋 Возможности: {client.get_available_capabilities()}")
            
            # Простой чат
            print("\n💬 Простой чат:")
            response = await client.chat_completion([
                {"role": "user", "content": "Объясни что такое машинное обучение простыми словами"}
            ], max_tokens=150)
            print(f"Ответ: {response[:200]}...")
            
            # Streaming
            print("\n🌊 Streaming ответ:")
            async for chunk in client.chat_completion_stream([
                {"role": "user", "content": "Перечисли 3 преимущества Python"}
            ], max_tokens=100):
                print(chunk, end="", flush=True)
            print("\n")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        print("   Проверьте настройки в .env файле и доступность сервера")


async def demo_advanced_universal_client():
    """Демонстрация продвинутого универсального клиента"""
    print("⚡ Продвинутый универсальный клиент")
    print("-" * 50)
    
    async with create_advanced_client() as client:
        print(f"Возможности: {client.get_available_capabilities()}")
        
        # Тестирование возможностей
        test_results = await client.test_capabilities()
        print("Результаты тестирования:")
        for capability, result in test_results.items():
            status = "✅" if result else "❌"
            print(f"  {status} {capability}")
        
        # Structured output с автоматическим fallback
        print("\n📋 Тестирование structured output...")
        try:
            task = await client.chat_completion_structured([
                {"role": "system", "content": "Создавай задачи в JSON формате согласно схеме."},
                {"role": "user", "content": "Создай задачу: разработать REST API для управления задачами"}
            ], response_model=Task)
            
            print("✅ Structured output успешно:")
            print(f"   Название: {task.title}")
            print(f"   Описание: {task.description}")
            print(f"   Приоритет: {task.priority}")
            print(f"   Часы: {task.estimated_hours}")
            print(f"   Теги: {task.tags}")
            
        except Exception as e:
            print(f"❌ Structured output недоступен: {e}")
        
        # Function calling
        print("\n🔧 Тестирование function calling...")
        
        def get_weather(city: str) -> str:
            """Получить информацию о погоде в городе"""
            return f"В городе {city}: солнечно, +22°C, влажность 60%, ветер 5 м/с"
        
        client.register_function(
            name="get_weather",
            function=get_weather,
            description="Получить текущую погоду в указанном городе",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Название города"}
                },
                "required": ["city"]
            }
        )
        
        try:
            response = await client.chat_completion([
                {"role": "user", "content": "Какая сейчас погода в Москве?"}
            ])
            print(f"✅ Function calling: {response}")
        except Exception as e:
            print(f"❌ Function calling недоступен: {e}")
        
        # Reasoning
        print("\n🧠 Тестирование reasoning...")
        try:
            response = await client.reasoning_completion([
                {"role": "user", "content": "Реши задачу: У Анны было 15 яблок, она дала 3 яблока Борису и 5 яблок Вере. Сколько яблок осталось у Анны?"}
            ], problem_type="math")
            print(f"✅ Reasoning: {response}")
        except Exception as e:
            print(f"❌ Reasoning недоступен: {e}")


async def demo_custom_configuration():
    """Демонстрация кастомной конфигурации"""
    print("⚙️ Кастомная конфигурация универсального клиента")
    print("-" * 50)
    
    # Выбираем конкретные возможности
    capabilities = {
        UniversalCapability.CHAT_COMPLETION,
        UniversalCapability.STREAMING,
        UniversalCapability.STRUCTURED_OUTPUT,
        UniversalCapability.FUNCTION_CALLING,
        UniversalCapability.REASONING
    }
    
    # Кастомная конфигурация LLM
    config = LLMConfig(
        temperature=0.3,  # Низкая температура для стабильности
        max_tokens=1500,
        stream=False
    )
    
    async with create_universal_client(
        config=config,
        capabilities=capabilities,
        auto_fallback=True,
        prefer_streaming=False
    ) as client:
        
        print(f"Кастомные возможности: {client.get_available_capabilities()}")
        print(f"Информация о клиенте: {client.get_client_info()}")
        
        # Комплексная задача с structured output
        try:
            project = await client.chat_completion_structured([
                {"role": "system", "content": "Создавай планы проектов в JSON формате."},
                {"role": "user", "content": """
                Создай план проекта для разработки веб-приложения для управления задачами.
                Включи 4-5 основных задач с описанием, приоритетом и оценкой времени.
                """}
            ], response_model=ProjectPlan)
            
            print("✅ Комплексный structured output:")
            print(f"   Проект: {project.name}")
            print(f"   Описание: {project.description}")
            print(f"   Общее время: {project.total_hours} часов")
            print(f"   Дедлайн: {project.deadline}")
            print(f"   Задачи ({len(project.tasks)}):")
            for i, task in enumerate(project.tasks, 1):
                print(f"     {i}. {task.title} (приоритет: {task.priority}, {task.estimated_hours}ч)")
                
        except Exception as e:
            print(f"❌ Комплексный structured output не удался: {e}")


async def demo_fallback_mechanisms():
    """Демонстрация механизмов fallback"""
    print("🔄 Механизмы автоматического fallback")
    print("-" * 50)
    
    async with create_advanced_client() as client:
        
        # Демонстрация fallback для structured output
        print("📋 Тестирование fallback для structured output...")
        
        # Простая модель для тестирования
        class SimpleResponse(BaseModel):
            answer: str
            confidence: float
            reasoning: str
        
        try:
            # Этот запрос должен пройти через различные режимы fallback
            response = await client.chat_completion_structured([
                {"role": "system", "content": "Отвечай в JSON формате согласно схеме."},
                {"role": "user", "content": "Оцени вероятность дождя завтра в процентах и объясни свои рассуждения"}
            ], response_model=SimpleResponse)
            
            print("✅ Fallback structured output успешно:")
            print(f"   Ответ: {response.answer}")
            print(f"   Уверенность: {response.confidence}")
            print(f"   Рассуждения: {response.reasoning}")
            
        except Exception as e:
            print(f"❌ Все методы fallback не удались: {e}")
        
        # Демонстрация fallback для других возможностей
        print("\n🔧 Тестирование fallback для других возможностей...")
        
        # Если reasoning недоступен, должен использоваться обычный chat
        try:
            response = await client.reasoning_completion([
                {"role": "user", "content": "Объясни пошагово как приготовить омлет"}
            ])
            print(f"✅ Reasoning (или fallback): {response[:150]}...")
        except Exception as e:
            print(f"❌ Reasoning fallback не удался: {e}")


async def demo_from_capabilities_report():
    """Демонстрация создания клиента из отчета анализатора"""
    print("📊 Создание клиента из отчета анализатора возможностей")
    print("-" * 50)
    
    # Симулируем отчет анализатора (в реальности получается из model_capabilities_analyzer.py)
    mock_report = {
        "model_summaries": {
            "test_model": {
                "confirmed_capabilities": [
                    {"capability": "chat_completion", "success_rate": 1.0},
                    {"capability": "streaming", "success_rate": 0.95},
                    {"capability": "structured_output_native", "success_rate": 0.8},
                    {"capability": "function_calling", "success_rate": 0.9},
                    {"capability": "reasoning_cot", "success_rate": 0.7},
                ],
                "failed_capabilities": [
                    {"capability": "multimodal_vision", "error": "Not supported"},
                    {"capability": "asr_stt", "error": "Endpoint not available"},
                ]
            }
        },
        "general_recommendations": [
            "Модель хорошо поддерживает базовые возможности",
            "Structured output работает стабильно",
            "Function calling рекомендуется для автоматизации"
        ]
    }
    
    try:
        async with create_universal_client_from_report(
            mock_report, 
            model_name="test_model"
        ) as client:
            
            print(f"Возможности из анализа: {client.get_available_capabilities()}")
            
            # Используем подтвержденные возможности
            response = await client.chat_completion([
                {"role": "user", "content": "Тест возможностей на основе анализа"}
            ])
            print(f"✅ Chat completion: {response[:100]}...")
            
            # Тестируем structured output (подтвержден в анализе)
            if "structured_output" in client.get_available_capabilities():
                try:
                    weather = await client.chat_completion_structured([
                        {"role": "user", "content": "Создай прогноз погоды для Санкт-Петербурга"}
                    ], response_model=WeatherInfo)
                    
                    print("✅ Structured output из анализа:")
                    print(f"   Город: {weather.city}")
                    print(f"   Температура: {weather.temperature}°C")
                    print(f"   Условия: {weather.condition}")
                    
                except Exception as e:
                    print(f"❌ Structured output не удался: {e}")
            
    except Exception as e:
        print(f"❌ Создание клиента из отчета не удалось: {e}")


async def demo_performance_comparison():
    """Демонстрация сравнения производительности различных режимов"""
    print("⚡ Сравнение производительности различных режимов")
    print("-" * 50)
    
    import time
    
    # Тестовые сообщения
    test_messages = [
        {"role": "user", "content": "Напиши короткое стихотворение о программировании"}
    ]
    
    # Базовый клиент
    start_time = time.time()
    async with create_basic_client() as client:
        response1 = await client.chat_completion(test_messages)
    basic_time = time.time() - start_time
    
    # Продвинутый клиент
    start_time = time.time()
    async with create_advanced_client() as client:
        response2 = await client.chat_completion(test_messages)
    advanced_time = time.time() - start_time
    
    # Полнофункциональный клиент
    start_time = time.time()
    async with create_full_client() as client:
        response3 = await client.chat_completion(test_messages)
    full_time = time.time() - start_time
    
    print(f"Базовый клиент: {basic_time:.3f}s")
    print(f"Продвинутый клиент: {advanced_time:.3f}s")
    print(f"Полнофункциональный клиент: {full_time:.3f}s")
    
    print(f"\nДлина ответов:")
    print(f"Базовый: {len(response1)} символов")
    print(f"Продвинутый: {len(response2)} символов")
    print(f"Полнофункциональный: {len(response3)} символов")


async def main():
    """Главная функция демонстрации"""
    print("🎯 Kraken LLM Universal Client - Comprehensive Demo")
    print("=" * 70)
    
    # Проверяем конфигурацию
    if not os.getenv('LLM_ENDPOINT'):
        print("❌ Конфигурация не найдена!")
        print("   Пожалуйста, настройте .env файл по примеру .env.example")
        print("   Минимальные настройки:")
        print("   LLM_ENDPOINT=http://your-llm-server:port")
        print("   LLM_TOKEN=your-api-token")
        print("   LLM_MODEL=your-model-name")
        return
    
    print(f"🔗 Используем конфигурацию:")
    print(f"   Endpoint: {os.getenv('LLM_ENDPOINT')}")
    print(f"   Model: {os.getenv('LLM_MODEL', 'не указана')}")
    print()
    
    try:
        # Базовые возможности
        await demo_basic_universal_client()
        print()
        
        # Продвинутые возможности
        await demo_advanced_universal_client()
        print()
        
        # Кастомная конфигурация
        await demo_custom_configuration()
        print()
        
        # Механизмы fallback
        await demo_fallback_mechanisms()
        print()
        
        # Создание из отчета анализатора
        await demo_from_capabilities_report()
        print()
        
        # Сравнение производительности
        await demo_performance_comparison()
        
        print("\n" + "=" * 70)
        print("✅ Comprehensive demo завершен!")
        print("\n💡 Ключевые особенности UniversalLLMClient:")
        print("1. Автоматический fallback для structured output")
        print("2. Единый интерфейс для всех возможностей")
        print("3. Гибкая конфигурация возможностей")
        print("4. Создание на основе анализа модели")
        print("5. Автоматическое тестирование возможностей")
        print("6. Оптимизация производительности")
        
    except Exception as e:
        print(f"❌ Ошибка в демонстрации: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())