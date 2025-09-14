#!/usr/bin/env python3
"""
Полный workflow с UniversalLLMClient

Демонстрирует полный цикл работы:
1. Анализ возможностей модели
2. Создание оптимального универсального клиента
3. Использование всех доступных возможностей
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel

# Добавляем путь к модулям Kraken LLM
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Импорт анализатора возможностей
from model_capabilities_analyzer import ModelCapabilitiesAnalyzer

# Импорт универсального клиента
from kraken_llm import (
    create_universal_client_from_report,
    UniversalCapability,
    LLMConfig
)
from kraken_llm.tools import register_function, register_tool


# Модели для демонстрации
class ProjectTask(BaseModel):
    """Модель задачи проекта"""
    title: str
    description: str
    priority: int  # 1-5
    estimated_hours: float
    tags: List[str] = []
    completed: bool = False


class WeatherReport(BaseModel):
    """Модель отчета о погоде"""
    city: str
    temperature: int
    condition: str
    humidity: int
    wind_speed: float


# Функции для демонстрации function/tool calling
@register_function(description="Получить текущую погоду в городе")
def get_weather(city: str) -> Dict[str, Any]:
    """Получить информацию о погоде"""
    # Симуляция API погоды
    weather_data = {
        "Москва": {"temp": 15, "condition": "облачно", "humidity": 70, "wind": 5.2},
        "Санкт-Петербург": {"temp": 12, "condition": "дождь", "humidity": 85, "wind": 7.1},
        "Новосибирск": {"temp": 8, "condition": "снег", "humidity": 90, "wind": 3.5},
    }
    
    data = weather_data.get(city, {"temp": 20, "condition": "солнечно", "humidity": 60, "wind": 2.0})
    return {
        "city": city,
        "temperature": data["temp"],
        "condition": data["condition"],
        "humidity": data["humidity"],
        "wind_speed": data["wind"]
    }


@register_tool(description="Вычислить общее время проекта")
def calculate_project_time(tasks: List[Dict[str, Any]]) -> Dict[str, float]:
    """Вычислить общее время и статистику проекта"""
    total_hours = sum(task.get("estimated_hours", 0) for task in tasks)
    completed_hours = sum(
        task.get("estimated_hours", 0) 
        for task in tasks 
        if task.get("completed", False)
    )
    
    return {
        "total_hours": total_hours,
        "completed_hours": completed_hours,
        "remaining_hours": total_hours - completed_hours,
        "completion_percentage": (completed_hours / total_hours * 100) if total_hours > 0 else 0
    }


async def step1_analyze_capabilities():
    """Шаг 1: Анализ возможностей модели"""
    print("🔍 Шаг 1: Анализ возможностей модели")
    print("=" * 60)
    
    # Создаем анализатор
    analyzer = ModelCapabilitiesAnalyzer()
    
    if not analyzer.model_configs:
        print("⚠️ Не найдено конфигураций моделей в переменных окружения")
        print("Создаем симулированный отчет для демонстрации...")
        
        # Создаем симулированный отчет
        mock_report = {
            "metadata": {
                "timestamp": "2024-12-14T12:00:00",
                "execution_time": 45.2,
                "success_rate": 85.5
            },
            "model_summaries": {
                "demo_model": {
                    "confirmed_capabilities": [
                        {"capability": "chat_completion", "success_rate": 1.0, "avg_response_time": 1.2},
                        {"capability": "streaming", "success_rate": 0.95, "avg_response_time": 0.8},
                        {"capability": "structured_output_native", "success_rate": 0.9, "avg_response_time": 1.5},
                        {"capability": "function_calling", "success_rate": 0.8, "avg_response_time": 2.1},
                        {"capability": "tool_calling", "success_rate": 0.75, "avg_response_time": 2.3},
                        {"capability": "reasoning_cot", "success_rate": 0.7, "avg_response_time": 3.2},
                        {"capability": "embeddings", "success_rate": 0.85, "avg_response_time": 0.9},
                    ],
                    "recommended_clients": [
                        {"client": "StandardLLMClient", "success_rate": 0.95},
                        {"client": "StreamingLLMClient", "success_rate": 0.92},
                        {"client": "StructuredLLMClient", "success_rate": 0.88},
                    ],
                    "recommendations": [
                        "✅ Отличная совместимость с базовыми функциями",
                        "🔧 Рекомендуется использовать StandardLLMClient",
                        "⚡ Поддерживает streaming для real-time приложений",
                        "📋 Structured output работает стабильно",
                        "🛠️ Function calling требует дополнительной настройки"
                    ]
                }
            },
            "general_recommendations": [
                "🎯 Отличная общая совместимость с Kraken LLM",
                "📚 Используйте UniversalLLMClient для максимальной гибкости",
                "⚡ Включите streaming для улучшения UX",
                "🔧 Настройте function calling для расширенной функциональности"
            ]
        }
        
        return mock_report
    
    else:
        print(f"Найдено {len(analyzer.model_configs)} моделей для анализа")
        
        # Запускаем быстрый анализ
        report = await analyzer.analyze_all_models(quick_mode=True)
        
        # Сохраняем отчет
        filename = analyzer.save_report(report, "json")
        print(f"📄 Отчет сохранен: {filename}")
        
        return report


async def step2_create_optimal_client(capabilities_report: Dict[str, Any]):
    """Шаг 2: Создание оптимального универсального клиента"""
    print("\n🚀 Шаг 2: Создание оптимального универсального клиента")
    print("=" * 60)
    
    # Создаем клиент на основе результатов анализа
    client = create_universal_client_from_report(
        capabilities_report, 
        model_name="demo_model"  # Используем первую доступную модель
    )
    
    print(f"Создан универсальный клиент с возможностями:")
    capabilities = client.get_available_capabilities()
    for cap in capabilities:
        print(f"  ✅ {cap}")
    
    return client


async def step3_test_basic_capabilities(client):
    """Шаг 3: Тестирование базовых возможностей"""
    print("\n💬 Шаг 3: Тестирование базовых возможностей")
    print("=" * 60)
    
    # Базовый chat completion
    print("Тест базового chat completion:")
    try:
        response = await client.chat_completion([
            {"role": "system", "content": "Ты полезный ассистент для управления проектами."},
            {"role": "user", "content": "Привет! Помоги мне организовать работу над проектом."}
        ], max_tokens=100)
        
        print(f"✅ Ответ: {response}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    
    # Streaming (если доступен)
    if UniversalCapability.STREAMING in client.universal_config.capabilities:
        print("\nТест streaming:")
        try:
            print("Streaming ответ: ", end="", flush=True)
            async for chunk in client.chat_completion_stream([
                {"role": "user", "content": "Перечисли 3 важных принципа управления проектами"}
            ]):
                print(chunk, end="", flush=True)
            print()
        except Exception as e:
            print(f"❌ Streaming ошибка: {e}")


async def step4_test_structured_output(client):
    """Шаг 4: Тестирование structured output"""
    print("\n📋 Шаг 4: Тестирование structured output")
    print("=" * 60)
    
    if UniversalCapability.STRUCTURED_OUTPUT not in client.universal_config.capabilities:
        print("⚠️ Structured output недоступен")
        return
    
    try:
        # Создание задачи проекта
        task = await client.chat_completion_structured([
            {"role": "system", "content": "Создавай задачи в формате JSON согласно схеме."},
            {"role": "user", "content": "Создай задачу: разработать API для системы управления задачами. Приоритет высокий, примерно 20 часов работы."}
        ], response_model=ProjectTask)
        
        print("✅ Создана структурированная задача:")
        print(f"   Название: {task.title}")
        print(f"   Описание: {task.description}")
        print(f"   Приоритет: {task.priority}")
        print(f"   Время: {task.estimated_hours} часов")
        print(f"   Теги: {task.tags}")
        
        return task
        
    except Exception as e:
        print(f"❌ Ошибка structured output: {e}")
        return None


async def step5_test_function_calling(client):
    """Шаг 5: Тестирование function calling"""
    print("\n🔧 Шаг 5: Тестирование function calling")
    print("=" * 60)
    
    if UniversalCapability.FUNCTION_CALLING not in client.universal_config.capabilities:
        print("⚠️ Function calling недоступен")
        return
    
    # Регистрируем функцию
    client.register_function("get_weather", get_weather, "Получить погоду в городе")
    
    try:
        response = await client.chat_completion([
            {"role": "system", "content": "Используй доступные функции для получения актуальной информации."},
            {"role": "user", "content": "Какая сейчас погода в Москве? Используй функцию get_weather."}
        ], functions=[{
            "name": "get_weather",
            "description": "Получить информацию о погоде в городе",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Название города"}
                },
                "required": ["city"]
            }
        }], max_tokens=200)
        
        print(f"✅ Function calling ответ: {response}")
        
    except Exception as e:
        print(f"❌ Ошибка function calling: {e}")


async def step6_test_reasoning(client):
    """Шаг 6: Тестирование reasoning"""
    print("\n🧠 Шаг 6: Тестирование reasoning")
    print("=" * 60)
    
    if UniversalCapability.REASONING not in client.universal_config.capabilities:
        print("⚠️ Reasoning недоступен")
        return
    
    try:
        response = await client.reasoning_completion([
            {"role": "user", "content": """
            Проанализируй следующую ситуацию пошагово:
            
            У нас есть проект с 5 задачами:
            1. Планирование (8 часов) - выполнено
            2. Дизайн (12 часов) - выполнено  
            3. Разработка (40 часов) - в процессе (выполнено 60%)
            4. Тестирование (16 часов) - не начато
            5. Деплой (4 часов) - не начато
            
            Вопросы:
            1. Сколько часов уже потрачено?
            2. Сколько часов осталось?
            3. Какой процент проекта выполнен?
            4. Какие рекомендации по планированию?
            """}
        ], problem_type="analysis")
        
        print(f"✅ Reasoning анализ:")
        print(response)
        
    except Exception as e:
        print(f"❌ Ошибка reasoning: {e}")


async def step7_test_embeddings(client):
    """Шаг 7: Тестирование embeddings"""
    print("\n🔍 Шаг 7: Тестирование embeddings")
    print("=" * 60)
    
    if UniversalCapability.EMBEDDINGS not in client.universal_config.capabilities:
        print("⚠️ Embeddings недоступны")
        return
    
    try:
        # Тексты для векторизации
        project_docs = [
            "Техническое задание на разработку API",
            "Документация по архитектуре системы",
            "Руководство пользователя",
            "План тестирования",
            "Инструкция по деплою"
        ]
        
        embeddings = await client.get_embeddings(project_docs)
        print(f"✅ Создано {len(embeddings)} векторных представлений")
        
        # Поиск по сходству (если доступен)
        if UniversalCapability.SIMILARITY_SEARCH in client.universal_config.capabilities:
            results = await client.similarity_search(
                query_text="как развернуть приложение",
                candidate_texts=project_docs,
                top_k=2
            )
            
            print("🔍 Результаты поиска по сходству:")
            for i, result in enumerate(results[:2], 1):
                print(f"   {i}. {result}")
        
    except Exception as e:
        print(f"❌ Ошибка embeddings: {e}")


async def step8_comprehensive_example(client):
    """Шаг 8: Комплексный пример использования"""
    print("\n🎯 Шаг 8: Комплексный пример - Управление проектом")
    print("=" * 60)
    
    try:
        # 1. Создаем несколько задач через structured output
        tasks = []
        
        if UniversalCapability.STRUCTURED_OUTPUT in client.universal_config.capabilities:
            task_descriptions = [
                "Создать базу данных для хранения пользователей",
                "Разработать REST API для аутентификации", 
                "Написать фронтенд для регистрации пользователей",
                "Настроить CI/CD пайплайн",
                "Провести нагрузочное тестирование"
            ]
            
            for desc in task_descriptions:
                try:
                    task = await client.chat_completion_structured([
                        {"role": "system", "content": "Создавай задачи проекта в JSON формате."},
                        {"role": "user", "content": f"Создай задачу: {desc}. Оцени приоритет от 1 до 5 и время в часах."}
                    ], response_model=ProjectTask)
                    
                    tasks.append(task)
                except Exception as e:
                    print(f"   ⚠️ Не удалось создать задачу: {desc}")
        
        if tasks:
            print(f"✅ Создано {len(tasks)} задач проекта:")
            total_hours = 0
            for i, task in enumerate(tasks, 1):
                print(f"   {i}. {task.title} ({task.estimated_hours}ч, приоритет {task.priority})")
                total_hours += task.estimated_hours
            
            print(f"\n📊 Общее время проекта: {total_hours} часов")
        
        # 2. Анализ проекта через reasoning
        if UniversalCapability.REASONING in client.universal_config.capabilities and tasks:
            analysis = await client.reasoning_completion([
                {"role": "user", "content": f"""
                Проанализируй проект с задачами:
                {[f"{t.title} ({t.estimated_hours}ч, приоритет {t.priority})" for t in tasks]}
                
                Дай рекомендации по:
                1. Порядку выполнения задач
                2. Распределению ресурсов
                3. Рискам проекта
                4. Оптимизации времени
                """}
            ], problem_type="project_analysis")
            
            print(f"\n🧠 Анализ проекта:")
            print(analysis)
        
        # 3. Получаем информацию о погоде для планирования работы
        if UniversalCapability.FUNCTION_CALLING in client.universal_config.capabilities:
            weather_response = await client.chat_completion([
                {"role": "system", "content": "Используй функции для получения актуальной информации."},
                {"role": "user", "content": "Какая погода в Москве? Это поможет спланировать работу команды."}
            ], functions=[{
                "name": "get_weather",
                "description": "Получить погоду",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }])
            
            print(f"\n🌤️ Информация о погоде для планирования: {weather_response}")
        
    except Exception as e:
        print(f"❌ Ошибка в комплексном примере: {e}")


async def step9_performance_monitoring(client):
    """Шаг 9: Мониторинг производительности"""
    print("\n📈 Шаг 9: Мониторинг производительности")
    print("=" * 60)
    
    # Информация о клиенте
    info = client.get_client_info()
    print("Информация о клиенте:")
    print(f"   Возможности: {len(info['capabilities'])}")
    print(f"   Активные клиенты: {info['active_clients']}")
    print(f"   Конфигурация: {info['config']}")
    
    # Тестирование возможностей
    print("\nТестирование возможностей:")
    test_results = await client.test_capabilities()
    
    working_capabilities = []
    failed_capabilities = []
    
    for capability, result in test_results.items():
        if result:
            working_capabilities.append(capability)
            print(f"   ✅ {capability}")
        else:
            failed_capabilities.append(capability)
            print(f"   ❌ {capability}")
    
    print(f"\nСтатистика:")
    print(f"   Работающие возможности: {len(working_capabilities)}")
    print(f"   Неработающие возможности: {len(failed_capabilities)}")
    print(f"   Общий процент успеха: {len(working_capabilities) / len(test_results) * 100:.1f}%")


async def main():
    """Главная функция - полный workflow"""
    print("🎯 Kraken LLM Universal Client - Полный Workflow")
    print("=" * 80)
    print("Демонстрация полного цикла: анализ → создание → использование")
    print()
    
    try:
        # Шаг 1: Анализ возможностей
        capabilities_report = await step1_analyze_capabilities()
        
        # Шаг 2: Создание оптимального клиента
        client = await step2_create_optimal_client(capabilities_report)
        
        # Используем клиент в контекстном менеджере
        async with client:
            # Шаг 3: Базовые возможности
            await step3_test_basic_capabilities(client)
            
            # Шаг 4: Structured output
            await step4_test_structured_output(client)
            
            # Шаг 5: Function calling
            await step5_test_function_calling(client)
            
            # Шаг 6: Reasoning
            await step6_test_reasoning(client)
            
            # Шаг 7: Embeddings
            await step7_test_embeddings(client)
            
            # Шаг 8: Комплексный пример
            await step8_comprehensive_example(client)
            
            # Шаг 9: Мониторинг
            await step9_performance_monitoring(client)
        
        print("\n" + "=" * 80)
        print("🎉 Полный workflow завершен успешно!")
        print("\n💡 Следующие шаги:")
        print("1. Настройте переменные окружения для ваших моделей")
        print("2. Запустите model_capabilities_analyzer.py для реального анализа")
        print("3. Используйте результаты для создания оптимального клиента")
        print("4. Интегрируйте UniversalLLMClient в ваше приложение")
        
    except KeyboardInterrupt:
        print("\n❌ Workflow прерван пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка в workflow: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())