#!/usr/bin/env python3
"""
Пример демонстрации Structured Output с автоматическим fallback

Показывает, как UniversalLLMClient автоматически переключается между:
1. Нативный structured output (если поддерживается)
2. Outlines режим (fallback)
3. Обычный chat completion с JSON парсингом (финальный fallback)
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field

# Добавляем путь к модулям Kraken LLM
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kraken_llm import create_advanced_client, LLMConfig
from kraken_llm.client.universal import UniversalLLMClient, UniversalClientConfig, UniversalCapability


# Модели для демонстрации
class Task(BaseModel):
    """Модель задачи"""
    title: str = Field(description="Название задачи")
    description: str = Field(description="Подробное описание")
    priority: int = Field(description="Приоритет от 1 до 5", ge=1, le=5)
    estimated_hours: float = Field(description="Оценка времени в часах")
    tags: List[str] = Field(default=[], description="Теги задачи")
    completed: bool = Field(default=False, description="Статус выполнения")


class ProjectPlan(BaseModel):
    """Модель плана проекта"""
    project_name: str = Field(description="Название проекта")
    description: str = Field(description="Описание проекта")
    tasks: List[Task] = Field(description="Список задач")
    total_hours: float = Field(description="Общее время проекта")
    deadline: Optional[str] = Field(default=None, description="Дедлайн проекта")


class UserProfile(BaseModel):
    """Модель профиля пользователя"""
    name: str = Field(description="Имя пользователя")
    age: int = Field(description="Возраст", ge=0, le=150)
    email: str = Field(description="Email адрес")
    skills: List[str] = Field(description="Навыки и умения")
    experience_years: int = Field(description="Опыт работы в годах", ge=0)
    is_active: bool = Field(default=True, description="Активен ли пользователь")


async def demo_simple_structured_output():
    """Демонстрация простого structured output"""
    print("🎯 Демонстрация простого structured output")
    print("=" * 60)
    
    async with create_advanced_client() as client:
        try:
            print("Создаем простую задачу...")
            task = await client.chat_completion_structured([
                {"role": "system", "content": "Создавай задачи в JSON формате согласно схеме."},
                {"role": "user", "content": "Создай задачу: написать unit тесты для API. Это высокоприоритетная задача."}
            ], response_model=Task)
            
            print("✅ Задача создана успешно:")
            print(f"   📋 Название: {task.title}")
            print(f"   📝 Описание: {task.description}")
            print(f"   ⭐ Приоритет: {task.priority}/5")
            print(f"   ⏱️ Время: {task.estimated_hours} часов")
            print(f"   🏷️ Теги: {task.tags}")
            print(f"   ✅ Завершена: {task.completed}")
            
            return task
            
        except Exception as e:
            print(f"❌ Ошибка создания задачи: {e}")
            return None


async def demo_complex_structured_output():
    """Демонстрация сложного structured output"""
    print("\n🏗️ Демонстрация сложного structured output")
    print("=" * 60)
    
    async with create_advanced_client() as client:
        try:
            print("Создаем план проекта с несколькими задачами...")
            project = await client.chat_completion_structured([
                {"role": "system", "content": "Создавай планы проектов в JSON формате согласно схеме."},
                {"role": "user", "content": """
                Создай план проекта "Веб-приложение для управления задачами" со следующими задачами:
                1. Дизайн UI/UX (приоритет 4, 16 часов)
                2. Разработка backend API (приоритет 5, 40 часов) 
                3. Разработка frontend (приоритет 4, 32 часов)
                4. Тестирование (приоритет 3, 20 часов)
                5. Деплой (приоритет 2, 8 часов)
                
                Дедлайн: 2024-03-01
                """}
            ], response_model=ProjectPlan)
            
            print("✅ План проекта создан успешно:")
            print(f"   🚀 Проект: {project.project_name}")
            print(f"   📄 Описание: {project.description}")
            print(f"   ⏰ Общее время: {project.total_hours} часов")
            print(f"   📅 Дедлайн: {project.deadline}")
            print(f"   📋 Задачи ({len(project.tasks)}):")
            
            for i, task in enumerate(project.tasks, 1):
                print(f"      {i}. {task.title} (приоритет {task.priority}, {task.estimated_hours}ч)")
            
            return project
            
        except Exception as e:
            print(f"❌ Ошибка создания плана проекта: {e}")
            return None


async def demo_user_profile_extraction():
    """Демонстрация извлечения структурированной информации"""
    print("\n👤 Демонстрация извлечения профиля пользователя")
    print("=" * 60)
    
    async with create_advanced_client() as client:
        try:
            print("Извлекаем структурированную информацию из текста...")
            
            user_text = """
            Меня зовут Анна Петрова, мне 28 лет. Работаю Python разработчиком уже 5 лет.
            Мой email: anna.petrova@example.com. Знаю Python, Django, PostgreSQL, Docker, 
            Kubernetes. Также изучаю машинное обучение и data science. Активно участвую 
            в open source проектах.
            """
            
            profile = await client.chat_completion_structured([
                {"role": "system", "content": "Извлекай информацию о пользователе из текста в JSON формате."},
                {"role": "user", "content": f"Извлеки информацию о пользователе из текста:\n\n{user_text}"}
            ], response_model=UserProfile)
            
            print("✅ Профиль пользователя извлечен:")
            print(f"   👤 Имя: {profile.name}")
            print(f"   🎂 Возраст: {profile.age} лет")
            print(f"   📧 Email: {profile.email}")
            print(f"   💼 Опыт: {profile.experience_years} лет")
            print(f"   🛠️ Навыки: {', '.join(profile.skills)}")
            print(f"   ✅ Активен: {profile.is_active}")
            
            return profile
            
        except Exception as e:
            print(f"❌ Ошибка извлечения профиля: {e}")
            return None


async def demo_fallback_behavior():
    """Демонстрация поведения fallback механизмов"""
    print("\n🔄 Демонстрация fallback механизмов")
    print("=" * 60)
    
    # Создаем клиент с только structured output возможностью
    config = UniversalClientConfig(
        capabilities={
            UniversalCapability.CHAT_COMPLETION,
            UniversalCapability.STRUCTURED_OUTPUT
        }
    )
    
    async with UniversalLLMClient(LLMConfig(), config) as client:
        print("Тестируем различные сценарии fallback...")
        
        # Тест 1: Простая задача (должна работать)
        try:
            print("\n1️⃣ Тест простой задачи:")
            task = await client.chat_completion_structured([
                {"role": "user", "content": "Создай задачу: проверить email"}
            ], response_model=Task)
            
            print(f"   ✅ Успех: {task.title}")
            
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
        
        # Тест 2: Сложная модель (может потребовать fallback)
        try:
            print("\n2️⃣ Тест сложной модели:")
            project = await client.chat_completion_structured([
                {"role": "user", "content": "Создай простой проект с 2 задачами"}
            ], response_model=ProjectPlan)
            
            print(f"   ✅ Успех: {project.project_name} с {len(project.tasks)} задачами")
            
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
        
        # Тест 3: Информация о fallback
        info = client.get_client_info()
        print(f"\n📊 Информация о клиенте:")
        print(f"   Возможности: {info['capabilities']}")
        print(f"   Активные клиенты: {info['active_clients']}")


async def demo_performance_comparison():
    """Сравнение производительности разных режимов"""
    print("\n⚡ Сравнение производительности")
    print("=" * 60)
    
    import time
    
    async with create_advanced_client() as client:
        # Тест производительности для простой задачи
        print("Тестируем производительность для простых задач...")
        
        times = []
        for i in range(3):
            start_time = time.time()
            
            try:
                task = await client.chat_completion_structured([
                    {"role": "user", "content": f"Создай задачу номер {i+1}: тестирование производительности"}
                ], response_model=Task)
                
                elapsed = time.time() - start_time
                times.append(elapsed)
                print(f"   Попытка {i+1}: {elapsed:.2f}s - {task.title}")
                
            except Exception as e:
                elapsed = time.time() - start_time
                times.append(elapsed)
                print(f"   Попытка {i+1}: {elapsed:.2f}s - Ошибка: {str(e)[:50]}...")
        
        if times:
            avg_time = sum(times) / len(times)
            print(f"\n📈 Среднее время: {avg_time:.2f}s")
            print(f"📊 Диапазон: {min(times):.2f}s - {max(times):.2f}s")


async def main():
    """Главная функция демонстрации"""
    print("🎯 Structured Output с автоматическим Fallback")
    print("=" * 80)
    print("Демонстрация умного переключения между режимами structured output")
    print()
    
    try:
        # Простой structured output
        task = await demo_simple_structured_output()
        
        # Сложный structured output
        project = await demo_complex_structured_output()
        
        # Извлечение информации
        profile = await demo_user_profile_extraction()
        
        # Демонстрация fallback
        await demo_fallback_behavior()
        
        # Сравнение производительности
        await demo_performance_comparison()
        
        print("\n" + "=" * 80)
        print("🎉 Демонстрация завершена!")
        
        # Сводка результатов
        results = [task, project, profile]
        successful = sum(1 for r in results if r is not None)
        
        print(f"\n📊 Результаты:")
        print(f"   ✅ Успешных операций: {successful}/{len(results)}")
        print(f"   📈 Процент успеха: {successful/len(results)*100:.1f}%")
        
        print(f"\n💡 Ключевые особенности:")
        print(f"   🔄 Автоматический fallback на Outlines режим")
        print(f"   🛡️ Финальный fallback через обычный chat completion")
        print(f"   📝 Улучшенные промпты для лучшего JSON форматирования")
        print(f"   ⚡ Оптимизированная производительность")
        
    except KeyboardInterrupt:
        print("\n❌ Демонстрация прервана пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())