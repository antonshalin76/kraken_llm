#!/usr/bin/env python3
"""
Расширенное тестирование всех возможностей моделей через все доступные конфигурации.
"""

import asyncio
import os
import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from pydantic import BaseModel

from kraken_llm.client.adaptive import AdaptiveLLMClient, AdaptiveConfig, ModelCapability
from kraken_llm.client.factory import ClientFactory
from kraken_llm.config.settings import LLMConfig
from kraken_llm.client.standard import StandardLLMClient
from kraken_llm.client.streaming import StreamingLLMClient
from kraken_llm.client.structured import StructuredLLMClient
from kraken_llm.client.reasoning import ReasoningLLMClient
from kraken_llm.client.multimodal import MultimodalLLMClient
from kraken_llm.client.embeddings import EmbeddingsLLMClient
from kraken_llm.client.completion import CompletionLLMClient

load_dotenv()

@dataclass
class ModelConfig:
    """Конфигурация модели для тестирования"""
    name: str
    endpoint: str
    token: str
    model: str
    description: str

@dataclass
class TestResult:
    """Результат тестирования возможности"""
    capability: str
    client_type: str
    model_config: str
    success: bool
    response_time: float
    error: Optional[str] = None
    response_preview: Optional[str] = None

class TestModel(BaseModel):
    """Тестовая модель для structured output"""
    name: str
    value: int
    description: str

def get_model_configs() -> List[ModelConfig]:
    """Получить все конфигурации моделей из .env"""
    configs = []
    
    # Chat модель (использует /v1/chat/completions)
    if all([os.getenv("CHAT_ENDPOINT"), os.getenv("CHAT_TOKEN"), os.getenv("CHAT_MODEL")]):
        configs.append(ModelConfig(
            name="chat_model",
            endpoint=os.getenv("CHAT_ENDPOINT"),
            token=os.getenv("CHAT_TOKEN"),
            model=os.getenv("CHAT_MODEL"),
            description="Chat модель (/v1/chat/completions)"
        ))
    
    # Completion модель (использует /v1/completions)
    if all([os.getenv("COMPLETION_ENDPOINT"), os.getenv("COMPLETION_TOKEN"), os.getenv("COMPLETION_MODEL")]):
        configs.append(ModelConfig(
            name="completion_model",
            endpoint=os.getenv("COMPLETION_ENDPOINT"),
            token=os.getenv("COMPLETION_TOKEN"),
            model=os.getenv("COMPLETION_MODEL"),
            description="Completion модель (/v1/completions)"
        ))
    
    # Модель для эмбеддингов
    if all([os.getenv("EMBEDDING_ENDPOINT"), os.getenv("EMBEDDING_TOKEN"), os.getenv("EMBEDDING_MODEL")]):
        configs.append(ModelConfig(
            name="embedding_model",
            endpoint=os.getenv("EMBEDDING_ENDPOINT"),
            token=os.getenv("EMBEDDING_TOKEN"),
            model=os.getenv("EMBEDDING_MODEL"),
            description="Модель для эмбеддингов"
        ))
    
    # Для обратной совместимости - основная модель
    if not configs and all([os.getenv("LLM_ENDPOINT"), os.getenv("LLM_TOKEN"), os.getenv("LLM_MODEL")]):
        configs.append(ModelConfig(
            name="default_model",
            endpoint=os.getenv("LLM_ENDPOINT"),
            token=os.getenv("LLM_TOKEN"),
            model=os.getenv("LLM_MODEL"),
            description="Основная модель (обратная совместимость)"
        ))
    
    return configs

async def test_basic_chat_completion(client, model_name: str) -> TestResult:
    """Тестирование базового chat completion"""
    start_time = time.time()
    
    messages = [
        {"role": "user", "content": "Привет! Ответь одним словом: 'Работает'"}
    ]
    
    try:
        response = await client.chat_completion(messages, max_tokens=10)
        response_time = time.time() - start_time
        
        return TestResult(
            capability="chat_completion",
            client_type=type(client).__name__,
            model_config=model_name,
            success=True,
            response_time=response_time,
            response_preview=str(response)[:100] if response else None
        )
    except Exception as e:
        return TestResult(
            capability="chat_completion",
            client_type=type(client).__name__,
            model_config=model_name,
            success=False,
            response_time=time.time() - start_time,
            error=str(e)
        )

async def test_streaming_completion(client, model_name: str) -> TestResult:
    """Тестирование streaming completion"""
    start_time = time.time()
    
    messages = [
        {"role": "user", "content": "Считай от 1 до 5"}
    ]
    
    try:
        # Проверяем, поддерживает ли клиент streaming
        if not hasattr(client, 'chat_completion_stream'):
            return TestResult(
                capability="streaming",
                client_type=type(client).__name__,
                model_config=model_name,
                success=False,
                response_time=time.time() - start_time,
                error=f"{type(client).__name__} не поддерживает streaming"
            )
        
        response_stream = await client.chat_completion_stream(messages, max_tokens=50)
        
        # Собираем streaming ответ
        full_response = ""
        chunk_count = 0
        
        async for chunk in response_stream:
            chunk_count += 1
            
            # Обработка разных форматов ответов
            if hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0:
                choice = chunk.choices[0]
                
                # OpenAI формат
                if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                    content = choice.delta.content
                    if content:
                        full_response += content
                        
                # Альтернативный формат
                elif hasattr(choice, 'text'):
                    full_response += choice.text
                    
            # Простой текстовый формат
            elif isinstance(chunk, str):
                full_response += chunk
                
            # Ограничиваем количество чанков для тестирования
            if chunk_count > 50:
                break
        
        response_time = time.time() - start_time
        
        return TestResult(
            capability="streaming",
            client_type=type(client).__name__,
            model_config=model_name,
            success=True,
            response_time=response_time,
            response_preview=f"Streaming: {chunk_count} чанков, {len(full_response)} символов: {full_response[:50]}..."
        )
        
    except Exception as e:
        return TestResult(
            capability="streaming",
            client_type=type(client).__name__,
            model_config=model_name,
            success=False,
            response_time=time.time() - start_time,
            error=str(e)
        )

async def test_structured_output(client, model_name: str) -> TestResult:
    """Тестирование structured output"""
    start_time = time.time()
    
    messages = [
        {"role": "user", "content": "Верни JSON с name='test', value=42, description='тестовый объект'"}
    ]
    
    try:
        # Проверяем, поддерживает ли клиент structured output
        if hasattr(client, 'chat_completion_structured'):
            # Для StructuredLLMClient и CompletionLLMClient
            response = await client.chat_completion_structured(
                messages, 
                response_model=TestModel,
                max_tokens=100
            )
            response_time = time.time() - start_time
            
            return TestResult(
                capability="structured_output",
                client_type=type(client).__name__,
                model_config=model_name,
                success=True,
                response_time=response_time,
                response_preview=str(response)[:100] if response else None
            )
        else:
            # Для клиентов без structured support - не тестируем
            return TestResult(
                capability="structured_output",
                client_type=type(client).__name__,
                model_config=model_name,
                success=False,
                response_time=time.time() - start_time,
                error=f"{type(client).__name__} не поддерживает structured output. Используйте StructuredLLMClient для структурированных ответов."
            )
        
    except Exception as e:
        return TestResult(
            capability="structured_output",
            client_type=type(client).__name__,
            model_config=model_name,
            success=False,
            response_time=time.time() - start_time,
            error=str(e)
        )

async def test_function_calling(client, model_name: str) -> TestResult:
    """Тестирование function calling"""
    start_time = time.time()
    
    # Простая функция для тестирования
    def get_weather(city: str) -> str:
        return f"Погода в {city}: солнечно, +20°C"
    
    messages = [
        {"role": "user", "content": "Какая погода в Москве?"}
    ]
    
    functions = [{
        "name": "get_weather",
        "description": "Получить информацию о погоде в городе",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "Название города"
                }
            },
            "required": ["city"]
        }
    }]
    
    try:
        if hasattr(client, 'register_function'):
            client.register_function("get_weather", get_weather, "Получить погоду")
        
        response = await client.chat_completion(
            messages, 
            functions=functions,
            max_tokens=100
        )
        response_time = time.time() - start_time
        
        return TestResult(
            capability="function_calling",
            client_type=type(client).__name__,
            model_config=model_name,
            success=True,
            response_time=response_time,
            response_preview=str(response)[:100] if response else None
        )
    except Exception as e:
        return TestResult(
            capability="function_calling",
            client_type=type(client).__name__,
            model_config=model_name,
            success=False,
            response_time=time.time() - start_time,
            error=str(e)
        )

async def test_embeddings(client, model_name: str) -> TestResult:
    """Тестирование embeddings"""
    start_time = time.time()
    
    texts = ["Привет мир", "Hello world", "Тестовый текст"]
    
    try:
        if hasattr(client, 'get_embeddings'):
            response = await client.get_embeddings(texts)
        elif hasattr(client, 'create_embeddings'):
            response = await client.create_embeddings(texts)
        else:
            raise Exception("Embeddings не поддерживаются этим клиентом")
        
        response_time = time.time() - start_time
        
        return TestResult(
            capability="embeddings",
            client_type=type(client).__name__,
            model_config=model_name,
            success=True,
            response_time=response_time,
            response_preview=f"Embeddings для {len(texts)} текстов получены"
        )
    except Exception as e:
        return TestResult(
            capability="embeddings",
            client_type=type(client).__name__,
            model_config=model_name,
            success=False,
            response_time=time.time() - start_time,
            error=str(e)
        )

async def test_reasoning(client, model_name: str) -> TestResult:
    """Тестирование reasoning capabilities"""
    start_time = time.time()
    
    messages = [
        {"role": "user", "content": "Реши задачу: если у меня есть 10 яблок и я съел 3, сколько осталось?"}
    ]
    
    try:
        if hasattr(client, 'reasoning_completion'):
            response = await client.reasoning_completion(
                messages, 
                problem_type="math",
                max_tokens=100
            )
        else:
            response = await client.chat_completion(messages, max_tokens=100)
        
        response_time = time.time() - start_time
        
        return TestResult(
            capability="reasoning",
            client_type=type(client).__name__,
            model_config=model_name,
            success=True,
            response_time=response_time,
            response_preview=str(response)[:100] if response else None
        )
    except Exception as e:
        return TestResult(
            capability="reasoning",
            client_type=type(client).__name__,
            model_config=model_name,
            success=False,
            response_time=time.time() - start_time,
            error=str(e)
        )

async def test_adaptive_capabilities(model_config: ModelConfig) -> List[TestResult]:
    """Тестирование AdaptiveLLMClient возможностей для конкретной модели"""
    print(f"\n=== Тестирование {model_config.name} ({model_config.description}) ===")
    
    config = LLMConfig(
        endpoint=model_config.endpoint,
        api_key=model_config.token,
        model=model_config.model,
        temperature=0.7
    )
    
    adaptive_config = AdaptiveConfig(
        capability_detection_timeout=5.0,
        enable_performance_tracking=True
    )
    
    results = []
    
    try:
        async with AdaptiveLLMClient(config, adaptive_config) as client:
            print("1. Определение возможностей модели:")
            
            # Принудительно обновляем возможности
            capabilities = await client.get_model_capabilities(force_refresh=True)
            print(f"   Обнаруженные возможности: {[cap.value for cap in capabilities]}")
            
            # Получаем детальный отчет
            capabilities_dict = await client.detect_model_capabilities()
            
            print("2. Детальный отчет о возможностях:")
            for capability, supported in capabilities_dict.items():
                status = "✓" if supported else "✗"
                print(f"   {status} {capability}: {supported}")
            
            # Тестируем базовые возможности
            print("3. Тестирование базовых возможностей:")
            
            # Chat completion
            result = await test_basic_chat_completion(client, model_config.name)
            results.append(result)
            print(f"   Chat completion: {'✓' if result.success else '✗'} ({result.response_time:.2f}s)")
            
            # Smart completion
            messages = [{"role": "user", "content": "Привет! Расскажи короткую шутку."}]
            try:
                response = await client.smart_completion(messages, max_tokens=100)
                print(f"   Smart completion: ✓ ({len(str(response))} символов)")
            except Exception as e:
                print(f"   Smart completion: ✗ ({str(e)[:50]}...)")
            
            # Получаем отчет о производительности
            performance_report = client.get_performance_report()
            
            if performance_report["model_info"]:
                model_info = performance_report["model_info"]
                print(f"   Модель: {model_info['name']}")
                print(f"   Провайдер: {model_info['provider']}")
                print(f"   Возможности: {len(model_info['capabilities'])}")
    
    except Exception as e:
        print(f"   ❌ Ошибка тестирования AdaptiveLLMClient: {e}")
        results.append(TestResult(
            capability="adaptive_client",
            client_type="AdaptiveLLMClient",
            model_config=model_config.name,
            success=False,
            response_time=0.0,
            error=str(e)
        ))
    
    return results


def get_suitable_tests_for_client(client_name: str, model_name: str) -> List[tuple]:
    """Получить подходящие тесты для конкретного типа клиента и модели"""
    
    # Базовые тесты для chat/completion клиентов
    base_tests = [
        ("chat_completion", test_basic_chat_completion),
        ("streaming", test_streaming_completion),
        ("function_calling", test_function_calling),
        ("reasoning", test_reasoning),
    ]
    
    # Специфичные тесты для разных клиентов
    if client_name == "structured":
        # StructuredLLMClient - только structured output
        return [("structured_output", test_structured_output)]
    
    elif client_name == "embeddings":
        # Для embeddings клиента только тест эмбеддингов
        if "embedding" in model_name.lower():
            return [("embeddings", test_embeddings)]
        else:
            return []  # Не тестируем embeddings клиент на chat/completion моделях
    
    elif client_name == "completion":
        # Completion клиент поддерживает базовые операции + structured через промпт инжиниринг
        return [
            ("chat_completion", test_basic_chat_completion),
            ("streaming", test_streaming_completion),
            ("structured_output", test_structured_output),  # Через промпт инжиниринг
        ]
    
    elif client_name in ["standard", "streaming", "multimodal"]:
        # Эти клиенты не поддерживают structured output нативно
        tests = base_tests.copy()
            
        # Добавляем embeddings только для embedding моделей
        if "embedding" in model_name.lower():
            tests.append(("embeddings", test_embeddings))
            
        return tests
    
    elif client_name == "reasoning":
        # ReasoningLLMClient не поддерживает structured output
        tests = base_tests.copy()
        
        # Добавляем embeddings только для embedding моделей
        if "embedding" in model_name.lower():
            tests.append(("embeddings", test_embeddings))
            
        return tests
    
    return base_tests

async def test_all_client_types(model_config: ModelConfig) -> List[TestResult]:
    """Тестирование всех типов клиентов для конкретной модели"""
    print(f"\n=== Тестирование всех клиентов для {model_config.name} ===")
    
    config = LLMConfig(
        endpoint=model_config.endpoint,
        api_key=model_config.token,
        model=model_config.model,
        temperature=0.7
    )
    
    all_results = []
    
    # Список клиентов для тестирования
    client_types = [
        ("standard", StandardLLMClient),
        ("streaming", StreamingLLMClient),
        ("structured", StructuredLLMClient),
        ("reasoning", ReasoningLLMClient),
        ("multimodal", MultimodalLLMClient),
        ("embeddings", EmbeddingsLLMClient),
        ("completion", CompletionLLMClient),
    ]
    
    for client_name, client_class in client_types:
        print(f"\n--- Тестирование {client_name} клиента ---")
        
        # Получаем подходящие тесты для этого клиента
        suitable_tests = get_suitable_tests_for_client(client_name, model_config.name)
        
        if not suitable_tests:
            print(f"   ⚠ Нет подходящих тестов для {client_name} клиента с моделью {model_config.name}")
            continue
        
        try:
            async with client_class(config) as client:
                
                for test_name, test_func in suitable_tests:
                    try:
                        result = await test_func(client, model_config.name)
                        all_results.append(result)
                        
                        status = "✓" if result.success else "✗"
                        time_str = f"{result.response_time:.2f}s"
                        
                        if result.success:
                            preview = result.response_preview[:30] + "..." if result.response_preview and len(result.response_preview) > 30 else result.response_preview
                            print(f"   {status} {test_name}: {time_str} - {preview}")
                        else:
                            error_preview = result.error[:50] + "..." if result.error and len(result.error) > 50 else result.error
                            print(f"   {status} {test_name}: {time_str} - {error_preview}")
                            
                    except Exception as e:
                        print(f"   ✗ {test_name}: Ошибка теста - {str(e)[:50]}...")
                        all_results.append(TestResult(
                            capability=test_name,
                            client_type=client_name,
                            model_config=model_config.name,
                            success=False,
                            response_time=0.0,
                            error=str(e)
                        ))
                        
        except Exception as e:
            print(f"   ❌ Не удалось создать {client_name} клиент: {str(e)[:100]}...")
            all_results.append(TestResult(
                capability="client_creation",
                client_type=client_name,
                model_config=model_config.name,
                success=False,
                response_time=0.0,
                error=str(e)
            ))
    
    return all_results

async def test_client_factory(model_config: ModelConfig) -> List[TestResult]:
    """Тестирование ClientFactory"""
    print(f"\n=== Тестирование ClientFactory для {model_config.name} ===")
    
    config = LLMConfig(
        endpoint=model_config.endpoint,
        api_key=model_config.token,
        model=model_config.model,
        temperature=0.7
    )
    
    results = []
    
    # Тестируем создание разных типов клиентов через фабрику
    factory_methods = [
        ("adaptive", ClientFactory.create_adaptive_client),
        ("standard", ClientFactory.create_standard_client),
        ("streaming", ClientFactory.create_streaming_client),
        ("structured", ClientFactory.create_structured_client),
        ("reasoning", ClientFactory.create_reasoning_client),
        ("multimodal", ClientFactory.create_multimodal_client),
        ("embeddings", ClientFactory.create_embeddings_client),
    ]
    
    for method_name, factory_method in factory_methods:
        start_time = time.time()
        
        try:
            client = factory_method(config)
            
            # Простой тест - базовый chat completion
            messages = [{"role": "user", "content": "Тест"}]
            
            if hasattr(client, 'chat_completion'):
                response = await client.chat_completion(messages, max_tokens=10)
                
                results.append(TestResult(
                    capability="factory_creation",
                    client_type=f"Factory_{method_name}",
                    model_config=model_config.name,
                    success=True,
                    response_time=time.time() - start_time,
                    response_preview="Клиент создан и работает"
                ))
                print(f"   ✓ {method_name}: Создан и протестирован")
            else:
                results.append(TestResult(
                    capability="factory_creation",
                    client_type=f"Factory_{method_name}",
                    model_config=model_config.name,
                    success=False,
                    response_time=time.time() - start_time,
                    error="Нет метода chat_completion"
                ))
                print(f"   ✗ {method_name}: Нет метода chat_completion")
                
            await client.close()
            
        except Exception as e:
            results.append(TestResult(
                capability="factory_creation",
                client_type=f"Factory_{method_name}",
                model_config=model_config.name,
                success=False,
                response_time=time.time() - start_time,
                error=str(e)
            ))
            print(f"   ✗ {method_name}: {str(e)[:50]}...")
    
    return results

def print_summary_report(all_results: List[TestResult]):
    """Печать итогового отчета"""
    print("\n" + "="*80)
    print("📊 ИТОГОВЫЙ ОТЧЕТ ТЕСТИРОВАНИЯ")
    print("="*80)
    
    # Группируем результаты по моделям
    by_model = {}
    for result in all_results:
        if result.model_config not in by_model:
            by_model[result.model_config] = []
        by_model[result.model_config].append(result)
    
    for model_name, model_results in by_model.items():
        print(f"\n🔧 Модель: {model_name}")
        print("-" * 50)
        
        # Статистика по возможностям
        capabilities_stats = {}
        for result in model_results:
            if result.capability not in capabilities_stats:
                capabilities_stats[result.capability] = {"success": 0, "total": 0}
            capabilities_stats[result.capability]["total"] += 1
            if result.success:
                capabilities_stats[result.capability]["success"] += 1
        
        print("Возможности:")
        for capability, stats in capabilities_stats.items():
            success_rate = stats["success"] / stats["total"] * 100
            status = "✓" if success_rate > 50 else "⚠" if success_rate > 0 else "✗"
            print(f"  {status} {capability}: {stats['success']}/{stats['total']} ({success_rate:.0f}%)")
        
        # Статистика по клиентам
        client_stats = {}
        for result in model_results:
            if result.client_type not in client_stats:
                client_stats[result.client_type] = {"success": 0, "total": 0}
            client_stats[result.client_type]["total"] += 1
            if result.success:
                client_stats[result.client_type]["success"] += 1
        
        print("\nКлиенты:")
        for client_type, stats in client_stats.items():
            success_rate = stats["success"] / stats["total"] * 100
            status = "✓" if success_rate > 50 else "⚠" if success_rate > 0 else "✗"
            print(f"  {status} {client_type}: {stats['success']}/{stats['total']} ({success_rate:.0f}%)")
    
    # Общая статистика
    total_tests = len(all_results)
    successful_tests = sum(1 for r in all_results if r.success)
    success_rate = successful_tests / total_tests * 100 if total_tests > 0 else 0
    
    print(f"\n📈 ОБЩАЯ СТАТИСТИКА:")
    print(f"   Всего тестов: {total_tests}")
    print(f"   Успешных: {successful_tests}")
    print(f"   Процент успеха: {success_rate:.1f}%")
    
    # Топ ошибок
    errors = [r.error for r in all_results if r.error]
    if errors:
        print(f"\n❌ Частые ошибки:")
        error_counts = {}
        for error in errors:
            short_error = error[:100] + "..." if len(error) > 100 else error
            error_counts[short_error] = error_counts.get(short_error, 0) + 1
        
        for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   {count}x: {error}")

def generate_recommendations(all_results: List[TestResult]) -> Dict[str, List[str]]:
    """Генерация рекомендаций на основе результатов тестирования"""
    recommendations = {
        "chat_model": [],
        "embedding_model": [],
        "general": []
    }
    
    # Анализируем результаты по моделям
    by_model = {}
    for result in all_results:
        if result.model_config not in by_model:
            by_model[result.model_config] = []
        by_model[result.model_config].append(result)
    
    for model_name, model_results in by_model.items():
        model_recs = []
        
        # Анализируем успешные клиенты
        successful_clients = set()
        failed_clients = set()
        
        for result in model_results:
            if result.success:
                successful_clients.add(result.client_type)
            else:
                failed_clients.add(result.client_type)
        
        if model_name == "chat_model":
            if "StandardLLMClient" in successful_clients:
                model_recs.append("✅ Используйте StandardLLMClient для базовых chat completion задач")
            
            if "StructuredLLMClient" in successful_clients:
                model_recs.append("✅ Используйте StructuredLLMClient для получения структурированных ответов")
            
            if "ReasoningLLMClient" in successful_clients:
                model_recs.append("✅ Используйте ReasoningLLMClient для задач, требующих рассуждений")
            
            if "streaming" in [r.capability for r in model_results if not r.success]:
                model_recs.append("⚠️ Streaming может работать нестабильно - проверьте реализацию")
            
            if "function_calling" in [r.capability for r in model_results if r.success]:
                model_recs.append("✅ Function calling поддерживается - можно использовать инструменты")
        
        elif model_name == "embedding_model":
            if "EmbeddingsLLMClient" in successful_clients:
                model_recs.append("✅ Используйте EmbeddingsLLMClient для работы с эмбеддингами")
            
            if any(r.capability == "chat_completion" and not r.success for r in model_results):
                model_recs.append("ℹ️ Эта модель предназначена только для эмбеддингов, не для чата")
        
        recommendations[model_name] = model_recs
    
    # Общие рекомендации
    general_recs = []
    
    # Проверяем общую статистику успеха
    total_tests = len(all_results)
    successful_tests = sum(1 for r in all_results if r.success)
    success_rate = successful_tests / total_tests * 100 if total_tests > 0 else 0
    
    if success_rate > 70:
        general_recs.append("🎉 Отличная совместимость! Большинство функций работает корректно")
    elif success_rate > 40:
        general_recs.append("⚠️ Умеренная совместимость. Рекомендуется использовать проверенные клиенты")
    else:
        general_recs.append("❌ Низкая совместимость. Проверьте конфигурацию и доступность сервисов")
    
    # Анализируем частые ошибки
    errors = [r.error for r in all_results if r.error]
    if "HTTP 404" in str(errors):
        general_recs.append("🔧 Обнаружены ошибки 404 - проверьте правильность endpoint'ов")
    
    if "response_model обязателен" in str(errors):
        general_recs.append("📝 Для StructuredLLMClient всегда указывайте response_model")
    
    if "не поддерживает" in str(errors):
        general_recs.append("🎯 Используйте специализированные клиенты для конкретных задач")
    
    recommendations["general"] = general_recs
    
    return recommendations

def generate_capabilities_summary(all_results: List[TestResult]) -> Dict[str, Dict[str, List[str]]]:
    """Генерация сводки подтвержденных возможностей для каждой модели"""
    summary = {}
    
    # Группируем результаты по моделям
    by_model = {}
    for result in all_results:
        if result.model_config not in by_model:
            by_model[result.model_config] = []
        by_model[result.model_config].append(result)
    
    for model_name, model_results in by_model.items():
        model_summary = {
            "confirmed_capabilities": [],
            "working_clients": [],
            "failed_capabilities": [],
            "statistics": {}
        }
        
        # Анализируем возможности
        capabilities_status = {}
        clients_status = {}
        
        for result in model_results:
            # Статистика по возможностям
            if result.capability not in capabilities_status:
                capabilities_status[result.capability] = {"success": 0, "total": 0}
            capabilities_status[result.capability]["total"] += 1
            if result.success:
                capabilities_status[result.capability]["success"] += 1
            
            # Статистика по клиентам
            if result.client_type not in clients_status:
                clients_status[result.client_type] = {"success": 0, "total": 0}
            clients_status[result.client_type]["total"] += 1
            if result.success:
                clients_status[result.client_type]["success"] += 1
        
        # Определяем подтвержденные возможности (успешность > 50%)
        for capability, stats in capabilities_status.items():
            success_rate = stats["success"] / stats["total"] * 100
            if success_rate > 50:
                model_summary["confirmed_capabilities"].append(
                    f"{capability} ({stats['success']}/{stats['total']}, {success_rate:.0f}%)"
                )
            else:
                model_summary["failed_capabilities"].append(
                    f"{capability} ({stats['success']}/{stats['total']}, {success_rate:.0f}%)"
                )
        
        # Определяем работающие клиенты
        for client_type, stats in clients_status.items():
            success_rate = stats["success"] / stats["total"] * 100
            if success_rate > 50:
                model_summary["working_clients"].append(
                    f"{client_type} ({stats['success']}/{stats['total']}, {success_rate:.0f}%)"
                )
        
        # Общая статистика
        total_tests = len(model_results)
        successful_tests = sum(1 for r in model_results if r.success)
        model_summary["statistics"] = {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": successful_tests / total_tests * 100 if total_tests > 0 else 0
        }
        
        summary[model_name] = model_summary
    
    return summary

def print_capabilities_summary(summary: Dict[str, Dict[str, List[str]]]):
    """Печать сводки возможностей моделей"""
    print("\n" + "="*80)
    print("📋 СВОДНЫЙ ИТОГ: ПОДТВЕРЖДЕННЫЕ ВОЗМОЖНОСТИ МОДЕЛЕЙ")
    print("="*80)
    
    for model_name, model_data in summary.items():
        print(f"\n🎯 МОДЕЛЬ: {model_name.upper()}")
        print("-" * 60)
        
        stats = model_data["statistics"]
        print(f"📊 Общая статистика: {stats['successful_tests']}/{stats['total_tests']} тестов успешно ({stats['success_rate']:.1f}%)")
        
        if model_data["confirmed_capabilities"]:
            print(f"\n✅ ПОДТВЕРЖДЕННЫЕ ВОЗМОЖНОСТИ:")
            for capability in model_data["confirmed_capabilities"]:
                print(f"   • {capability}")
        else:
            print(f"\n❌ Подтвержденных возможностей не найдено")
        
        if model_data["working_clients"]:
            print(f"\n🔧 РАБОЧИЕ КЛИЕНТЫ:")
            for client in model_data["working_clients"]:
                print(f"   • {client}")
        
        if model_data["failed_capabilities"]:
            print(f"\n⚠️ ПРОБЛЕМНЫЕ ВОЗМОЖНОСТИ:")
            for capability in model_data["failed_capabilities"]:
                print(f"   • {capability}")

def print_recommendations(recommendations: Dict[str, List[str]]):
    """Печать рекомендаций"""
    print("\n" + "="*80)
    print("💡 РЕКОМЕНДАЦИИ ПО ИСПОЛЬЗОВАНИЮ")
    print("="*80)
    
    for model_name, recs in recommendations.items():
        if recs and model_name != "general":
            print(f"\n🔧 {model_name.upper()}:")
            for rec in recs:
                print(f"   {rec}")
    
    if recommendations.get("general"):
        print(f"\n🌟 ОБЩИЕ РЕКОМЕНДАЦИИ:")
        for rec in recommendations["general"]:
            print(f"   {rec}")

def save_results_to_json(all_results: List[TestResult], filename: str = "test_results.json"):
    """Сохранение результатов в JSON файл"""
    results_data = []
    for result in all_results:
        results_data.append({
            "capability": result.capability,
            "client_type": result.client_type,
            "model_config": result.model_config,
            "success": result.success,
            "response_time": result.response_time,
            "error": result.error,
            "response_preview": result.response_preview,
            "timestamp": time.time()
        })
    
    # Добавляем рекомендации и сводку возможностей
    recommendations = generate_recommendations(all_results)
    capabilities_summary = generate_capabilities_summary(all_results)
    
    final_data = {
        "test_results": results_data,
        "capabilities_summary": capabilities_summary,
        "recommendations": recommendations,
        "summary": {
            "total_tests": len(all_results),
            "successful_tests": sum(1 for r in all_results if r.success),
            "success_rate": sum(1 for r in all_results if r.success) / len(all_results) * 100 if all_results else 0,
            "test_timestamp": time.time()
        }
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Результаты сохранены в {filename}")


async def main():
    """Главная функция расширенного тестирования."""
    print("🚀 РАСШИРЕННОЕ ТЕСТИРОВАНИЕ ВСЕХ ВОЗМОЖНОСТЕЙ МОДЕЛЕЙ")
    print("=" * 80)
    print("Тестируем все модели через все доступные клиенты и возможности")
    print("=" * 80)
    
    # Получаем все конфигурации моделей
    model_configs = get_model_configs()
    
    if not model_configs:
        print("❌ Не найдено конфигураций моделей в .env файле")
        return
    
    print(f"📋 Найдено {len(model_configs)} конфигураций моделей:")
    for config in model_configs:
        print(f"   • {config.name}: {config.description}")
        print(f"     Endpoint: {config.endpoint}")
        print(f"     Model: {config.model}")
    
    all_results = []
    
    try:
        for model_config in model_configs:
            print(f"\n🔄 Начинаем тестирование модели: {model_config.name}")
            
            # 1. Тестируем AdaptiveLLMClient
            adaptive_results = await test_adaptive_capabilities(model_config)
            all_results.extend(adaptive_results)
            
            # 2. Тестируем все типы клиентов
            client_results = await test_all_client_types(model_config)
            all_results.extend(client_results)
            
            # 3. Тестируем ClientFactory
            factory_results = await test_client_factory(model_config)
            all_results.extend(factory_results)
            
            print(f"✅ Завершено тестирование модели: {model_config.name}")
        
        # Печатаем итоговый отчет
        print_summary_report(all_results)
        
        # Генерируем и показываем сводный итог возможностей
        capabilities_summary = generate_capabilities_summary(all_results)
        print_capabilities_summary(capabilities_summary)
        
        # Генерируем и показываем рекомендации
        recommendations = generate_recommendations(all_results)
        print_recommendations(recommendations)
        
        # Сохраняем результаты
        save_results_to_json(all_results, "kraken_capabilities_test_results.json")
        
        print("\n🎉 ТЕСТИРОВАНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        
    except Exception as e:
        print(f"\n❌ Критическая ошибка тестирования: {e}")
        import traceback
        traceback.print_exc()
        
        # Все равно сохраняем то, что успели протестировать
        if all_results:
            save_results_to_json(all_results, "kraken_capabilities_test_results_partial.json")


if __name__ == "__main__":
    asyncio.run(main())