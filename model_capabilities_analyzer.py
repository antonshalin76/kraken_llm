#!/usr/bin/env python3
"""
Kraken LLM Model Capabilities Analyzer

Корневая утилита для комплексного анализа возможностей LLM моделей.
Автоматически определяет поддерживаемые функции, оптимальные клиенты
и генерирует детальные рекомендации по использованию.

Возможности:
- Полный анализ всех типов клиентов Kraken LLM
- Определение поддерживаемых возможностей моделей
- Тестирование производительности и совместимости
- Генерация рекомендаций по использованию
- Экспорт результатов в JSON и Markdown форматы
- Интеграция с CI/CD пайплайнами

Использование:
    python model_capabilities_analyzer.py [--quick] [--output FORMAT] [--config FILE]

Автор: Kraken LLM Framework Team
Версия: 1.0.0
"""

import asyncio
import json
import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from dotenv import load_dotenv
from pydantic import BaseModel

# Добавляем путь к модулям Kraken LLM
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Импорты Kraken LLM
try:
    from kraken_llm.config.settings import LLMConfig
    from kraken_llm.client.factory import ClientFactory
    from kraken_llm.client.adaptive import AdaptiveLLMClient, AdaptiveConfig, ModelCapability
    from kraken_llm.client.standard import StandardLLMClient
    from kraken_llm.client.streaming import StreamingLLMClient
    from kraken_llm.client.structured import StructuredLLMClient
    from kraken_llm.client.reasoning import ReasoningLLMClient, ReasoningConfig, ReasoningModelType
    from kraken_llm.client.multimodal import MultimodalLLMClient, MultimodalConfig
    from kraken_llm.client.asr import ASRClient, ASRConfig
    from kraken_llm.client.embeddings import EmbeddingsLLMClient
    from kraken_llm.client.completion import CompletionLLMClient
    from kraken_llm.exceptions.base import KrakenError
    from kraken_llm.tools import register_function, register_tool
    from kraken_llm.utils.media import MediaUtils
except ImportError as e:
    print(f"❌ Ошибка импорта Kraken LLM: {e}")
    print("Убедитесь, что Kraken LLM установлен: pip install -e .")
    sys.exit(1)

# Загружаем переменные окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_capabilities_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CapabilityType(str, Enum):
    """Типы возможностей модели"""
    # Базовые возможности
    CHAT_COMPLETION = "chat_completion"
    STREAMING = "streaming"
    COMPLETION_LEGACY = "completion_legacy"
    
    # Структурированный вывод
    STRUCTURED_OUTPUT_NATIVE = "structured_output_native"
    STRUCTURED_OUTPUT_OUTLINES = "structured_output_outlines"
    STRUCTURED_OUTPUT_JSON = "structured_output_json"
    
    # Вызов функций и инструментов
    FUNCTION_CALLING = "function_calling"
    TOOL_CALLING = "tool_calling"
    
    # Векторные представления
    EMBEDDINGS = "embeddings"
    SIMILARITY_SEARCH = "similarity_search"
    
    # Рассуждения
    REASONING_COT = "reasoning_cot"
    REASONING_NATIVE_THINKING = "reasoning_native_thinking"
    REASONING_STREAMING = "reasoning_streaming"
    
    # Мультимодальность
    MULTIMODAL_VISION = "multimodal_vision"
    MULTIMODAL_AUDIO = "multimodal_audio"
    MULTIMODAL_VIDEO = "multimodal_video"
    MULTIMODAL_MIXED = "multimodal_mixed"
    
    # Речевые технологии
    ASR_STT = "asr_stt"
    ASR_TTS = "asr_tts"
    ASR_DIARIZATION = "asr_diarization"
    ASR_EMOTION_ANALYSIS = "asr_emotion_analysis"
    ASR_VAD = "asr_vad"
    
    # Адаптивные возможности
    ADAPTIVE_MODE = "adaptive_mode"
    CAPABILITY_DETECTION = "capability_detection"
    SMART_COMPLETION = "smart_completion"
    
    # Производительность
    PARALLEL_PROCESSING = "parallel_processing"
    BATCH_PROCESSING = "batch_processing"
    STREAMING_AGGREGATION = "streaming_aggregation"


class ClientType(str, Enum):
    """Типы клиентов Kraken LLM"""
    STANDARD = "StandardLLMClient"
    STREAMING = "StreamingLLMClient"
    STRUCTURED = "StructuredLLMClient"
    REASONING = "ReasoningLLMClient"
    MULTIMODAL = "MultimodalLLMClient"
    ADAPTIVE = "AdaptiveLLMClient"
    ASR = "ASRClient"
    EMBEDDINGS = "EmbeddingsLLMClient"
    COMPLETION = "CompletionLLMClient"


class ModelType(str, Enum):
    """Типы моделей"""
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    MULTIMODAL = "multimodal"
    ASR = "asr"
    REASONING = "reasoning"
    UNIVERSAL = "universal"


@dataclass
class ModelConfig:
    """Конфигурация модели для анализа"""
    name: str
    endpoint: str
    api_key: str
    model: str
    description: str
    model_type: ModelType = ModelType.CHAT
    provider: str = "unknown"
    version: Optional[str] = None
    context_length: Optional[int] = None
    supports_system_messages: bool = True
    supports_streaming: bool = True
    max_tokens_limit: Optional[int] = None


@dataclass
class TestResult:
    """Результат тестирования возможности"""
    capability: CapabilityType
    client_type: ClientType
    model_name: str
    success: bool
    response_time: float
    confidence_score: float = 0.0  # 0.0-1.0, насколько уверены в результате
    error_message: Optional[str] = None
    response_preview: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, float]] = None


class TestModel(BaseModel):
    """Pydantic модель для тестирования structured output"""
    name: str
    value: int
    description: str
    tags: List[str] = []
    confidence: float = 0.0
    metadata: Dict[str, Any] = {}


class ModelCapabilitiesAnalyzer:
    """
    Анализатор возможностей моделей Kraken LLM.
    
    Выполняет комплексный анализ всех доступных моделей,
    определяет их возможности и генерирует рекомендации.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Инициализация анализатора.
        
        Args:
            config_file: Путь к файлу конфигурации (опционально)
        """
        self.results: List[TestResult] = []
        self.model_configs: List[ModelConfig] = []
        self.start_time = time.time()
        self.config_file = config_file
        
        # Статистика
        self.stats = {
            "total_tests": 0,
            "successful_tests": 0,
            "failed_tests": 0,
            "models_tested": 0,
            "clients_tested": 0,
            "capabilities_found": set(),
            "execution_time": 0.0
        }
        
        self._load_model_configs()
        logger.info(f"Инициализирован анализатор для {len(self.model_configs)} моделей")   
 
    def _load_model_configs(self) -> None:
        """Загрузка конфигураций моделей из переменных окружения"""
        configs = []
        
        # Определяем все возможные модели из переменных окружения
        model_patterns = [
            ("CHAT", ModelType.CHAT, "Chat модель для диалогов"),
            ("COMPLETION", ModelType.COMPLETION, "Completion модель для автодополнения"),
            ("EMBEDDING", ModelType.EMBEDDING, "Модель для векторных представлений"),
            ("MULTIMODAL", ModelType.MULTIMODAL, "Мультимодальная модель"),
            ("ASR", ModelType.ASR, "ASR модель для речевых технологий"),
            ("REASONING", ModelType.REASONING, "Рассуждающая модель"),
            ("LLM", ModelType.UNIVERSAL, "Универсальная модель (обратная совместимость)")
        ]
        
        for prefix, model_type, description in model_patterns:
            endpoint_key = f"{prefix}_ENDPOINT"
            token_key = f"{prefix}_TOKEN" if prefix != "LLM" else f"{prefix}_API_KEY"
            model_key = f"{prefix}_MODEL"
            
            # Альтернативные ключи для токенов
            alt_token_keys = [f"{prefix}_API_KEY", f"{prefix}_KEY"]
            
            endpoint = os.getenv(endpoint_key)
            model_name = os.getenv(model_key)
            
            # Ищем токен по разным ключам
            token = os.getenv(token_key)
            if not token:
                for alt_key in alt_token_keys:
                    token = os.getenv(alt_key)
                    if token:
                        break
            
            if endpoint and model_name and token:
                # Определяем провайдера по endpoint
                provider = self._detect_provider(endpoint)
                
                config = ModelConfig(
                    name=f"{prefix.lower()}_model",
                    endpoint=endpoint,
                    api_key=token,
                    model=model_name,
                    description=description,
                    model_type=model_type,
                    provider=provider,
                    # Дополнительные параметры из переменных окружения
                    version=os.getenv(f"{prefix}_VERSION"),
                    context_length=self._parse_int_env(f"{prefix}_CONTEXT_LENGTH"),
                    max_tokens_limit=self._parse_int_env(f"{prefix}_MAX_TOKENS")
                )
                configs.append(config)
                logger.info(f"Загружена конфигурация: {config.name} ({config.provider})")
        
        if not configs:
            logger.warning("Не найдено конфигураций моделей в переменных окружения")
            logger.info("Пример настройки:")
            logger.info("  CHAT_ENDPOINT=http://localhost:8080")
            logger.info("  CHAT_TOKEN=your_token")
            logger.info("  CHAT_MODEL=chat")
        
        self.model_configs = configs
    
    def _detect_provider(self, endpoint: str) -> str:
        """Определение провайдера по endpoint"""
        endpoint_lower = endpoint.lower()
        
        if "openai" in endpoint_lower:
            return "OpenAI"
        elif "anthropic" in endpoint_lower:
            return "Anthropic"
        elif "huggingface" in endpoint_lower or "hf" in endpoint_lower:
            return "HuggingFace"
        elif "ollama" in endpoint_lower:
            return "Ollama"
        elif "localhost" in endpoint_lower or "127.0.0.1" in endpoint_lower:
            return "Local"
        elif any(ip in endpoint_lower for ip in ["10.", "192.168.", "172."]):
            return "Private Network"
        else:
            return "Unknown"
    
    def _parse_int_env(self, key: str) -> Optional[int]:
        """Парсинг целого числа из переменной окружения"""
        value = os.getenv(key)
        if value:
            try:
                return int(value)
            except ValueError:
                logger.warning(f"Некорректное значение для {key}: {value}")
        return None
    
    async def analyze_all_models(self, quick_mode: bool = False) -> Dict[str, Any]:
        """
        Анализ всех моделей.
        
        Args:
            quick_mode: Быстрый режим (только основные тесты)
            
        Returns:
            Результаты анализа
        """
        if not self.model_configs:
            logger.error("Нет моделей для анализа")
            return {"error": "No models configured"}
        
        logger.info(f"Начинается анализ {len(self.model_configs)} моделей")
        logger.info(f"Режим: {'быстрый' if quick_mode else 'полный'}")
        
        self.stats["models_tested"] = len(self.model_configs)
        
        for model_config in self.model_configs:
            logger.info(f"Анализ модели: {model_config.name}")
            
            try:
                # Анализируем модель
                model_results = await self._analyze_single_model(model_config, quick_mode)
                self.results.extend(model_results)
                
                # Обновляем статистику
                for result in model_results:
                    self.stats["total_tests"] += 1
                    if result.success:
                        self.stats["successful_tests"] += 1
                        self.stats["capabilities_found"].add(result.capability.value)
                    else:
                        self.stats["failed_tests"] += 1
                
                logger.info(f"Завершен анализ модели: {model_config.name}")
                
            except Exception as e:
                logger.error(f"Ошибка анализа модели {model_config.name}: {e}")
                # Добавляем результат с ошибкой
                self.results.append(TestResult(
                    capability=CapabilityType.CHAT_COMPLETION,
                    client_type=ClientType.STANDARD,
                    model_name=model_config.name,
                    success=False,
                    response_time=0.0,
                    error_message=str(e)
                ))
        
        self.stats["execution_time"] = time.time() - self.start_time
        self.stats["clients_tested"] = len(set(r.client_type for r in self.results))
        
        logger.info(f"Анализ завершен за {self.stats['execution_time']:.2f}s")
        
        return self._generate_analysis_report()  
  
    async def _analyze_single_model(self, model_config: ModelConfig, quick_mode: bool) -> List[TestResult]:
        """Анализ одной модели"""
        results = []
        
        # Создаем базовую конфигурацию LLM
        llm_config = LLMConfig(
            endpoint=model_config.endpoint,
            api_key=model_config.api_key,
            model=model_config.model,
            temperature=0.7,
            max_tokens=100  # Для тестов используем небольшое значение
        )
        
        # Определяем какие клиенты тестировать
        clients_to_test = self._get_clients_for_model(model_config, quick_mode)
        
        for client_type, client_class, client_config in clients_to_test:
            logger.info(f"  Тестирование {client_type.value}")
            
            try:
                # Создаем клиент с дополнительной конфигурацией если нужно
                if client_config:
                    client = client_class(llm_config, client_config)
                else:
                    client = client_class(llm_config)
                
                # Получаем тесты для этого клиента
                tests_to_run = self._get_tests_for_client(client_type, model_config, quick_mode)
                
                async with client:
                    for test_method, capability in tests_to_run:
                        try:
                            result = await test_method(client, model_config, capability)
                            results.append(result)
                            
                            status = "✅" if result.success else "❌"
                            logger.info(f"    {status} {capability.value}: {result.response_time:.2f}s")
                            
                        except Exception as e:
                            logger.error(f"    ❌ {capability.value}: {str(e)[:50]}...")
                            results.append(TestResult(
                                capability=capability,
                                client_type=client_type,
                                model_name=model_config.name,
                                success=False,
                                response_time=0.0,
                                error_message=str(e)
                            ))
                
            except Exception as e:
                logger.error(f"  ❌ Не удалось создать {client_type.value}: {str(e)[:50]}...")
                results.append(TestResult(
                    capability=CapabilityType.CHAT_COMPLETION,
                    client_type=client_type,
                    model_name=model_config.name,
                    success=False,
                    response_time=0.0,
                    error_message=f"Client creation failed: {str(e)}"
                ))
        
        # Специальный тест для AdaptiveLLMClient
        if not quick_mode or model_config.model_type in [ModelType.CHAT, ModelType.UNIVERSAL]:
            adaptive_results = await self._test_adaptive_client(model_config)
            results.extend(adaptive_results)
        
        return results
    
    def _get_clients_for_model(self, model_config: ModelConfig, quick_mode: bool) -> List[Tuple[ClientType, Any, Any]]:
        """Определение клиентов для тестирования конкретной модели"""
        clients = []
        
        # Базовые клиенты для всех типов моделей (кроме embedding)
        if model_config.model_type != ModelType.EMBEDDING:
            clients.extend([
                (ClientType.STANDARD, StandardLLMClient, None),
                (ClientType.STREAMING, StreamingLLMClient, None),
            ])
            
            if not quick_mode:
                clients.extend([
                    (ClientType.STRUCTURED, StructuredLLMClient, None),
                    (ClientType.REASONING, ReasoningLLMClient, ReasoningConfig(
                        model_type=ReasoningModelType.PROMPT_BASED,
                        enable_cot=True
                    )),
                    (ClientType.MULTIMODAL, MultimodalLLMClient, MultimodalConfig()),
                ])
        
        # Специализированные клиенты
        if model_config.model_type == ModelType.EMBEDDING:
            clients.append((ClientType.EMBEDDINGS, EmbeddingsLLMClient, None))
        
        if model_config.model_type == ModelType.COMPLETION:
            clients.append((ClientType.COMPLETION, CompletionLLMClient, None))
        
        if model_config.model_type == ModelType.ASR and not quick_mode:
            clients.append((ClientType.ASR, ASRClient, ASRConfig()))
        
        return clients
    
    def _get_tests_for_client(self, client_type: ClientType, model_config: ModelConfig, quick_mode: bool) -> List[Tuple[Any, CapabilityType]]:
        """Получение тестов для конкретного клиента"""
        tests = []
        
        # Базовые тесты
        if client_type in [ClientType.STANDARD, ClientType.STREAMING, ClientType.REASONING, ClientType.MULTIMODAL]:
            tests.append((self._test_chat_completion, CapabilityType.CHAT_COMPLETION))
            
            if client_type == ClientType.STREAMING:
                tests.append((self._test_streaming, CapabilityType.STREAMING))
            
            if not quick_mode:
                tests.extend([
                    (self._test_function_calling, CapabilityType.FUNCTION_CALLING),
                    (self._test_tool_calling, CapabilityType.TOOL_CALLING),
                ])
        
        # Специализированные тесты
        if client_type == ClientType.STRUCTURED:
            tests.extend([
                (self._test_structured_output_native, CapabilityType.STRUCTURED_OUTPUT_NATIVE),
            ])
            if not quick_mode:
                tests.append((self._test_structured_output_outlines, CapabilityType.STRUCTURED_OUTPUT_OUTLINES))
        
        if client_type == ClientType.EMBEDDINGS:
            tests.append((self._test_embeddings, CapabilityType.EMBEDDINGS))
            if not quick_mode:
                tests.append((self._test_similarity_search, CapabilityType.SIMILARITY_SEARCH))
        
        if client_type == ClientType.REASONING and not quick_mode:
            tests.extend([
                (self._test_reasoning_cot, CapabilityType.REASONING_COT),
                (self._test_reasoning_native_thinking, CapabilityType.REASONING_NATIVE_THINKING),
            ])
        
        if client_type == ClientType.MULTIMODAL and not quick_mode:
            tests.append((self._test_multimodal_vision, CapabilityType.MULTIMODAL_VISION))
        
        if client_type == ClientType.ASR and not quick_mode:
            tests.extend([
                (self._test_asr_stt, CapabilityType.ASR_STT),
                (self._test_asr_tts, CapabilityType.ASR_TTS),
            ])
        
        if client_type == ClientType.COMPLETION:
            tests.append((self._test_completion_legacy, CapabilityType.COMPLETION_LEGACY))
        
        return tests    

    # Методы тестирования возможностей
    
    async def _test_chat_completion(self, client, model_config: ModelConfig, capability: CapabilityType) -> TestResult:
        """Тест базового chat completion"""
        start_time = time.time()
        
        messages = [
            {"role": "user", "content": "Ответь одним словом: 'Работает'"}
        ]
        
        try:
            response = await client.chat_completion(messages, max_tokens=10)
            response_time = time.time() - start_time
            
            # Оценка качества ответа
            response_str = str(response).lower()
            confidence = 1.0 if "работает" in response_str else 0.7
            
            return TestResult(
                capability=capability,
                client_type=ClientType(type(client).__name__),
                model_name=model_config.name,
                success=True,
                response_time=response_time,
                confidence_score=confidence,
                response_preview=str(response)[:100] if response else None,
                performance_metrics={"tokens_per_second": 10 / response_time if response_time > 0 else 0}
            )
        except Exception as e:
            return TestResult(
                capability=capability,
                client_type=ClientType(type(client).__name__),
                model_name=model_config.name,
                success=False,
                response_time=time.time() - start_time,
                confidence_score=0.0,
                error_message=str(e)
            )
    
    async def _test_streaming(self, client, model_config: ModelConfig, capability: CapabilityType) -> TestResult:
        """Тест потокового режима"""
        start_time = time.time()
        
        messages = [
            {"role": "user", "content": "Считай от 1 до 3"}
        ]
        
        try:
            if not hasattr(client, 'chat_completion_stream'):
                return TestResult(
                    capability=capability,
                    client_type=ClientType(type(client).__name__),
                    model_name=model_config.name,
                    success=False,
                    response_time=time.time() - start_time,
                    confidence_score=0.0,
                    error_message=f"{type(client).__name__} не поддерживает streaming"
                )
            
            response_stream = client.chat_completion_stream(messages, max_tokens=50)
            
            full_response = ""
            chunk_count = 0
            
            # Получаем первые несколько chunks для тестирования
            async def collect_chunks():
                nonlocal full_response, chunk_count
                async for chunk in response_stream:
                    chunk_count += 1
                    
                    # Обработка разных форматов chunks
                    if hasattr(chunk, 'choices') and chunk.choices:
                        choice = chunk.choices[0]
                        if hasattr(choice, 'delta') and hasattr(choice.delta, 'content') and choice.delta.content:
                            full_response += choice.delta.content
                        elif hasattr(choice, 'text'):
                            full_response += choice.text
                    elif isinstance(chunk, str):
                        full_response += chunk
                    
                    # Ограничиваем количество chunks для тестирования
                    if chunk_count >= 5:
                        break
            
            await asyncio.wait_for(collect_chunks(), timeout=15.0)
            response_time = time.time() - start_time
            
            # Оценка качества streaming
            confidence = min(1.0, chunk_count / 3.0)  # Ожидаем минимум 3 chunk'а
            
            return TestResult(
                capability=capability,
                client_type=ClientType(type(client).__name__),
                model_name=model_config.name,
                success=chunk_count > 0,
                response_time=response_time,
                confidence_score=confidence,
                response_preview=f"Streaming: {chunk_count} chunks, {len(full_response)} chars",
                performance_metrics={
                    "chunks_per_second": chunk_count / response_time if response_time > 0 else 0,
                    "chars_per_second": len(full_response) / response_time if response_time > 0 else 0
                }
            )
            
        except asyncio.TimeoutError:
            return TestResult(
                capability=capability,
                client_type=ClientType(type(client).__name__),
                model_name=model_config.name,
                success=False,
                response_time=time.time() - start_time,
                confidence_score=0.0,
                error_message="Streaming timeout"
            )
        except Exception as e:
            return TestResult(
                capability=capability,
                client_type=ClientType(type(client).__name__),
                model_name=model_config.name,
                success=False,
                response_time=time.time() - start_time,
                confidence_score=0.0,
                error_message=str(e)
            )
    
    async def _test_structured_output_native(self, client, model_config: ModelConfig, capability: CapabilityType) -> TestResult:
        """Тест нативного structured output"""
        start_time = time.time()
        
        messages = [
            {"role": "system", "content": "Отвечай только валидным JSON без дополнительного текста."},
            {"role": "user", "content": 'Создай JSON объект: {"name": "test_user", "value": 42, "description": "test object", "tags": ["demo"], "confidence": 0.95}'}
        ]
        
        try:
            if not hasattr(client, 'chat_completion_structured'):
                return TestResult(
                    capability=capability,
                    client_type=ClientType(type(client).__name__),
                    model_name=model_config.name,
                    success=False,
                    response_time=time.time() - start_time,
                    confidence_score=0.0,
                    error_message=f"{type(client).__name__} не поддерживает structured output"
                )
            
            response = await client.chat_completion_structured(
                messages,
                response_model=TestModel,
                max_tokens=200,
                stream=False  # Нативный режим
            )
            response_time = time.time() - start_time
            
            # Проверяем качество structured output
            confidence = 1.0
            if not isinstance(response, TestModel):
                confidence = 0.5
            elif response.name != "test_user" or response.value != 42:
                confidence = 0.7
            
            return TestResult(
                capability=capability,
                client_type=ClientType(type(client).__name__),
                model_name=model_config.name,
                success=True,
                response_time=response_time,
                confidence_score=confidence,
                response_preview=f"Native SO: {response.name}, {response.value}",
                metadata={"response_type": type(response).__name__}
            )
            
        except Exception as e:
            return TestResult(
                capability=capability,
                client_type=ClientType(type(client).__name__),
                model_name=model_config.name,
                success=False,
                response_time=time.time() - start_time,
                confidence_score=0.0,
                error_message=str(e)
            )    
 
    async def _test_structured_output_outlines(self, client, model_config: ModelConfig, capability: CapabilityType) -> TestResult:
        """Тест structured output через Outlines"""
        start_time = time.time()
        
        messages = [
            {"role": "user", "content": 'Создай объект пользователя с полями name="outlines_test", value=99, description="outlines mode test"'}
        ]
        
        try:
            if not isinstance(client, StructuredLLMClient):
                return TestResult(
                    capability=capability,
                    client_type=ClientType(type(client).__name__),
                    model_name=model_config.name,
                    success=False,
                    response_time=time.time() - start_time,
                    confidence_score=0.0,
                    error_message="Outlines режим доступен только для StructuredLLMClient"
                )
            
            # Пытаемся использовать outlines режим
            response = await client._structured_stream_outlines(
                messages,
                response_model=TestModel,
                max_tokens=200
            )
            response_time = time.time() - start_time
            
            confidence = 1.0 if isinstance(response, TestModel) else 0.5
            
            return TestResult(
                capability=capability,
                client_type=ClientType(type(client).__name__),
                model_name=model_config.name,
                success=True,
                response_time=response_time,
                confidence_score=confidence,
                response_preview=f"Outlines SO: {response.name if hasattr(response, 'name') else str(response)[:50]}",
                metadata={"mode": "outlines"}
            )
            
        except Exception as e:
            return TestResult(
                capability=capability,
                client_type=ClientType(type(client).__name__),
                model_name=model_config.name,
                success=False,
                response_time=time.time() - start_time,
                confidence_score=0.0,
                error_message=str(e)
            )
    
    async def _test_function_calling(self, client, model_config: ModelConfig, capability: CapabilityType) -> TestResult:
        """Тест function calling"""
        start_time = time.time()
        
        # Регистрируем тестовую функцию
        @register_function(description="Получить информацию о погоде")
        def get_weather(city: str) -> str:
            return f"Погода в {city}: солнечно, +20°C"
        
        messages = [
            {"role": "user", "content": "Какая погода в Москве? Используй функцию get_weather."}
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
                function_call="auto",
                max_tokens=200
            )
            response_time = time.time() - start_time
            
            # Анализируем ответ на наличие function calling
            response_str = str(response).lower()
            
            # Проверяем индикаторы function calling
            has_function_structure = any(indicator in str(response) for indicator in [
                '"function_call"', '"name": "get_weather"', '"arguments"'
            ])
            
            has_weather_result = any(indicator in response_str for indicator in [
                'солнечно', '+20°c', '20°c', 'погода в москве'
            ])
            
            is_regular_text = any(indicator in response_str for indicator in [
                'к сожалению', 'не могу использовать', 'не имею доступа'
            ])
            
            # Оценка успешности
            success = (has_function_structure or has_weather_result) and not is_regular_text
            confidence = 0.0
            
            if has_function_structure:
                confidence += 0.6
            if has_weather_result:
                confidence += 0.4
            if is_regular_text:
                confidence = max(0.0, confidence - 0.5)
            
            return TestResult(
                capability=capability,
                client_type=ClientType(type(client).__name__),
                model_name=model_config.name,
                success=success,
                response_time=response_time,
                confidence_score=min(1.0, confidence),
                response_preview=f"FC: {'✓' if success else '✗'} - {str(response)[:60]}",
                metadata={
                    "has_function_structure": has_function_structure,
                    "has_result": has_weather_result,
                    "is_regular_text": is_regular_text
                }
            )
            
        except Exception as e:
            return TestResult(
                capability=capability,
                client_type=ClientType(type(client).__name__),
                model_name=model_config.name,
                success=False,
                response_time=time.time() - start_time,
                confidence_score=0.0,
                error_message=str(e)
            )
    
    async def _test_tool_calling(self, client, model_config: ModelConfig, capability: CapabilityType) -> TestResult:
        """Тест tool calling"""
        start_time = time.time()
        
        # Регистрируем тестовый инструмент
        @register_tool(description="Вычислить сумму двух чисел")
        def calculate_sum(a: int, b: int) -> int:
            return a + b
        
        messages = [
            {"role": "user", "content": "Вычисли сумму 15 и 27 используя инструмент calculate_sum."}
        ]
        
        tools = [{
            "type": "function",
            "function": {
                "name": "calculate_sum",
                "description": "Вычисляет сумму двух чисел",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer", "description": "Первое число"},
                        "b": {"type": "integer", "description": "Второе число"}
                    },
                    "required": ["a", "b"]
                }
            }
        }]
        
        try:
            if hasattr(client, 'register_tool'):
                client.register_tool("calculate_sum", calculate_sum, "Вычислить сумму")
            
            response = await client.chat_completion(
                messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=200
            )
            response_time = time.time() - start_time
            
            # Анализируем ответ на наличие tool calling
            response_str = str(response).lower()
            
            has_tool_structure = any(indicator in str(response) for indicator in [
                '"tool_calls"', '"type": "function"', '"name": "calculate_sum"'
            ])
            
            has_correct_result = '42' in response_str  # 15 + 27 = 42
            
            is_regular_text = any(indicator in response_str for indicator in [
                'к сожалению', 'не могу использовать', 'инструменты недоступны'
            ])
            
            success = (has_tool_structure or has_correct_result) and not is_regular_text
            confidence = 0.0
            
            if has_tool_structure:
                confidence += 0.6
            if has_correct_result:
                confidence += 0.4
            if is_regular_text:
                confidence = max(0.0, confidence - 0.5)
            
            return TestResult(
                capability=capability,
                client_type=ClientType(type(client).__name__),
                model_name=model_config.name,
                success=success,
                response_time=response_time,
                confidence_score=min(1.0, confidence),
                response_preview=f"TC: {'✓' if success else '✗'} - {str(response)[:60]}",
                metadata={
                    "has_tool_structure": has_tool_structure,
                    "has_correct_result": has_correct_result,
                    "is_regular_text": is_regular_text
                }
            )
            
        except Exception as e:
            return TestResult(
                capability=capability,
                client_type=ClientType(type(client).__name__),
                model_name=model_config.name,
                success=False,
                response_time=time.time() - start_time,
                confidence_score=0.0,
                error_message=str(e)
            )  
  
    async def _test_embeddings(self, client, model_config: ModelConfig, capability: CapabilityType) -> TestResult:
        """Тест создания embeddings"""
        start_time = time.time()
        
        texts = ["Тестовый текст для векторизации", "Второй тестовый текст"]
        
        try:
            if hasattr(client, 'get_embeddings'):
                response = await client.get_embeddings(texts)
            elif hasattr(client, 'create_embeddings'):
                response = await client.create_embeddings(texts)
            else:
                raise Exception("Embeddings не поддерживаются этим клиентом")
            
            response_time = time.time() - start_time
            
            # Проверяем качество embeddings
            confidence = 1.0
            embedding_info = ""
            
            if hasattr(response, 'data') and response.data:
                embeddings = response.data
                if len(embeddings) == len(texts):
                    first_embedding = embeddings[0].embedding if hasattr(embeddings[0], 'embedding') else embeddings[0]
                    if isinstance(first_embedding, list) and len(first_embedding) > 0:
                        embedding_info = f"{len(embeddings)} embeddings, dim={len(first_embedding)}"
                    else:
                        confidence = 0.5
                else:
                    confidence = 0.7
            elif isinstance(response, list):
                if len(response) == len(texts) and all(isinstance(emb, list) for emb in response):
                    embedding_info = f"{len(response)} embeddings, dim={len(response[0])}"
                else:
                    confidence = 0.5
            else:
                confidence = 0.3
                embedding_info = "Unknown format"
            
            return TestResult(
                capability=capability,
                client_type=ClientType(type(client).__name__),
                model_name=model_config.name,
                success=True,
                response_time=response_time,
                confidence_score=confidence,
                response_preview=f"Embeddings: {embedding_info}",
                performance_metrics={"embeddings_per_second": len(texts) / response_time if response_time > 0 else 0}
            )
            
        except Exception as e:
            return TestResult(
                capability=capability,
                client_type=ClientType(type(client).__name__),
                model_name=model_config.name,
                success=False,
                response_time=time.time() - start_time,
                confidence_score=0.0,
                error_message=str(e)
            )
    
    async def _test_similarity_search(self, client, model_config: ModelConfig, capability: CapabilityType) -> TestResult:
        """Тест поиска по сходству"""
        start_time = time.time()
        
        query = "программирование на Python"
        candidates = [
            "Изучение языка Python",
            "Готовка борща",
            "Разработка на JavaScript",
            "Python для начинающих"
        ]
        
        try:
            if hasattr(client, 'similarity_search'):
                response = await client.similarity_search(
                    query_text=query,
                    candidate_texts=candidates,
                    top_k=2
                )
                response_time = time.time() - start_time
                
                # Проверяем качество поиска
                confidence = 1.0
                if isinstance(response, list) and len(response) > 0:
                    # Ожидаем, что Python-связанные тексты будут в топе
                    top_result = response[0]
                    if "python" in str(top_result).lower():
                        confidence = 1.0
                    else:
                        confidence = 0.5
                else:
                    confidence = 0.3
                
                return TestResult(
                    capability=capability,
                    client_type=ClientType(type(client).__name__),
                    model_name=model_config.name,
                    success=True,
                    response_time=response_time,
                    confidence_score=confidence,
                    response_preview=f"Similarity search: {len(response) if isinstance(response, list) else 1} results"
                )
            else:
                return TestResult(
                    capability=capability,
                    client_type=ClientType(type(client).__name__),
                    model_name=model_config.name,
                    success=False,
                    response_time=time.time() - start_time,
                    confidence_score=0.0,
                    error_message="Similarity search не поддерживается"
                )
                
        except Exception as e:
            return TestResult(
                capability=capability,
                client_type=ClientType(type(client).__name__),
                model_name=model_config.name,
                success=False,
                response_time=time.time() - start_time,
                confidence_score=0.0,
                error_message=str(e)
            )
    
    async def _test_reasoning_cot(self, client, model_config: ModelConfig, capability: CapabilityType) -> TestResult:
        """Тест Chain of Thought reasoning"""
        start_time = time.time()
        
        messages = [
            {"role": "system", "content": "Решай задачи пошагово, объясняя каждый шаг своих рассуждений."},
            {"role": "user", "content": "Реши задачу: В магазине было 50 яблок. Утром продали 15, днем еще 12. Сколько яблок осталось?"}
        ]
        
        try:
            if hasattr(client, 'reasoning_completion'):
                response = await client.reasoning_completion(
                    messages,
                    problem_type="math",
                    max_tokens=300
                )
            else:
                response = await client.chat_completion(messages, max_tokens=300)
            
            response_time = time.time() - start_time
            
            # Анализируем качество рассуждений
            response_str = str(response).lower()
            
            has_steps = any(word in response_str for word in [
                'шаг', 'сначала', 'затем', 'итак', 'следовательно', 'первый', 'второй'
            ])
            
            has_calculation = any(char in response_str for char in ['=', '+', '-', '*', '/'])
            
            has_correct_answer = '23' in response_str  # 50 - 15 - 12 = 23
            
            # Оценка качества
            confidence = 0.0
            if has_steps:
                confidence += 0.4
            if has_calculation:
                confidence += 0.3
            if has_correct_answer:
                confidence += 0.3
            
            success = confidence >= 0.5
            
            return TestResult(
                capability=capability,
                client_type=ClientType(type(client).__name__),
                model_name=model_config.name,
                success=success,
                response_time=response_time,
                confidence_score=confidence,
                response_preview=f"CoT: {'✓' if success else '✗'} steps={has_steps}, calc={has_calculation}, answer={has_correct_answer}",
                metadata={
                    "has_steps": has_steps,
                    "has_calculation": has_calculation,
                    "has_correct_answer": has_correct_answer
                }
            )
            
        except Exception as e:
            return TestResult(
                capability=capability,
                client_type=ClientType(type(client).__name__),
                model_name=model_config.name,
                success=False,
                response_time=time.time() - start_time,
                confidence_score=0.0,
                error_message=str(e)
            )
    
    async def _test_reasoning_native_thinking(self, client, model_config: ModelConfig, capability: CapabilityType) -> TestResult:
        """Тест нативного thinking режима"""
        start_time = time.time()
        
        messages = [
            {"role": "user", "content": "Реши задачу пошагово: если у меня есть 10 яблок и я съел 3, сколько осталось?"}
        ]
        
        try:
            if hasattr(client, 'reasoning_completion'):
                response = await client.reasoning_completion(
                    messages,
                    problem_type="math",
                    max_tokens=300,
                    enable_streaming=False
                )
                
                # Проверяем наличие thinking структур
                has_thinking_blocks = hasattr(response, 'thinking_blocks') and response.thinking_blocks
                has_steps = hasattr(response, 'steps') and response.steps
                
                response_str = str(response)
                has_thinking_markers = '<thinking>' in response_str or 'думаю' in response_str.lower()
                has_correct_answer = '7' in response_str
                
                confidence = 0.0
                if has_thinking_blocks:
                    confidence += 0.5
                if has_steps:
                    confidence += 0.3
                if has_thinking_markers:
                    confidence += 0.2
                if has_correct_answer:
                    confidence += 0.2
                
                success = confidence >= 0.4
                
                return TestResult(
                    capability=capability,
                    client_type=ClientType(type(client).__name__),
                    model_name=model_config.name,
                    success=success,
                    response_time=time.time() - start_time,
                    confidence_score=min(1.0, confidence),
                    response_preview=f"Native thinking: {'✓' if success else '✗'} blocks={has_thinking_blocks}, steps={has_steps}",
                    metadata={
                        "has_thinking_blocks": has_thinking_blocks,
                        "has_steps": has_steps,
                        "has_thinking_markers": has_thinking_markers,
                        "has_correct_answer": has_correct_answer
                    }
                )
            else:
                return TestResult(
                    capability=capability,
                    client_type=ClientType(type(client).__name__),
                    model_name=model_config.name,
                    success=False,
                    response_time=time.time() - start_time,
                    confidence_score=0.0,
                    error_message="Native thinking не поддерживается"
                )
                
        except Exception as e:
            return TestResult(
                capability=capability,
                client_type=ClientType(type(client).__name__),
                model_name=model_config.name,
                success=False,
                response_time=time.time() - start_time,
                confidence_score=0.0,
                error_message=str(e)
            ) 
   
    # Дополнительные тесты для специализированных клиентов
    
    async def _test_multimodal_vision(self, client, model_config: ModelConfig, capability: CapabilityType) -> TestResult:
        """Тест мультимодальных возможностей (vision)"""
        start_time = time.time()
        
        try:
            # Создаем простое тестовое изображение
            test_image_path = Path("test_image.png")
            if not test_image_path.exists():
                # Создаем простое изображение для тестирования
                from PIL import Image, ImageDraw
                img = Image.new('RGB', (100, 100), color='blue')
                draw = ImageDraw.Draw(img)
                draw.text((10, 40), "TEST", fill='white')
                img.save(test_image_path)
            
            if hasattr(client, 'vision_completion'):
                response = await client.vision_completion(
                    text_prompt="Что изображено на картинке?",
                    images=str(test_image_path),
                    detail_level="low"
                )
                response_time = time.time() - start_time
                
                # Очищаем тестовое изображение
                if test_image_path.exists():
                    test_image_path.unlink()
                
                confidence = 0.8 if response else 0.5
                
                return TestResult(
                    capability=capability,
                    client_type=ClientType(type(client).__name__),
                    model_name=model_config.name,
                    success=True,
                    response_time=response_time,
                    confidence_score=confidence,
                    response_preview=f"Vision: {str(response)[:60] if response else 'No response'}"
                )
            else:
                return TestResult(
                    capability=capability,
                    client_type=ClientType(type(client).__name__),
                    model_name=model_config.name,
                    success=False,
                    response_time=time.time() - start_time,
                    confidence_score=0.0,
                    error_message="Vision capabilities не поддерживаются"
                )
                
        except Exception as e:
            # Очищаем тестовое изображение в случае ошибки
            test_image_path = Path("test_image.png")
            if test_image_path.exists():
                test_image_path.unlink()
            
            return TestResult(
                capability=capability,
                client_type=ClientType(type(client).__name__),
                model_name=model_config.name,
                success=False,
                response_time=time.time() - start_time,
                confidence_score=0.0,
                error_message=str(e)
            )
    
    async def _test_asr_stt(self, client, model_config: ModelConfig, capability: CapabilityType) -> TestResult:
        """Тест Speech-to-Text"""
        start_time = time.time()
        
        try:
            if hasattr(client, 'speech_to_text'):
                # Для тестирования используем заглушку
                # В реальном использовании здесь был бы аудио файл
                return TestResult(
                    capability=capability,
                    client_type=ClientType(type(client).__name__),
                    model_name=model_config.name,
                    success=False,  # Не можем протестировать без аудио файла
                    response_time=time.time() - start_time,
                    confidence_score=0.0,
                    error_message="Требуется аудио файл для тестирования STT"
                )
            else:
                return TestResult(
                    capability=capability,
                    client_type=ClientType(type(client).__name__),
                    model_name=model_config.name,
                    success=False,
                    response_time=time.time() - start_time,
                    confidence_score=0.0,
                    error_message="STT не поддерживается"
                )
                
        except Exception as e:
            return TestResult(
                capability=capability,
                client_type=ClientType(type(client).__name__),
                model_name=model_config.name,
                success=False,
                response_time=time.time() - start_time,
                confidence_score=0.0,
                error_message=str(e)
            )
    
    async def _test_asr_tts(self, client, model_config: ModelConfig, capability: CapabilityType) -> TestResult:
        """Тест Text-to-Speech"""
        start_time = time.time()
        
        try:
            if hasattr(client, 'text_to_speech'):
                response = await client.text_to_speech(
                    text="Тестовое сообщение для синтеза речи",
                    voice="default"
                )
                response_time = time.time() - start_time
                
                confidence = 0.8 if response else 0.3
                
                return TestResult(
                    capability=capability,
                    client_type=ClientType(type(client).__name__),
                    model_name=model_config.name,
                    success=bool(response),
                    response_time=response_time,
                    confidence_score=confidence,
                    response_preview=f"TTS: {'Success' if response else 'Failed'}"
                )
            else:
                return TestResult(
                    capability=capability,
                    client_type=ClientType(type(client).__name__),
                    model_name=model_config.name,
                    success=False,
                    response_time=time.time() - start_time,
                    confidence_score=0.0,
                    error_message="TTS не поддерживается"
                )
                
        except Exception as e:
            return TestResult(
                capability=capability,
                client_type=ClientType(type(client).__name__),
                model_name=model_config.name,
                success=False,
                response_time=time.time() - start_time,
                confidence_score=0.0,
                error_message=str(e)
            )
    
    async def _test_completion_legacy(self, client, model_config: ModelConfig, capability: CapabilityType) -> TestResult:
        """Тест legacy completion API"""
        start_time = time.time()
        
        try:
            if hasattr(client, 'text_completion'):
                response = await client.text_completion(
                    prompt="Завершите предложение: Искусственный интеллект это",
                    max_tokens=50
                )
                response_time = time.time() - start_time
                
                confidence = 0.8 if response and len(str(response)) > 10 else 0.3
                
                return TestResult(
                    capability=capability,
                    client_type=ClientType(type(client).__name__),
                    model_name=model_config.name,
                    success=bool(response),
                    response_time=response_time,
                    confidence_score=confidence,
                    response_preview=f"Completion: {str(response)[:60] if response else 'No response'}"
                )
            else:
                return TestResult(
                    capability=capability,
                    client_type=ClientType(type(client).__name__),
                    model_name=model_config.name,
                    success=False,
                    response_time=time.time() - start_time,
                    confidence_score=0.0,
                    error_message="Legacy completion не поддерживается"
                )
                
        except Exception as e:
            return TestResult(
                capability=capability,
                client_type=ClientType(type(client).__name__),
                model_name=model_config.name,
                success=False,
                response_time=time.time() - start_time,
                confidence_score=0.0,
                error_message=str(e)
            )
    
    async def _test_adaptive_client(self, model_config: ModelConfig) -> List[TestResult]:
        """Специальный тест для AdaptiveLLMClient"""
        results = []
        
        config = LLMConfig(
            endpoint=model_config.endpoint,
            api_key=model_config.api_key,
            model=model_config.model,
            temperature=0.7
        )
        
        adaptive_config = AdaptiveConfig(
            capability_detection_timeout=15.0,
            enable_performance_tracking=True
        )
        
        try:
            async with AdaptiveLLMClient(config, adaptive_config) as client:
                # Тест определения возможностей
                start_time = time.time()
                capabilities = await client.get_model_capabilities(force_refresh=True)
                detection_time = time.time() - start_time
                
                results.append(TestResult(
                    capability=CapabilityType.CAPABILITY_DETECTION,
                    client_type=ClientType.ADAPTIVE,
                    model_name=model_config.name,
                    success=len(capabilities) > 0,
                    response_time=detection_time,
                    confidence_score=min(1.0, len(capabilities) / 5.0),  # Ожидаем минимум 5 возможностей
                    response_preview=f"Detected {len(capabilities)} capabilities",
                    metadata={"detected_capabilities": [cap.value for cap in capabilities]}
                ))
                
                # Тест smart completion
                try:
                    messages = [{"role": "user", "content": "Привет! Как дела?"}]
                    start_time = time.time()
                    response = await client.smart_completion(messages, max_tokens=50)
                    response_time = time.time() - start_time
                    
                    results.append(TestResult(
                        capability=CapabilityType.SMART_COMPLETION,
                        client_type=ClientType.ADAPTIVE,
                        model_name=model_config.name,
                        success=bool(response),
                        response_time=response_time,
                        confidence_score=0.9 if response else 0.1,
                        response_preview=f"Smart completion: {str(response)[:50] if response else 'Failed'}"
                    ))
                except Exception as e:
                    results.append(TestResult(
                        capability=CapabilityType.SMART_COMPLETION,
                        client_type=ClientType.ADAPTIVE,
                        model_name=model_config.name,
                        success=False,
                        response_time=0.0,
                        confidence_score=0.0,
                        error_message=str(e)
                    ))
                
                # Тест адаптивного режима
                results.append(TestResult(
                    capability=CapabilityType.ADAPTIVE_MODE,
                    client_type=ClientType.ADAPTIVE,
                    model_name=model_config.name,
                    success=True,
                    response_time=detection_time,
                    confidence_score=1.0,
                    response_preview="Adaptive mode functional",
                    metadata={"total_capabilities": len(capabilities)}
                ))
        
        except Exception as e:
            results.append(TestResult(
                capability=CapabilityType.ADAPTIVE_MODE,
                client_type=ClientType.ADAPTIVE,
                model_name=model_config.name,
                success=False,
                response_time=0.0,
                confidence_score=0.0,
                error_message=str(e)
            ))
        
        return results    

    def _generate_analysis_report(self) -> Dict[str, Any]:
        """Генерация итогового отчета анализа"""
        
        # Группируем результаты по моделям
        by_model = {}
        for result in self.results:
            if result.model_name not in by_model:
                by_model[result.model_name] = []
            by_model[result.model_name].append(result)
        
        # Генерируем сводку по каждой модели
        model_summaries = {}
        for model_name, model_results in by_model.items():
            model_summaries[model_name] = self._generate_model_summary(model_name, model_results)
        
        # Общие рекомендации
        general_recommendations = self._generate_general_recommendations()
        
        # Итоговый отчет
        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "execution_time": self.stats["execution_time"],
                "kraken_version": "1.0.0",
                "analyzer_version": "1.0.0",
                "total_models": self.stats["models_tested"],
                "total_tests": self.stats["total_tests"],
                "successful_tests": self.stats["successful_tests"],
                "success_rate": (self.stats["successful_tests"] / self.stats["total_tests"] * 100) if self.stats["total_tests"] > 0 else 0
            },
            "model_configs": [asdict(config) for config in self.model_configs],
            "test_results": [asdict(result) for result in self.results],
            "model_summaries": model_summaries,
            "general_recommendations": general_recommendations,
            "statistics": self.stats
        }
        
        return report
    
    def _generate_model_summary(self, model_name: str, results: List[TestResult]) -> Dict[str, Any]:
        """Генерация сводки для одной модели"""
        
        # Статистика по возможностям
        capabilities_stats = {}
        for result in results:
            cap_key = result.capability.value
            if cap_key not in capabilities_stats:
                capabilities_stats[cap_key] = {
                    "success_count": 0,
                    "total_count": 0,
                    "avg_response_time": 0.0,
                    "avg_confidence": 0.0,
                    "errors": []
                }
            
            stats = capabilities_stats[cap_key]
            stats["total_count"] += 1
            stats["avg_response_time"] += result.response_time
            stats["avg_confidence"] += result.confidence_score
            
            if result.success:
                stats["success_count"] += 1
            else:
                if result.error_message:
                    stats["errors"].append(result.error_message)
        
        # Вычисляем средние значения
        for stats in capabilities_stats.values():
            if stats["total_count"] > 0:
                stats["avg_response_time"] /= stats["total_count"]
                stats["avg_confidence"] /= stats["total_count"]
                stats["success_rate"] = stats["success_count"] / stats["total_count"]
        
        # Статистика по клиентам
        clients_stats = {}
        for result in results:
            client_key = result.client_type.value
            if client_key not in clients_stats:
                clients_stats[client_key] = {"success": 0, "total": 0, "avg_confidence": 0.0}
            
            clients_stats[client_key]["total"] += 1
            clients_stats[client_key]["avg_confidence"] += result.confidence_score
            if result.success:
                clients_stats[client_key]["success"] += 1
        
        # Вычисляем средние значения для клиентов
        for stats in clients_stats.values():
            if stats["total"] > 0:
                stats["success_rate"] = stats["success"] / stats["total"]
                stats["avg_confidence"] /= stats["total"]
        
        # Подтвержденные возможности (успешность > 50% и уверенность > 0.5)
        confirmed_capabilities = []
        for cap_name, stats in capabilities_stats.items():
            if stats["success_rate"] > 0.5 and stats["avg_confidence"] > 0.5:
                confirmed_capabilities.append({
                    "capability": cap_name,
                    "success_rate": stats["success_rate"],
                    "confidence": stats["avg_confidence"],
                    "avg_response_time": stats["avg_response_time"]
                })
        
        # Рекомендуемые клиенты (успешность > 70%)
        recommended_clients = []
        for client_name, stats in clients_stats.items():
            if stats["success_rate"] > 0.7:
                recommended_clients.append({
                    "client": client_name,
                    "success_rate": stats["success_rate"],
                    "confidence": stats["avg_confidence"]
                })
        
        # Генерируем рекомендации для модели
        model_recommendations = self._generate_model_recommendations(model_name, results, confirmed_capabilities, recommended_clients)
        
        return {
            "model_name": model_name,
            "total_tests": len(results),
            "successful_tests": sum(1 for r in results if r.success),
            "success_rate": sum(1 for r in results if r.success) / len(results) if results else 0,
            "avg_confidence": sum(r.confidence_score for r in results) / len(results) if results else 0,
            "confirmed_capabilities": confirmed_capabilities,
            "recommended_clients": recommended_clients,
            "capabilities_stats": capabilities_stats,
            "clients_stats": clients_stats,
            "recommendations": model_recommendations
        }
    
    def _generate_model_recommendations(self, model_name: str, results: List[TestResult], 
                                      confirmed_capabilities: List[Dict], recommended_clients: List[Dict]) -> List[str]:
        """Генерация рекомендаций для конкретной модели"""
        recommendations = []
        
        # Найдем конфигурацию модели
        model_config = next((config for config in self.model_configs if config.name == model_name), None)
        
        if not model_config:
            return ["Конфигурация модели не найдена"]
        
        # Общая оценка модели
        success_rate = sum(1 for r in results if r.success) / len(results) if results else 0
        
        if success_rate > 0.8:
            recommendations.append("🎉 Отличная совместимость! Модель поддерживает большинство функций Kraken LLM")
        elif success_rate > 0.6:
            recommendations.append("✅ Хорошая совместимость. Модель подходит для большинства задач")
        elif success_rate > 0.3:
            recommendations.append("⚠️ Частичная совместимость. Модель работает с ограниченным набором функций")
        else:
            recommendations.append("❌ Низкая совместимость. Требуется дополнительная настройка")
        
        # Рекомендации по клиентам
        if recommended_clients:
            client_names = [client["client"] for client in recommended_clients]
            recommendations.append(f"🔧 Рекомендуемые клиенты: {', '.join(client_names)}")
        
        # Рекомендации по возможностям
        if confirmed_capabilities:
            cap_names = [cap["capability"] for cap in confirmed_capabilities[:5]]  # Топ 5
            recommendations.append(f"⚡ Поддерживаемые возможности: {', '.join(cap_names)}")
        
        # Специфичные рекомендации по типу модели
        if model_config.model_type == ModelType.EMBEDDING:
            if any(cap["capability"] == "embeddings" for cap in confirmed_capabilities):
                recommendations.append("📊 Используйте EmbeddingsLLMClient для векторизации текста")
            else:
                recommendations.append("⚠️ Проблемы с embeddings API. Проверьте endpoint и модель")
        
        elif model_config.model_type == ModelType.CHAT:
            if any(cap["capability"] == "structured_output_native" for cap in confirmed_capabilities):
                recommendations.append("📋 Поддерживает structured output - используйте StructuredLLMClient")
            
            if any(cap["capability"] == "streaming" for cap in confirmed_capabilities):
                recommendations.append("🌊 Поддерживает streaming - используйте StreamingLLMClient")
            
            if any(cap["capability"] == "function_calling" for cap in confirmed_capabilities):
                recommendations.append("🔧 Поддерживает function calling - регистрируйте функции")
        
        # Рекомендации по производительности
        avg_response_time = sum(r.response_time for r in results if r.success) / max(1, sum(1 for r in results if r.success))
        
        if avg_response_time < 1.0:
            recommendations.append("⚡ Быстрая модель - подходит для real-time приложений")
        elif avg_response_time > 5.0:
            recommendations.append("🐌 Медленная модель - рассмотрите использование streaming режима")
        
        # Рекомендации по провайдеру
        if model_config.provider == "Local":
            recommendations.append("🏠 Локальная модель - отличная приватность, но может быть медленнее")
        elif model_config.provider == "OpenAI":
            recommendations.append("🚀 OpenAI модель - высокое качество и скорость")
        
        return recommendations
    
    def _generate_general_recommendations(self) -> List[str]:
        """Генерация общих рекомендаций"""
        recommendations = []
        
        # Анализ общей статистики
        if self.stats["total_tests"] == 0:
            return ["❌ Тесты не выполнялись. Проверьте конфигурацию моделей"]
        
        success_rate = self.stats["successful_tests"] / self.stats["total_tests"]
        
        if success_rate > 0.8:
            recommendations.append("🎯 Отличная общая совместимость! Kraken LLM хорошо работает с вашими моделями")
        elif success_rate > 0.6:
            recommendations.append("✅ Хорошая совместимость. Большинство функций работает корректно")
        else:
            recommendations.append("⚠️ Требуется настройка. Многие функции работают нестабильно")
        
        # Рекомендации по использованию
        recommendations.extend([
            "📚 Изучите документацию: README.md для подробного описания возможностей",
            "🔧 Используйте AdaptiveLLMClient для автоматического выбора оптимального режима",
            "⚡ Для production используйте клиенты с высокой успешностью (>70%)",
            "🧪 Тестируйте новые возможности в development окружении",
        ])
        
        # Рекомендации по конфигурации
        if len(self.model_configs) == 1:
            recommendations.append("💡 Рассмотрите добавление дополнительных моделей для разных задач")
        
        # Рекомендации по мониторингу
        recommendations.extend([
            "📊 Регулярно запускайте анализ для мониторинга изменений в API",
            "🔄 Используйте CI/CD интеграцию для автоматического тестирования",
        ])
        
        return recommendations   
 
    def save_report(self, report: Dict[str, Any], output_format: str = "json", filename: Optional[str] = None) -> str:
        """Сохранение отчета в файл"""
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if output_format == "json":
                filename = f"model_capabilities_report_{timestamp}.json"
            elif output_format == "markdown":
                filename = f"model_capabilities_report_{timestamp}.md"
            else:
                filename = f"model_capabilities_report_{timestamp}.txt"
        
        if output_format == "json":
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        elif output_format == "markdown":
            markdown_content = self._generate_markdown_report(report)
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
        
        else:  # text format
            text_content = self._generate_text_report(report)
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(text_content)
        
        logger.info(f"Отчет сохранен: {filename}")
        return filename
    
    def _generate_markdown_report(self, report: Dict[str, Any]) -> str:
        """Генерация отчета в Markdown формате"""
        
        md = []
        md.append("# Kraken LLM Model Capabilities Analysis Report")
        md.append("")
        md.append(f"**Generated:** {report['metadata']['timestamp']}")
        md.append(f"**Execution Time:** {report['metadata']['execution_time']:.2f}s")
        md.append(f"**Success Rate:** {report['metadata']['success_rate']:.1f}%")
        md.append("")
        
        # Общая статистика
        md.append("## 📊 Overall Statistics")
        md.append("")
        md.append(f"- **Models Tested:** {report['metadata']['total_models']}")
        md.append(f"- **Total Tests:** {report['metadata']['total_tests']}")
        md.append(f"- **Successful Tests:** {report['metadata']['successful_tests']}")
        md.append(f"- **Failed Tests:** {report['metadata']['total_tests'] - report['metadata']['successful_tests']}")
        md.append("")
        
        # Результаты по моделям
        md.append("## 🤖 Model Analysis Results")
        md.append("")
        
        for model_name, summary in report['model_summaries'].items():
            md.append(f"### {model_name}")
            md.append("")
            
            # Статус модели
            success_rate = summary['success_rate'] * 100
            if success_rate > 80:
                status = "🟢 Excellent"
            elif success_rate > 60:
                status = "🟡 Good"
            elif success_rate > 30:
                status = "🟠 Partial"
            else:
                status = "🔴 Poor"
            
            md.append(f"**Status:** {status} ({success_rate:.1f}% success rate)")
            md.append(f"**Confidence:** {summary['avg_confidence']:.2f}")
            md.append("")
            
            # Подтвержденные возможности
            if summary['confirmed_capabilities']:
                md.append("**✅ Confirmed Capabilities:**")
                for cap in summary['confirmed_capabilities']:
                    md.append(f"- {cap['capability']} ({cap['success_rate']:.0%} success, {cap['avg_response_time']:.2f}s avg)")
                md.append("")
            
            # Рекомендуемые клиенты
            if summary['recommended_clients']:
                md.append("**🔧 Recommended Clients:**")
                for client in summary['recommended_clients']:
                    md.append(f"- {client['client']} ({client['success_rate']:.0%} success)")
                md.append("")
            
            # Рекомендации
            if summary['recommendations']:
                md.append("**💡 Recommendations:**")
                for rec in summary['recommendations']:
                    md.append(f"- {rec}")
                md.append("")
        
        # Общие рекомендации
        md.append("## 🎯 General Recommendations")
        md.append("")
        for rec in report['general_recommendations']:
            md.append(f"- {rec}")
        md.append("")
        
        # Примеры использования
        md.append("## 📝 Usage Examples")
        md.append("")
        md.append("### Basic Usage")
        md.append("```python")
        md.append("from kraken_llm import create_client")
        md.append("")
        md.append("async with create_client() as client:")
        md.append('    response = await client.chat_completion([')
        md.append('        {"role": "user", "content": "Hello!"}')
        md.append('    ])')
        md.append("```")
        md.append("")
        
        # Найдем лучший клиент для примера
        best_client = None
        best_success_rate = 0
        
        for summary in report['model_summaries'].values():
            for client in summary['recommended_clients']:
                if client['success_rate'] > best_success_rate:
                    best_success_rate = client['success_rate']
                    best_client = client['client']
        
        if best_client:
            md.append(f"### Recommended Client ({best_client})")
            md.append("```python")
            
            if best_client == "StructuredLLMClient":
                md.append("from kraken_llm import create_structured_client")
                md.append("from pydantic import BaseModel")
                md.append("")
                md.append("class Response(BaseModel):")
                md.append("    answer: str")
                md.append("")
                md.append("async with create_structured_client() as client:")
                md.append("    result = await client.chat_completion_structured([")
                md.append('        {"role": "user", "content": "Answer structured"}')
                md.append("    ], response_model=Response)")
            
            elif best_client == "StreamingLLMClient":
                md.append("from kraken_llm import create_streaming_client")
                md.append("")
                md.append("async with create_streaming_client() as client:")
                md.append("    async for chunk in client.chat_completion_stream([")
                md.append('        {"role": "user", "content": "Tell a story"}')
                md.append("    ]):")
                md.append('        print(chunk, end="", flush=True)')
            
            else:
                md.append(f"from kraken_llm import create_client")
                md.append("")
                md.append(f'client = create_client(client_type="{best_client.lower().replace("llmclient", "")}")')
            
            md.append("```")
            md.append("")
        
        return "\n".join(md)
    
    def _generate_text_report(self, report: Dict[str, Any]) -> str:
        """Генерация отчета в текстовом формате"""
        
        lines = []
        lines.append("=" * 80)
        lines.append("KRAKEN LLM MODEL CAPABILITIES ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Generated: {report['metadata']['timestamp']}")
        lines.append(f"Execution Time: {report['metadata']['execution_time']:.2f}s")
        lines.append(f"Success Rate: {report['metadata']['success_rate']:.1f}%")
        lines.append("")
        
        # Общая статистика
        lines.append("OVERALL STATISTICS")
        lines.append("-" * 40)
        lines.append(f"Models Tested: {report['metadata']['total_models']}")
        lines.append(f"Total Tests: {report['metadata']['total_tests']}")
        lines.append(f"Successful Tests: {report['metadata']['successful_tests']}")
        lines.append(f"Failed Tests: {report['metadata']['total_tests'] - report['metadata']['successful_tests']}")
        lines.append("")
        
        # Результаты по моделям
        lines.append("MODEL ANALYSIS RESULTS")
        lines.append("-" * 40)
        
        for model_name, summary in report['model_summaries'].items():
            lines.append("")
            lines.append(f"Model: {model_name}")
            lines.append(f"Success Rate: {summary['success_rate'] * 100:.1f}%")
            lines.append(f"Confidence: {summary['avg_confidence']:.2f}")
            
            if summary['confirmed_capabilities']:
                lines.append("Confirmed Capabilities:")
                for cap in summary['confirmed_capabilities']:
                    lines.append(f"  - {cap['capability']} ({cap['success_rate']:.0%})")
            
            if summary['recommended_clients']:
                lines.append("Recommended Clients:")
                for client in summary['recommended_clients']:
                    lines.append(f"  - {client['client']} ({client['success_rate']:.0%})")
            
            if summary['recommendations']:
                lines.append("Recommendations:")
                for rec in summary['recommendations']:
                    lines.append(f"  - {rec}")
        
        lines.append("")
        lines.append("GENERAL RECOMMENDATIONS")
        lines.append("-" * 40)
        for rec in report['general_recommendations']:
            lines.append(f"- {rec}")
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def print_summary(self, report: Dict[str, Any]) -> None:
        """Печать краткой сводки в консоль"""
        
        print("\n" + "=" * 80)
        print("🚀 KRAKEN LLM MODEL CAPABILITIES ANALYSIS")
        print("=" * 80)
        
        # Общая статистика
        metadata = report['metadata']
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Models Tested: {metadata['total_models']}")
        print(f"   Total Tests: {metadata['total_tests']}")
        print(f"   Success Rate: {metadata['success_rate']:.1f}%")
        print(f"   Execution Time: {metadata['execution_time']:.2f}s")
        
        # Результаты по моделям
        print(f"\n🤖 MODEL RESULTS:")
        for model_name, summary in report['model_summaries'].items():
            success_rate = summary['success_rate'] * 100
            
            if success_rate > 80:
                status = "🟢"
            elif success_rate > 60:
                status = "🟡"
            elif success_rate > 30:
                status = "🟠"
            else:
                status = "🔴"
            
            print(f"   {status} {model_name}: {success_rate:.1f}% success")
            
            # Топ возможности
            if summary['confirmed_capabilities']:
                top_caps = [cap['capability'] for cap in summary['confirmed_capabilities'][:3]]
                print(f"      ✅ Top capabilities: {', '.join(top_caps)}")
            
            # Лучший клиент
            if summary['recommended_clients']:
                best_client = summary['recommended_clients'][0]['client']
                print(f"      🔧 Best client: {best_client}")
        
        # Топ рекомендации
        print(f"\n💡 TOP RECOMMENDATIONS:")
        for rec in report['general_recommendations'][:5]:
            print(f"   • {rec}")
        
        print("\n" + "=" * 80)


async def main():
    """Главная функция анализатора"""
    
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(
        description="Kraken LLM Model Capabilities Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python model_capabilities_analyzer.py                    # Полный анализ
  python model_capabilities_analyzer.py --quick            # Быстрый анализ
  python model_capabilities_analyzer.py --output markdown  # Markdown отчет
  python model_capabilities_analyzer.py --config custom.env # Кастомная конфигурация
        """
    )
    
    parser.add_argument(
        "--quick", 
        action="store_true", 
        help="Быстрый режим (только основные тесты)"
    )
    
    parser.add_argument(
        "--output", 
        choices=["json", "markdown", "text"], 
        default="json",
        help="Формат выходного файла (по умолчанию: json)"
    )
    
    parser.add_argument(
        "--config", 
        type=str,
        help="Путь к файлу конфигурации (.env)"
    )
    
    parser.add_argument(
        "--filename", 
        type=str,
        help="Имя выходного файла"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Подробный вывод"
    )
    
    args = parser.parse_args()
    
    # Настройка логирования
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Загрузка дополнительной конфигурации
    if args.config:
        load_dotenv(args.config)
    
    print("🚀 Kraken LLM Model Capabilities Analyzer v1.0.0")
    print("=" * 60)
    
    try:
        # Создаем анализатор
        analyzer = ModelCapabilitiesAnalyzer(config_file=args.config)
        
        if not analyzer.model_configs:
            print("❌ Не найдено конфигураций моделей!")
            print("\nПример настройки переменных окружения:")
            print("  CHAT_ENDPOINT=http://localhost:8080")
            print("  CHAT_TOKEN=your_token")
            print("  CHAT_MODEL=chat")
            print("\nИли используйте --config для указания файла конфигурации")
            return 1
        
        print(f"📋 Найдено {len(analyzer.model_configs)} моделей для анализа")
        
        # Запускаем анализ
        report = await analyzer.analyze_all_models(quick_mode=args.quick)
        
        # Печатаем краткую сводку
        analyzer.print_summary(report)
        
        # Сохраняем отчет
        output_file = analyzer.save_report(report, args.output, args.filename)
        
        print(f"\n📄 Детальный отчет сохранен: {output_file}")
        
        # Дополнительные рекомендации
        success_rate = report['metadata']['success_rate']
        if success_rate > 80:
            print("🎉 Отличная совместимость! Можете использовать все возможности Kraken LLM")
        elif success_rate > 60:
            print("✅ Хорошая совместимость. Изучите рекомендации для оптимального использования")
        else:
            print("⚠️ Требуется настройка. Обратитесь к документации для решения проблем")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n❌ Анализ прерван пользователем")
        return 1
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        print(f"❌ Критическая ошибка: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))