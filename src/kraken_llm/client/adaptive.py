"""
Адаптивный клиент для Kraken LLM фреймворка.

Этот модуль предоставляет AdaptiveLLMClient, который автоматически
определяет возможности модели и выбирает оптимальный режим работы.
"""

import asyncio
import json
import time
import aiohttp
from typing import Dict, Any, List, Optional, Union, Set
from pydantic import BaseModel, Field
from enum import Enum
import logging

from .base import BaseLLMClient
from .standard import StandardLLMClient
from .streaming import StreamingLLMClient
from .structured import StructuredLLMClient
from .reasoning import ReasoningLLMClient
from .multimodal import MultimodalLLMClient
from ..exceptions.validation import ValidationError
from ..exceptions.api import APIError

logger = logging.getLogger(__name__)


class ModelCapability(str, Enum):
    """Перечисление возможностей модели"""
    CHAT_COMPLETION = "chat_completion"
    STREAMING = "streaming"
    FUNCTION_CALLING = "function_calling"
    TOOL_CALLING = "tool_calling"
    STRUCTURED_OUTPUT = "structured_output"
    VISION = "vision"
    AUDIO = "audio"
    VIDEO = "video"
    REASONING = "reasoning"
    EMBEDDINGS = "embeddings"
    CODE_GENERATION = "code_generation"
    MULTILINGUAL = "multilingual"


class ModelInfo(BaseModel):
    """Информация о модели"""
    name: str = Field(..., description="Название модели")
    provider: str = Field(..., description="Провайдер модели")
    capabilities: Set[ModelCapability] = Field(
        default_factory=set, description="Возможности модели")
    max_tokens: Optional[int] = Field(
        None, description="Максимальное количество токенов")
    context_length: Optional[int] = Field(None, description="Длина контекста")
    supports_system_messages: bool = Field(
        True, description="Поддержка системных сообщений")
    cost_per_token: Optional[float] = Field(
        None, description="Стоимость за токен")
    last_updated: Optional[float] = Field(
        None, description="Время последнего обновления информации")


class PerformanceMetrics(BaseModel):
    """Метрики производительности"""
    avg_latency: float = Field(0.0, description="Средняя задержка в секундах")
    success_rate: float = Field(1.0, description="Процент успешных запросов")
    total_requests: int = Field(0, description="Общее количество запросов")
    failed_requests: int = Field(
        0, description="Количество неудачных запросов")
    avg_tokens_per_second: float = Field(
        0.0, description="Средняя скорость генерации токенов")


class AdaptiveConfig(BaseModel):
    """Конфигурация адаптивного клиента"""
    capability_cache_ttl: int = Field(
        3600, description="TTL кэша возможностей в секундах")
    performance_window: int = Field(
        100, description="Окно для расчета метрик производительности")
    auto_fallback: bool = Field(
        True, description="Автоматический fallback при ошибках")
    prefer_streaming: bool = Field(
        False, description="Предпочитать streaming режим")
    capability_detection_timeout: float = Field(
        10.0, description="Таймаут определения возможностей")
    enable_performance_tracking: bool = Field(
        True, description="Включить отслеживание производительности")


class AdaptiveLLMClient(BaseLLMClient):
    """
    Адаптивный клиент, который автоматически определяет возможности модели
    и выбирает оптимальный режим работы.

    Особенности:
    - Автоматическое определение capabilities
    - Кэширование информации о модели
    - Fallback механизмы
    - Метрики производительности
    - Динамическое переключение между клиентами
    """

    def __init__(self, config, adaptive_config: Optional[AdaptiveConfig] = None):
        """
        Инициализация адаптивного клиента.

        Args:
            config: Базовая конфигурация LLM
            adaptive_config: Конфигурация адаптивного режима
        """
        super().__init__(config)
        self.adaptive_config = adaptive_config or AdaptiveConfig()

        # Кэш информации о модели
        self._model_info: Optional[ModelInfo] = None
        self._capabilities_cache: Dict[str, Any] = {}
        self._last_capability_check: Optional[float] = None

        # Клиенты для разных режимов
        self._clients: Dict[str, BaseLLMClient] = {}

        # Метрики производительности
        self._performance_metrics: Dict[str, PerformanceMetrics] = {}
        self._request_history: List[Dict[str, Any]] = []

        logger.info("Инициализирован AdaptiveLLMClient")

    async def get_model_capabilities(self, force_refresh: bool = False) -> Set[ModelCapability]:
        """
        Определяет возможности модели.

        Args:
            force_refresh: Принудительно обновить информацию о возможностях

        Returns:
            Набор возможностей модели
        """
        current_time = time.time()

        # Проверяем кэш
        if (not force_refresh and
            self._last_capability_check and
                current_time - self._last_capability_check < self.adaptive_config.capability_cache_ttl):
            return self._model_info.capabilities if self._model_info else set()

        print("Определяем возможности модели...")

        capabilities = set()

        try:
            # Тестируем базовые возможности
            basic_caps = await self._test_basic_capabilities()
            capabilities.update(basic_caps)
            print(f"   Базовые возможности: {[cap.value for cap in basic_caps]}")

            # Тестируем продвинутые возможности
            advanced_caps = await self._test_advanced_capabilities()
            capabilities.update(advanced_caps)
            print(f"   Продвинутые возможности: {[cap.value for cap in advanced_caps]}")

            # Обновляем информацию о модели
            self._model_info = ModelInfo(
                name=self.config.model,
                provider=self._detect_provider(),
                capabilities=capabilities,
                last_updated=current_time
            )

            self._last_capability_check = current_time

            print(f"   Итого обнаружено возможностей: {[cap.value for cap in capabilities]}")

        except Exception as e:
            print(f"Ошибка определения возможностей: {e}")
            # Возвращаем минимальный набор возможностей
            capabilities = {ModelCapability.CHAT_COMPLETION}

        return capabilities

    async def _test_basic_capabilities(self) -> Set[ModelCapability]:
        """Тестирует базовые возможности модели"""
        capabilities = set()

        try:
            # Тест базового chat completion
            test_messages = [{"role": "user", "content": "Hello"}]

            standard_client = self._get_client("standard")
            response = await asyncio.wait_for(
                standard_client.chat_completion(test_messages, max_tokens=5),
                timeout=self.adaptive_config.capability_detection_timeout
            )

            if response:
                capabilities.add(ModelCapability.CHAT_COMPLETION)
                logger.debug("✓ Chat completion поддерживается")

        except Exception as e:
            logger.debug(f"✗ Chat completion не поддерживается: {e}")

        try:
            # Тест streaming
            streaming_client = self._get_client("streaming")
            chunks = []

            async for chunk in streaming_client.chat_completion_stream(test_messages, max_tokens=5):
                chunks.append(chunk)
                if len(chunks) >= 2:  # Достаточно для проверки
                    break

            if chunks:
                capabilities.add(ModelCapability.STREAMING)
                logger.debug("✓ Streaming поддерживается")

        except Exception as e:
            logger.debug(f"✗ Streaming не поддерживается: {e}")

        return capabilities

    async def _test_advanced_capabilities(self) -> Set[ModelCapability]:
        """Тестирует продвинутые возможности модели"""
        capabilities = set()

        # Тест function calling с проверкой реального вызова функций
        try:
            test_functions = [{
                "name": "test_function",
                "description": "A test function that returns the number 42",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "number"}},
                    "required": ["x"]
                }
            }]

            standard_client = self._get_client("standard")
            response = await asyncio.wait_for(
                standard_client.chat_completion(
                    [{"role": "user", "content": "Please call test_function with x=1. You must use the function."}],
                    functions=test_functions,
                    function_call="auto",
                    max_tokens=100
                ),
                timeout=self.adaptive_config.capability_detection_timeout
            )

            # Логируем полный ответ для анализа
            logger.debug(f"Function calling тест - полный ответ: {response}")

            # Проверяем, содержит ли ответ реальный вызов функции или function call структуру
            response_str = str(response)
            response_lower = response_str.lower()
            
            # Проверяем структуру function call
            has_function_call_structure = any(indicator in response_str for indicator in [
                '"function_call"',
                '"name": "test_function"',
                '"arguments"'
            ])
            
            # Проверяем упоминание функции в контексте вызова
            has_function_mention = "test_function" in response_lower
            
            # Проверяем, что это НЕ отказ от использования функций
            is_refusal = any(indicator in response_lower for indicator in [
                "i don't have",
                "i cannot",
                "i'm not able",
                "к сожалению",
                "не могу использовать",
                "не имею доступа",
                "функции недоступны"
            ])

            # Function calling поддерживается, если есть структура ИЛИ упоминание функции, но НЕ отказ
            if (has_function_call_structure or has_function_mention) and not is_refusal:
                capabilities.add(ModelCapability.FUNCTION_CALLING)
                logger.debug("✓ Function calling поддерживается")
            else:
                logger.debug(f"✗ Function calling: отказ или отсутствие структуры вызова")
                if is_refusal:
                    logger.debug("Модель отказывается использовать функции")

        except APIError as e:
            # Проверяем специфические ошибки API
            if "500" in str(e) or "not supported" in str(e).lower():
                logger.debug(f"✗ Function calling не поддерживается API: {e}")
            else:
                logger.debug(f"✗ Function calling ошибка: {e}")
        except Exception as e:
            logger.debug(f"✗ Function calling не поддерживается: {e}")

        # Тест tool calling с проверкой реального вызова инструментов
        try:
            test_tools = [{
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "A test tool that returns a greeting",
                    "parameters": {
                        "type": "object",
                        "properties": {"y": {"type": "string"}},
                        "required": ["y"]
                    }
                }
            }]

            response = await asyncio.wait_for(
                standard_client.chat_completion(
                    [{"role": "user", "content": "Please use test_tool with y='hello'. You must use the tool."}],
                    tools=test_tools,
                    tool_choice="auto",
                    max_tokens=100
                ),
                timeout=self.adaptive_config.capability_detection_timeout
            )

            # Логируем полный ответ для анализа
            logger.debug(f"Tool calling тест - полный ответ: {response}")

            # Проверяем, содержит ли ответ реальный вызов инструмента или tool call структуру
            response_str = str(response)
            response_lower = response_str.lower()
            
            # Проверяем структуру tool call
            has_tool_call_structure = any(indicator in response_str for indicator in [
                '"tool_calls"',
                '"type": "function"',
                '"name": "test_tool"',
                '"arguments"'
            ])
            
            # Проверяем упоминание инструмента в контексте вызова
            has_tool_mention = "test_tool" in response_lower
            
            # Проверяем, что это НЕ отказ от использования инструментов
            is_refusal = any(indicator in response_lower for indicator in [
                "i don't have",
                "i cannot", 
                "i'm not able",
                "к сожалению",
                "не могу использовать",
                "не имею доступа",
                "инструменты недоступны"
            ])

            # Tool calling поддерживается, если есть структура ИЛИ упоминание инструмента, но НЕ отказ
            if (has_tool_call_structure or has_tool_mention) and not is_refusal:
                capabilities.add(ModelCapability.TOOL_CALLING)
                logger.debug("✓ Tool calling поддерживается")
            else:
                logger.debug(f"✗ Tool calling: отказ или отсутствие структуры вызова")
                if is_refusal:
                    logger.debug("Модель отказывается использовать инструменты")

        except APIError as e:
            # Проверяем специфические ошибки API
            if "500" in str(e) or "not supported" in str(e).lower():
                logger.debug(f"✗ Tool calling не поддерживается API: {e}")
            else:
                logger.debug(f"✗ Tool calling ошибка: {e}")
        except Exception as e:
            logger.debug(f"✗ Tool calling не поддерживается: {e}")

        # Дополнительная проверка через прямые HTTP запросы
        capabilities.update(await self._test_direct_api_capabilities())

        # Тест structured output
        try:
            from pydantic import BaseModel

            class TestModel(BaseModel):
                name: str
                value: int

            structured_client = self._get_client("structured")
            result = await asyncio.wait_for(
                structured_client.chat_completion_structured(
                    [{"role": "user", "content": "Return JSON with name='test' and value=42"}],
                    TestModel,
                    max_tokens=50
                ),
                timeout=self.adaptive_config.capability_detection_timeout
            )

            if result and hasattr(result, 'name') and hasattr(result, 'value'):
                capabilities.add(ModelCapability.STRUCTURED_OUTPUT)
                logger.debug("✓ Structured output поддерживается")

        except Exception as e:
            logger.debug(f"✗ Structured output не поддерживается: {e}")

        # Тест vision (если есть тестовое изображение)
        try:
            # Создаем простое тестовое изображение в base64
            test_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

            vision_messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do you see?"},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{test_image_b64}"}}
                ]
            }]

            response = await asyncio.wait_for(
                standard_client.chat_completion(
                    vision_messages, max_tokens=20),
                timeout=self.adaptive_config.capability_detection_timeout
            )

            if response and len(response) > 5:  # Получили осмысленный ответ
                capabilities.add(ModelCapability.VISION)
                logger.debug("✓ Vision поддерживается")

        except Exception as e:
            logger.debug(f"✗ Vision не поддерживается: {e}")

        return capabilities

    async def _test_direct_api_capabilities(self) -> Set[ModelCapability]:
        """Тестирует возможности через прямые HTTP запросы к API"""
        capabilities = set()

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json"
                }

                # Тест function calling
                function_payload = {
                    "model": self.config.model,
                    "messages": [{"role": "user", "content": "Test function calling"}],
                    "functions": [{
                        "name": "test_func",
                        "description": "Test function",
                        "parameters": {"type": "object", "properties": {}}
                    }],
                    "max_tokens": 10,
                    "stream": False
                }

                try:
                    async with session.post(
                        f"{self.config.endpoint}/v1/chat/completions",
                        headers=headers,
                        json=function_payload,
                        timeout=aiohttp.ClientTimeout(
                            total=self.adaptive_config.capability_detection_timeout)
                    ) as response:
                        if response.status == 200:
                            capabilities.add(ModelCapability.FUNCTION_CALLING)
                            logger.debug(
                                "✓ Function calling поддерживается (прямая проверка)")
                        elif response.status == 500:
                            logger.debug(
                                "✗ Function calling не поддерживается (500 ошибка)")
                        else:
                            logger.debug(
                                f"✗ Function calling неопределенный статус: {response.status}")
                except Exception as e:
                    logger.debug(
                        f"✗ Function calling прямая проверка не удалась: {e}")

                # Тест tool calling
                tool_payload = {
                    "model": self.config.model,
                    "messages": [{"role": "user", "content": "Test tool calling"}],
                    "tools": [{
                        "type": "function",
                        "function": {
                            "name": "test_tool",
                            "description": "Test tool",
                            "parameters": {"type": "object", "properties": {}}
                        }
                    }],
                    "max_tokens": 10,
                    "stream": False
                }

                try:
                    async with session.post(
                        f"{self.config.endpoint}/v1/chat/completions",
                        headers=headers,
                        json=tool_payload,
                        timeout=aiohttp.ClientTimeout(
                            total=self.adaptive_config.capability_detection_timeout)
                    ) as response:
                        if response.status == 200:
                            capabilities.add(ModelCapability.TOOL_CALLING)
                            logger.debug(
                                "✓ Tool calling поддерживается (прямая проверка)")
                        elif response.status == 500:
                            logger.debug(
                                "✗ Tool calling не поддерживается (500 ошибка)")
                        else:
                            logger.debug(
                                f"✗ Tool calling неопределенный статус: {response.status}")
                except Exception as e:
                    logger.debug(
                        f"✗ Tool calling прямая проверка не удалась: {e}")

        except Exception as e:
            logger.debug(f"Ошибка прямой проверки API: {e}")

        return capabilities

    def _detect_provider(self) -> str:
        """Определяет провайдера модели по endpoint и модели"""
        endpoint = self.config.endpoint.lower()
        model = self.config.model.lower()

        if "openai" in endpoint or "gpt" in model:
            return "openai"
        elif "anthropic" in endpoint or "claude" in model:
            return "anthropic"
        elif "cohere" in endpoint:
            return "cohere"
        elif "huggingface" in endpoint:
            return "huggingface"
        elif "ollama" in endpoint:
            return "ollama"
        else:
            return "unknown"

    def _get_client(self, client_type: str) -> BaseLLMClient:
        """Получает клиент указанного типа"""
        if client_type not in self._clients:
            if client_type == "standard":
                self._clients[client_type] = StandardLLMClient(self.config)
            elif client_type == "streaming":
                self._clients[client_type] = StreamingLLMClient(self.config)
            elif client_type == "structured":
                self._clients[client_type] = StructuredLLMClient(self.config)
            elif client_type == "reasoning":
                self._clients[client_type] = ReasoningLLMClient(self.config)
            elif client_type == "multimodal":
                self._clients[client_type] = MultimodalLLMClient(self.config)
            else:
                raise ValueError(f"Неизвестный тип клиента: {client_type}")

        return self._clients[client_type]

    async def smart_completion(
        self,
        messages: List[Dict[str, str]],
        preferred_mode: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Умный completion, который автоматически выбирает оптимальный режим.

        Args:
            messages: Сообщения для обработки
            preferred_mode: Предпочитаемый режим (standard, streaming, structured, etc.)
            **kwargs: Дополнительные параметры

        Returns:
            Ответ модели
        """
        start_time = time.time()

        # Определяем возможности модели
        capabilities = await self.get_model_capabilities()

        # Выбираем оптимальный режим
        selected_mode = self._select_optimal_mode(
            messages, capabilities, preferred_mode, **kwargs
        )

        logger.info(f"Выбран режим: {selected_mode}")

        try:
            # Выполняем запрос
            client = self._get_client(selected_mode)

            if selected_mode == "streaming" and self.adaptive_config.prefer_streaming:
                # Агрегируем streaming ответ
                chunks = []
                async for chunk in client.chat_completion_stream(messages, **kwargs):
                    chunks.append(chunk)
                result = "".join(chunks)
            elif selected_mode == "structured":
                # Для structured режима используем специальный метод
                response_model = kwargs.pop(
                    "response_model", None)  # Убираем из kwargs
                if response_model:
                    # Убираем response_model из kwargs для избежания дублирования
                    structured_kwargs = {
                        k: v for k, v in kwargs.items() if k != "response_model"}
                    result = await client.chat_completion_structured(messages, response_model, **structured_kwargs)
                else:
                    # Fallback к стандартному режиму если нет response_model
                    logger.warning(
                        "Structured режим выбран, но response_model не указан, переключаемся на standard")
                    standard_client = self._get_client("standard")
                    result = await standard_client.chat_completion(messages, **kwargs)
            else:
                # Для остальных режимов убираем response_model из kwargs
                clean_kwargs = {k: v for k,
                                v in kwargs.items() if k != "response_model"}
                result = await client.chat_completion(messages, **clean_kwargs)

            # Записываем метрики
            self._record_performance_metrics(selected_mode, start_time, True)

            return result

        except Exception as e:
            # Записываем ошибку
            self._record_performance_metrics(selected_mode, start_time, False)

            # Пробуем fallback если включен
            if self.adaptive_config.auto_fallback and selected_mode != "standard":
                logger.warning(
                    f"Ошибка в режиме {selected_mode}, пробуем fallback: {e}")
                return await self._fallback_completion(messages, **kwargs)

            raise APIError(f"Ошибка в режиме {selected_mode}: {e}")

    def _select_optimal_mode(
        self,
        messages: List[Dict[str, str]],
        capabilities: Set[ModelCapability],
        preferred_mode: Optional[str],
        **kwargs
    ) -> str:
        """Выбирает оптимальный режим работы"""

        # Если указан предпочитаемый режим и он поддерживается
        if preferred_mode:
            mode_capability_map = {
                "streaming": ModelCapability.STREAMING,
                "structured": ModelCapability.STRUCTURED_OUTPUT,
                "reasoning": ModelCapability.REASONING,
                "multimodal": ModelCapability.VISION
            }

            required_cap = mode_capability_map.get(preferred_mode)
            if not required_cap or required_cap in capabilities:
                return preferred_mode

        # Автоматический выбор на основе контекста

        # Проверяем наличие изображений в сообщениях
        has_images = any(
            isinstance(msg.get("content"), list) and
            any(item.get("type") ==
                "image_url" for item in msg["content"] if isinstance(item, dict))
            for msg in messages
        )

        if has_images and ModelCapability.VISION in capabilities:
            return "multimodal"

        # Проверяем запрос на structured output (только если есть response_model)
        if (kwargs.get("response_model") and
                ModelCapability.STRUCTURED_OUTPUT in capabilities):
            return "structured"

        # Проверяем запрос на рассуждения
        reasoning_keywords = ["пошагово", "step by step",
                              "рассуждение", "reasoning", "объясни"]
        if any(keyword in str(messages).lower() for keyword in reasoning_keywords):
            if ModelCapability.REASONING in capabilities:
                return "reasoning"

        # Проверяем предпочтение streaming
        if (self.adaptive_config.prefer_streaming and
                ModelCapability.STREAMING in capabilities):
            return "streaming"

        # По умолчанию используем standard
        return "standard"

    async def _fallback_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Fallback completion через стандартный клиент"""
        try:
            standard_client = self._get_client("standard")
            return await standard_client.chat_completion(messages, **kwargs)
        except Exception as e:
            raise APIError(f"Fallback также не удался: {e}")

    def _record_performance_metrics(self, mode: str, start_time: float, success: bool):
        """Записывает метрики производительности"""
        if not self.adaptive_config.enable_performance_tracking:
            return

        execution_time = time.time() - start_time

        if mode not in self._performance_metrics:
            self._performance_metrics[mode] = PerformanceMetrics()

        metrics = self._performance_metrics[mode]
        metrics.total_requests += 1

        if success:
            # Обновляем среднюю задержку
            total_latency = metrics.avg_latency * \
                (metrics.total_requests - 1) + execution_time
            metrics.avg_latency = total_latency / metrics.total_requests
        else:
            metrics.failed_requests += 1

        # Обновляем success rate
        metrics.success_rate = (
            metrics.total_requests - metrics.failed_requests) / metrics.total_requests

        # Добавляем в историю
        self._request_history.append({
            "mode": mode,
            "timestamp": time.time(),
            "execution_time": execution_time,
            "success": success
        })

        # Ограничиваем размер истории
        if len(self._request_history) > self.adaptive_config.performance_window:
            self._request_history.pop(0)

    def get_performance_report(self) -> Dict[str, Any]:
        """Возвращает отчет о производительности"""
        return {
            "model_info": self._model_info.dict() if self._model_info else None,
            "performance_metrics": {
                mode: metrics.dict() for mode, metrics in self._performance_metrics.items()
            },
            "recent_requests": len(self._request_history),
            "cache_status": {
                "last_capability_check": self._last_capability_check,
                "cache_age": time.time() - self._last_capability_check if self._last_capability_check else None
            }
        }

    async def detect_model_capabilities(self) -> Dict[str, bool]:
        """
        Определяет возможности модели и возвращает их в виде словаря.

        Returns:
            Словарь с возможностями модели
        """
        capabilities = await self.get_model_capabilities()

        return {
            "streaming": ModelCapability.STREAMING in capabilities,
            "structured_output": ModelCapability.STRUCTURED_OUTPUT in capabilities,
            "function_calling": ModelCapability.FUNCTION_CALLING in capabilities,
            "tool_calling": ModelCapability.TOOL_CALLING in capabilities,
            "vision": ModelCapability.VISION in capabilities,
            "audio": ModelCapability.AUDIO in capabilities,
            "video": ModelCapability.VIDEO in capabilities,
            "reasoning": ModelCapability.REASONING in capabilities,
            "embeddings": ModelCapability.EMBEDDINGS in capabilities
        }

    async def chat_completion_adaptive(
        self,
        messages: List[Dict[str, str]],
        preferred_mode: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Выполняет адаптивный chat completion с автоматическим выбором режима.

        Args:
            messages: Сообщения для обработки
            preferred_mode: Предпочтительный режим (опционально)
            **kwargs: Дополнительные параметры

        Returns:
            Ответ модели
        """
        return await self.smart_completion(
            messages=messages,
            preferred_mode=preferred_mode,
            **kwargs
        )

    # Методы базового класса
    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Делегирует к smart_completion"""
        return await self.smart_completion(messages, **kwargs)

    async def chat_completion_stream(self, messages: List[Dict[str, str]], **kwargs):
        """Streaming через адаптивный режим"""
        capabilities = await self.get_model_capabilities()

        if ModelCapability.STREAMING in capabilities:
            streaming_client = self._get_client("streaming")
            async for chunk in streaming_client.chat_completion_stream(messages, **kwargs):
                yield chunk
        else:
            # Эмулируем streaming через обычный запрос
            result = await self.smart_completion(messages, **kwargs)
            for char in result:
                yield char
                await asyncio.sleep(0.01)  # Небольшая задержка для эмуляции

    async def chat_completion_structured(self, messages: List[Dict[str, str]], response_model, **kwargs):
        """Structured output через адаптивный режим"""
        capabilities = await self.get_model_capabilities()

        # Добавляем response_model в kwargs для передачи в smart_completion
        kwargs["response_model"] = response_model

        if ModelCapability.STRUCTURED_OUTPUT in capabilities:
            structured_client = self._get_client("structured")
            # Убираем response_model из kwargs чтобы избежать дублирования
            structured_kwargs = {k: v for k,
                                 v in kwargs.items() if k != "response_model"}
            return await structured_client.chat_completion_structured(messages, response_model, **structured_kwargs)
        else:
            raise NotImplementedError(
                "Модель не поддерживает structured output")
