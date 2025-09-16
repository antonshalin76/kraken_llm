#!/usr/bin/env python3
"""
Universal Multi-Client для Kraken LLM

Универсальный клиент, который объединяет все возможности различных специализированных
клиентов в единый интерфейс с гибкой конфигурацией.

Позволяет пользователю:
- Настроить только нужные возможности
- Использовать единый интерфейс для всех операций
- Автоматически выбирать оптимальный режим выполнения
- Легко переключаться между возможностями
"""

import logging
from typing import Dict, List, Any, Optional, Union, Set, Type, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from pydantic import BaseModel

from ..config.settings import LLMConfig
from ..exceptions.base import KrakenError
from ..exceptions.validation import ValidationError
from .base import BaseLLMClient
from .standard import StandardLLMClient
from .streaming import StreamingLLMClient
from .structured import StructuredLLMClient
from .reasoning import ReasoningLLMClient, ReasoningConfig, ReasoningModelType
from .multimodal import MultimodalLLMClient, MultimodalConfig
from .adaptive import AdaptiveLLMClient, AdaptiveConfig
from .asr import ASRClient, ASRConfig
from .embeddings import EmbeddingsLLMClient
from .completion import CompletionLLMClient
import os

logger = logging.getLogger(__name__)


class UniversalCapability(str, Enum):
    """Возможности универсального клиента"""
    # Базовые возможности
    CHAT_COMPLETION = "chat_completion"
    STREAMING = "streaming"
    COMPLETION_LEGACY = "completion_legacy"

    # Структурированный вывод
    STRUCTURED_OUTPUT = "structured_output"

    # Вызов функций и инструментов
    FUNCTION_CALLING = "function_calling"
    TOOL_CALLING = "tool_calling"

    # Векторные представления
    EMBEDDINGS = "embeddings"
    SIMILARITY_SEARCH = "similarity_search"

    # Рассуждения
    REASONING = "reasoning"
    NATIVE_THINKING = "native_thinking"

    # Мультимодальность
    MULTIMODAL = "multimodal"
    VISION = "vision"
    AUDIO = "audio"

    # Речевые технологии
    ASR = "asr"
    STT = "speech_to_text"
    TTS = "text_to_speech"

    # Адаптивные возможности
    ADAPTIVE = "adaptive"
    AUTO_DETECTION = "auto_detection"


@dataclass
class UniversalClientConfig:
    """Конфигурация универсального клиента"""

    # Основные возможности
    capabilities: Set[UniversalCapability] = field(default_factory=set)

    # Конфигурации для специализированных клиентов
    reasoning_config: Optional[ReasoningConfig] = None
    multimodal_config: Optional[MultimodalConfig] = None
    adaptive_config: Optional[AdaptiveConfig] = None
    asr_config: Optional[ASRConfig] = None

    # Настройки поведения
    auto_fallback: bool = True  # Автоматический fallback на базовые возможности
    prefer_streaming: bool = False  # Предпочитать streaming режим когда возможно
    enable_caching: bool = True  # Кэширование клиентов

    # Настройки производительности
    concurrent_requests: bool = True  # Поддержка параллельных запросов
    timeout_multiplier: float = 1.0  # Множитель для таймаутов

    @classmethod
    def from_capabilities_report(cls, capabilities_report: Dict[str, Any],
                                 model_name: Optional[str] = None) -> 'UniversalClientConfig':
        """
        Создание конфигурации на основе отчета анализатора возможностей

        Args:
            capabilities_report: Отчет от ModelCapabilitiesAnalyzer
            model_name: Имя модели (если None, используется первая доступная)
        """
        capabilities = set()

        # Определяем модель для анализа
        if model_name:
            model_summary = capabilities_report.get(
                'model_summaries', {}).get(model_name)
        else:
            # Берем первую доступную модель
            model_summaries = capabilities_report.get('model_summaries', {})
            model_summary = next(iter(model_summaries.values())
                                 ) if model_summaries else None

        if not model_summary:
            logger.warning(
                "Не найдена информация о модели, используется базовая конфигурация")
            return cls(capabilities={UniversalCapability.CHAT_COMPLETION})

        # Маппинг возможностей из отчета в UniversalCapability
        capability_mapping = {
            'chat_completion': UniversalCapability.CHAT_COMPLETION,
            'streaming': UniversalCapability.STREAMING,
            'structured_output_native': UniversalCapability.STRUCTURED_OUTPUT,
            'structured_output_outlines': UniversalCapability.STRUCTURED_OUTPUT,
            'function_calling': UniversalCapability.FUNCTION_CALLING,
            'tool_calling': UniversalCapability.TOOL_CALLING,
            'embeddings': UniversalCapability.EMBEDDINGS,
            'similarity_search': UniversalCapability.SIMILARITY_SEARCH,
            'reasoning_cot': UniversalCapability.REASONING,
            'reasoning_native_thinking': UniversalCapability.NATIVE_THINKING,
            'multimodal_vision': UniversalCapability.VISION,
            'multimodal_audio': UniversalCapability.AUDIO,
            'asr_stt': UniversalCapability.STT,
            'asr_tts': UniversalCapability.TTS,
            'adaptive_mode': UniversalCapability.ADAPTIVE,
            'completion_legacy': UniversalCapability.COMPLETION_LEGACY,
        }

        # Добавляем подтвержденные возможности
        confirmed_caps = model_summary.get('confirmed_capabilities', [])
        for cap_info in confirmed_caps:
            cap_name = cap_info.get('capability')
            if cap_name in capability_mapping:
                capabilities.add(capability_mapping[cap_name])

        # Настройка специализированных конфигураций
        reasoning_config = None
        if UniversalCapability.REASONING in capabilities or UniversalCapability.NATIVE_THINKING in capabilities:
            reasoning_config = ReasoningConfig(
                model_type=ReasoningModelType.NATIVE_THINKING if UniversalCapability.NATIVE_THINKING in capabilities else ReasoningModelType.PROMPT_BASED,
                enable_cot=UniversalCapability.REASONING in capabilities
            )

        multimodal_config = None
        if UniversalCapability.VISION in capabilities or UniversalCapability.AUDIO in capabilities:
            multimodal_config = MultimodalConfig()

        adaptive_config = None
        if UniversalCapability.ADAPTIVE in capabilities:
            adaptive_config = AdaptiveConfig()

        asr_config = None
        if UniversalCapability.STT in capabilities or UniversalCapability.TTS in capabilities:
            asr_config = ASRConfig()

        return cls(
            capabilities=capabilities,
            reasoning_config=reasoning_config,
            multimodal_config=multimodal_config,
            adaptive_config=adaptive_config,
            asr_config=asr_config,
            prefer_streaming=UniversalCapability.STREAMING in capabilities
        )

    @classmethod
    def all_capabilities(cls) -> 'UniversalClientConfig':
        """Конфигурация со всеми возможными возможностями"""
        return cls(
            capabilities=set(UniversalCapability),
            reasoning_config=ReasoningConfig(),
            multimodal_config=MultimodalConfig(),
            adaptive_config=AdaptiveConfig(),
            asr_config=ASRConfig(),
            prefer_streaming=True
        )

    @classmethod
    def basic(cls) -> 'UniversalClientConfig':
        """Базовая конфигурация для простых задач"""
        return cls(
            capabilities={
                UniversalCapability.CHAT_COMPLETION,
                UniversalCapability.STREAMING
            }
        )

    @classmethod
    def advanced(cls) -> 'UniversalClientConfig':
        """Продвинутая конфигурация для сложных задач"""
        return cls(
            capabilities={
                UniversalCapability.CHAT_COMPLETION,
                UniversalCapability.STREAMING,
                UniversalCapability.STRUCTURED_OUTPUT,
                UniversalCapability.FUNCTION_CALLING,
                UniversalCapability.TOOL_CALLING,
                UniversalCapability.REASONING
            },
            reasoning_config=ReasoningConfig(),
            prefer_streaming=True
        )


class UniversalLLMClient(BaseLLMClient):
    """
    Универсальный клиент, объединяющий все возможности Kraken LLM

    Предоставляет единый интерфейс для всех типов операций с LLM,
    автоматически выбирая оптимальный специализированный клиент
    для каждой задачи.
    """

    def __init__(self, config: LLMConfig, universal_config: Optional[UniversalClientConfig] = None):
        """
        Инициализация универсального клиента

        Args:
            config: Базовая конфигурация LLM
            universal_config: Конфигурация возможностей (по умолчанию - базовая)
        """
        super().__init__(config)

        self.universal_config = universal_config or UniversalClientConfig.basic()
        self._clients: Dict[str, BaseLLMClient] = {}
        self._initialized = False

        logger.info(
            f"Создан UniversalLLMClient с возможностями: {[cap.value for cap in self.universal_config.capabilities]}")

    async def __aenter__(self):
        """Асинхронный контекстный менеджер - вход"""
        await self._initialize_clients()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Асинхронный контекстный менеджер - выход"""
        await self._cleanup_clients()

    async def _initialize_clients(self) -> None:
        """Инициализация специализированных клиентов"""
        if self._initialized:
            return

        logger.info("Инициализация специализированных клиентов...")

        # Базовые kwargs из текущего config для наследования не сетевых параметров
        base_kwargs = self.config.model_dump()

        # Локальный хелпер: выбрать профильный конфиг из .env
        def _create_config_for_client_type(client_type: str, base_kwargs_local: Dict[str, Any]) -> LLMConfig:
            profile_map = {
                'standard': ['CHAT', 'LLM'],
                'streaming': ['CHAT', 'LLM'],
                'structured': ['CHAT', 'LLM'],
                'reasoning': ['LLM_REASONING', 'REASONING', 'LLM'],
                'multimodal': ['MULTIMODAL', 'LLM'],
                'adaptive': ['LLM'],
                'asr': ['ASR', 'LLM'],
                'embeddings': ['EMBEDDING', 'LLM'],
                'completion': ['COMPLETION', 'LLM'],
                'universal': ['LLM']
            }
            profiles = profile_map.get(client_type or 'adaptive', ['LLM'])
            for prefix in profiles:
                endpoint = os.getenv(f"{prefix}_ENDPOINT")
                model = os.getenv(f"{prefix}_MODEL")
                token = os.getenv(f"{prefix}_TOKEN") or os.getenv(f"{prefix}_API_KEY") or os.getenv(f"{prefix}_KEY")
                if endpoint and model:
                    cfg_kwargs = {
                        **base_kwargs_local,
                        'endpoint': endpoint,
                        'model': model,
                    }
                    if token:
                        cfg_kwargs['api_key'] = token
                    return LLMConfig(**cfg_kwargs)
            return LLMConfig(**base_kwargs_local)

        # Всегда создаем базовый клиент (используем профиль CHAT/LLM из .env, если есть)
        std_cfg = _create_config_for_client_type('standard', base_kwargs)
        self._clients['standard'] = StandardLLMClient(std_cfg)

        # Создаем клиенты на основе конфигурации (подбираем профильные ENDPOINT/MODEL/TOKEN из .env)
        if UniversalCapability.STREAMING in self.universal_config.capabilities:
            stream_cfg = _create_config_for_client_type('streaming', base_kwargs)
            self._clients['streaming'] = StreamingLLMClient(stream_cfg)

        if UniversalCapability.STRUCTURED_OUTPUT in self.universal_config.capabilities:
            so_cfg = _create_config_for_client_type('structured', base_kwargs)
            self._clients['structured'] = StructuredLLMClient(so_cfg)

        if (UniversalCapability.REASONING in self.universal_config.capabilities or
                UniversalCapability.NATIVE_THINKING in self.universal_config.capabilities):
            reason_cfg = _create_config_for_client_type('reasoning', base_kwargs)
            self._clients['reasoning'] = ReasoningLLMClient(
                reason_cfg,
                self.universal_config.reasoning_config
            )

        if (UniversalCapability.MULTIMODAL in self.universal_config.capabilities or
            UniversalCapability.VISION in self.universal_config.capabilities or
                UniversalCapability.AUDIO in self.universal_config.capabilities):
            mm_cfg = _create_config_for_client_type('multimodal', base_kwargs)
            self._clients['multimodal'] = MultimodalLLMClient(
                mm_cfg,
                self.universal_config.multimodal_config
            )

        if UniversalCapability.ADAPTIVE in self.universal_config.capabilities:
            ad_cfg = _create_config_for_client_type('adaptive', base_kwargs)
            self._clients['adaptive'] = AdaptiveLLMClient(
                ad_cfg,
                self.universal_config.adaptive_config
            )

        if (UniversalCapability.ASR in self.universal_config.capabilities or
            UniversalCapability.STT in self.universal_config.capabilities or
                UniversalCapability.TTS in self.universal_config.capabilities):
            asr_cfg = _create_config_for_client_type('asr', base_kwargs)
            self._clients['asr'] = ASRClient(
                asr_cfg,
                self.universal_config.asr_config
            )

        if UniversalCapability.EMBEDDINGS in self.universal_config.capabilities:
            emb_cfg = _create_config_for_client_type('embeddings', base_kwargs)
            self._clients['embeddings'] = EmbeddingsLLMClient(emb_cfg)

        if UniversalCapability.COMPLETION_LEGACY in self.universal_config.capabilities:
            comp_cfg = _create_config_for_client_type('completion', base_kwargs)
            self._clients['completion'] = CompletionLLMClient(comp_cfg)

        # Инициализируем все клиенты
        for client_name, client in self._clients.items():
            try:
                await client.__aenter__()
                logger.debug(f"Инициализирован клиент: {client_name}")
            except Exception as e:
                logger.warning(
                    f"Не удалось инициализировать клиент {client_name}: {e}")
                if not self.universal_config.auto_fallback:
                    raise

        self._initialized = True
        logger.info(f"Инициализировано {len(self._clients)} клиентов")

    async def _cleanup_clients(self) -> None:
        """Очистка ресурсов клиентов"""
        for client_name, client in self._clients.items():
            try:
                await client.__aexit__(None, None, None)
                logger.debug(f"Очищен клиент: {client_name}")
            except Exception as e:
                logger.warning(
                    f"Ошибка при очистке клиента {client_name}: {e}")

        self._clients.clear()
        self._initialized = False

    def _get_optimal_client(self, operation: str, **kwargs) -> BaseLLMClient:
        """
        Выбор оптимального клиента для операции

        Args:
            operation: Тип операции
            **kwargs: Дополнительные параметры

        Returns:
            Оптимальный клиент для операции
        """
        # Логика выбора клиента на основе операции и параметров

        # Для structured output
        if operation == 'structured' and 'structured' in self._clients:
            return self._clients['structured']

        # Для reasoning
        if operation in ['reasoning', 'thinking'] and 'reasoning' in self._clients:
            return self._clients['reasoning']

        # Для multimodal
        if operation in ['vision', 'multimodal', 'audio'] and 'multimodal' in self._clients:
            return self._clients['multimodal']

        # Для embeddings
        if operation in ['embeddings', 'similarity'] and 'embeddings' in self._clients:
            return self._clients['embeddings']

        # Для ASR
        if operation in ['stt', 'tts', 'asr'] and 'asr' in self._clients:
            return self._clients['asr']

        # Для completion
        if operation == 'completion' and 'completion' in self._clients:
            return self._clients['completion']

        # Для streaming (если предпочтителен или явно запрошен)
        if (kwargs.get('stream') or self.universal_config.prefer_streaming) and 'streaming' in self._clients:
            return self._clients['streaming']

        # Adaptive клиент как универсальный выбор
        if 'adaptive' in self._clients:
            return self._clients['adaptive']

        # Fallback на стандартный клиент
        return self._clients.get('standard')

    async def _structured_fallback(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[BaseModel],
        **kwargs
    ) -> BaseModel:
        """
        Fallback метод для structured output через обычный chat completion
        с последующим парсингом JSON
        """
        from ..structured.validator import StructuredOutputValidator

        # Улучшаем промпт для получения JSON
        enhanced_messages = self._enhance_messages_for_json(
            messages, response_model)

        # Получаем ответ через обычный chat completion
        response = await self.chat_completion(
            enhanced_messages,
            max_tokens=kwargs.get('max_tokens', 1000),
            # Низкая температура для стабильности
            temperature=kwargs.get('temperature', 0.1),
            **{k: v for k, v in kwargs.items() if k not in ['max_tokens', 'temperature']}
        )

        # Парсим ответ в structured format
        validator = StructuredOutputValidator()
        return validator.validate_response(response, response_model, strict=False)

    def _enhance_messages_for_json(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[BaseModel]
    ) -> List[Dict[str, str]]:
        """Улучшение промпта для получения валидного JSON"""
        from ..structured.validator import StructuredOutputValidator

        # Генерируем пример JSON
        validator = StructuredOutputValidator()
        example = validator.create_example_instance(response_model)
        example_json = example.model_dump_json(indent=2)

        # Создаем системный промпт
        system_prompt = {
            "role": "system",
            "content": f"""Отвечай ТОЛЬКО валидным JSON в точном соответствии со схемой.
Не добавляй никакого дополнительного текста, объяснений или markdown форматирования.

Схема ответа:
{response_model.model_json_schema()}

Пример правильного ответа:
{example_json}

ВАЖНО: Ответ должен содержать ТОЛЬКО JSON, без дополнительного текста!"""
        }

        # Добавляем системный промпт в начало, если его нет
        enhanced_messages = []
        has_system = any(msg.get('role') == 'system' for msg in messages)

        if not has_system:
            enhanced_messages.append(system_prompt)

        # Добавляем оригинальные сообщения
        for msg in messages:
            if msg.get('role') == 'system' and not has_system:
                # Объединяем с нашим системным промптом
                enhanced_msg = {
                    "role": "system",
                    "content": f"{system_prompt['content']}\n\n{msg['content']}"
                }
                enhanced_messages.append(enhanced_msg)
                has_system = True
            else:
                enhanced_messages.append(msg.copy())

        # Улучшаем последнее пользовательское сообщение
        if enhanced_messages and enhanced_messages[-1].get('role') == 'user':
            last_msg = enhanced_messages[-1]
            last_msg['content'] += f"\n\nОтветь в формате JSON согласно схеме {response_model.__name__}:"

        return enhanced_messages

    # Основные методы интерфейса

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = None,
        **kwargs
    ) -> Any:
        """
        Базовое завершение чата

        Автоматически выбирает оптимальный клиент на основе параметров
        """
        if not self._initialized:
            await self._initialize_clients()

        # Определяем тип операции
        operation = 'streaming' if stream else 'chat'

        client = self._get_optimal_client(operation, stream=stream, **kwargs)

        if not client:
            raise KrakenError("Нет доступного клиента для chat completion")

        return await client.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            **kwargs
        )

    async def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncGenerator[Any, None]:
        """Потоковое завершение чата"""
        if not self._initialized:
            await self._initialize_clients()

        client = self._get_optimal_client('streaming', stream=True, **kwargs)

        if not client or not hasattr(client, 'chat_completion_stream'):
            # Fallback на обычный chat_completion с stream=True
            response = await self.chat_completion(messages, stream=True, **kwargs)
            if hasattr(response, '__aiter__'):
                async for chunk in response:
                    yield chunk
            else:
                yield response
            return

        async for chunk in client.chat_completion_stream(messages, **kwargs):
            yield chunk

    async def chat_completion_structured(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[BaseModel],
        **kwargs
    ) -> BaseModel:
        """
        Структурированное завершение чата с автоматическим fallback

        Пытается использовать нативный structured output, при неудаче
        автоматически переключается на Outlines режим.
        """
        if not self._initialized:
            await self._initialize_clients()

        if UniversalCapability.STRUCTURED_OUTPUT not in self.universal_config.capabilities:
            raise ValidationError(
                "Structured output не настроен в конфигурации")

        client = self._get_optimal_client('structured', **kwargs)

        if not client or not hasattr(client, 'chat_completion_structured'):
            raise KrakenError("Нет доступного клиента для structured output")

        # Сначала пытаемся нативный режим
        try:
            logger.debug("Попытка нативного structured output")
            return await client.chat_completion_structured(
                messages=messages,
                response_model=response_model,
                stream=False,  # Нативный режим
                **kwargs
            )
        except Exception as native_error:
            logger.warning(
                f"Нативный structured output не удался: {native_error}")

            # Fallback на Outlines режим
            try:
                logger.info(
                    "Переключение на Outlines режим для structured output")

                # Проверяем, поддерживает ли клиент Outlines
                if hasattr(client, '_structured_stream_outlines'):
                    return await client._structured_stream_outlines(
                        messages=messages,
                        response_model=response_model,
                        **kwargs
                    )
                elif hasattr(client, 'structured_completion'):
                    # Альтернативный метод
                    return await client.structured_completion(
                        messages=messages,
                        response_model=response_model,
                        stream=True,  # Outlines режим через streaming
                        **kwargs
                    )
                else:
                    # Последний fallback - через обычный chat с постобработкой
                    logger.info(
                        "Использование fallback через обычный chat completion")
                    return await self._structured_fallback(messages, response_model, **kwargs)

            except Exception as outlines_error:
                logger.error(
                    f"Outlines режим также не удался: {outlines_error}")

                # Финальный fallback
                try:
                    return await self._structured_fallback(messages, response_model, **kwargs)
                except Exception as fallback_error:
                    # Объединяем ошибки для лучшей диагностики
                    error_msg = (
                        f"Все методы structured output не удались:\n"
                        f"1. Нативный: {native_error}\n"
                        f"2. Outlines: {outlines_error}\n"
                        f"3. Fallback: {fallback_error}"
                    )
                    raise KrakenError(error_msg)

    # Специализированные методы

    async def reasoning_completion(
        self,
        messages: List[Dict[str, str]],
        problem_type: str = "general",
        **kwargs
    ) -> Any:
        """Завершение с рассуждениями"""
        if not self._initialized:
            await self._initialize_clients()

        if UniversalCapability.REASONING not in self.universal_config.capabilities:
            raise ValidationError("Reasoning не настроен в конфигурации")

        client = self._get_optimal_client('reasoning', **kwargs)

        if not client or not hasattr(client, 'reasoning_completion'):
            # Fallback на обычный chat completion с reasoning промптом
            reasoning_prompt = {
                "role": "system",
                "content": "Решай задачи пошагово, объясняя каждый шаг своих рассуждений."
            }
            enhanced_messages = [reasoning_prompt] + messages
            return await self.chat_completion(enhanced_messages, **kwargs)

        return await client.reasoning_completion(
            messages=messages,
            problem_type=problem_type,
            **kwargs
        )

    async def vision_completion(
        self,
        text_prompt: str,
        images: Union[str, Path, List[Union[str, Path]]],
        **kwargs
    ) -> Any:
        """Анализ изображений"""
        if not self._initialized:
            await self._initialize_clients()

        if UniversalCapability.VISION not in self.universal_config.capabilities:
            raise ValidationError("Vision не настроен в конфигурации")

        client = self._get_optimal_client('vision', **kwargs)

        if not client or not hasattr(client, 'vision_completion'):
            raise KrakenError("Нет доступного клиента для vision")

        return await client.vision_completion(
            text_prompt=text_prompt,
            images=images,
            **kwargs
        )

    async def get_embeddings(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> Any:
        """Получение векторных представлений"""
        if not self._initialized:
            await self._initialize_clients()

        if UniversalCapability.EMBEDDINGS not in self.universal_config.capabilities:
            raise ValidationError("Embeddings не настроены в конфигурации")

        client = self._get_optimal_client('embeddings', **kwargs)

        if not client or not hasattr(client, 'get_embeddings'):
            raise KrakenError("Нет доступного клиента для embeddings")

        return await client.get_embeddings(texts, **kwargs)

    async def similarity_search(
        self,
        query_text: str,
        candidate_texts: List[str],
        top_k: Optional[int] = None,
        **kwargs
    ) -> Any:
        """Поиск по сходству"""
        if not self._initialized:
            await self._initialize_clients()

        if UniversalCapability.SIMILARITY_SEARCH not in self.universal_config.capabilities:
            raise ValidationError(
                "Similarity search не настроен в конфигурации")

        client = self._get_optimal_client('similarity', **kwargs)

        if not client or not hasattr(client, 'similarity_search'):
            raise KrakenError("Нет доступного клиента для similarity search")

        return await client.similarity_search(
            query_text=query_text,
            candidate_texts=candidate_texts,
            top_k=top_k,
            **kwargs
        )

    async def speech_to_text(
        self,
        audio_file: Union[str, Path],
        **kwargs
    ) -> Any:
        """Преобразование речи в текст"""
        if not self._initialized:
            await self._initialize_clients()

        if UniversalCapability.STT not in self.universal_config.capabilities:
            raise ValidationError("Speech-to-text не настроен в конфигурации")

        client = self._get_optimal_client('stt', **kwargs)

        if not client or not hasattr(client, 'speech_to_text'):
            raise KrakenError("Нет доступного клиента для STT")

        return await client.speech_to_text(audio_file, **kwargs)

    async def text_to_speech(
        self,
        text: str,
        **kwargs
    ) -> Any:
        """Преобразование текста в речь"""
        if not self._initialized:
            await self._initialize_clients()

        if UniversalCapability.TTS not in self.universal_config.capabilities:
            raise ValidationError("Text-to-speech не настроен в конфигурации")

        client = self._get_optimal_client('tts', **kwargs)

        if not client or not hasattr(client, 'text_to_speech'):
            raise KrakenError("Нет доступного клиента для TTS")

        return await client.text_to_speech(text, **kwargs)

    async def text_completion(
        self,
        prompt: str,
        **kwargs
    ) -> Any:
        """Legacy text completion"""
        if not self._initialized:
            await self._initialize_clients()

        if UniversalCapability.COMPLETION_LEGACY not in self.universal_config.capabilities:
            raise ValidationError(
                "Legacy completion не настроен в конфигурации")

        client = self._get_optimal_client('completion', **kwargs)

        if not client or not hasattr(client, 'text_completion'):
            raise KrakenError("Нет доступного клиента для text completion")

        return await client.text_completion(prompt, **kwargs)

    # Методы управления функциями и инструментами

    def register_function(
        self,
        name: str,
        function,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None
    ) -> None:
        """Регистрация функции во всех поддерживающих клиентах"""
        for client in self._clients.values():
            if hasattr(client, 'register_function'):
                try:
                    client.register_function(
                        name, function, description, parameters)
                except Exception as e:
                    logger.warning(
                        f"Не удалось зарегистрировать функцию {name} в {type(client).__name__}: {e}")

    def register_tool(
        self,
        name: str,
        tool,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None
    ) -> None:
        """Регистрация инструмента во всех поддерживающих клиентах"""
        for client in self._clients.values():
            if hasattr(client, 'register_tool'):
                try:
                    client.register_tool(name, tool, description, parameters)
                except Exception as e:
                    logger.warning(
                        f"Не удалось зарегистрировать инструмент {name} в {type(client).__name__}: {e}")

    # Информационные методы

    def get_available_capabilities(self) -> List[str]:
        """Получение списка доступных возможностей"""
        return [cap.value for cap in self.universal_config.capabilities]

    def get_active_clients(self) -> List[str]:
        """Получение списка активных клиентов"""
        return list(self._clients.keys())

    def get_client_info(self) -> Dict[str, Any]:
        """Получение информации о клиенте"""
        return {
            "capabilities": self.get_available_capabilities(),
            "active_clients": self.get_active_clients(),
            "config": {
                "auto_fallback": self.universal_config.auto_fallback,
                "prefer_streaming": self.universal_config.prefer_streaming,
                "enable_caching": self.universal_config.enable_caching,
                "concurrent_requests": self.universal_config.concurrent_requests,
            },
            "initialized": self._initialized
        }

    async def test_capabilities(self) -> Dict[str, bool]:
        """Тестирование доступных возможностей"""
        if not self._initialized:
            await self._initialize_clients()

        results = {}

        # Тест базового chat completion
        if UniversalCapability.CHAT_COMPLETION in self.universal_config.capabilities:
            try:
                response = await self.chat_completion([
                    {"role": "user", "content": "Тест"}
                ], max_tokens=5)
                results["chat_completion"] = bool(response)
            except Exception:
                results["chat_completion"] = False

        # Тест streaming
        if UniversalCapability.STREAMING in self.universal_config.capabilities:
            try:
                chunks = []
                async for chunk in self.chat_completion_stream([
                    {"role": "user", "content": "Тест"}
                ]):
                    chunks.append(chunk)
                    if len(chunks) >= 2:  # Достаточно для теста
                        break
                results["streaming"] = len(chunks) > 0
            except Exception:
                results["streaming"] = False

        # Тест structured output
        if UniversalCapability.STRUCTURED_OUTPUT in self.universal_config.capabilities:
            try:
                class TestModel(BaseModel):
                    test: str

                response = await self.chat_completion_structured([
                    {"role": "user", "content": "Создай JSON с полем test='ok'"}
                ], response_model=TestModel)
                results["structured_output"] = isinstance(response, TestModel)
            except Exception:
                results["structured_output"] = False

        # Добавляем результаты для других возможностей
        for capability in self.universal_config.capabilities:
            if capability.value not in results:
                results[capability.value] = capability.value in [
                    client.__class__.__name__.lower() for client in self._clients.values()]

        return results


# Удобные функции для создания клиентов

def create_universal_client(
    config: Optional[LLMConfig] = None,
    capabilities: Optional[Set[UniversalCapability]] = None,
    **kwargs
) -> UniversalLLMClient:
    """
    Создание универсального клиента с простой настройкой

    Args:
        config: Конфигурация LLM (если None, создается из переменных окружения)
        capabilities: Набор возможностей (если None, используется базовый набор)
        **kwargs: Дополнительные параметры для UniversalClientConfig

    Returns:
        Настроенный UniversalLLMClient
    """
    if config is None:
        config = LLMConfig()

    if capabilities is None:
        capabilities = {
            UniversalCapability.CHAT_COMPLETION,
            UniversalCapability.STREAMING
        }

    universal_config = UniversalClientConfig(
        capabilities=capabilities,
        **kwargs
    )

    return UniversalLLMClient(config, universal_config)


def create_universal_client_from_report(
    capabilities_report: Dict[str, Any],
    config: Optional[LLMConfig] = None,
    model_name: Optional[str] = None
) -> UniversalLLMClient:
    """
    Создание универсального клиента на основе отчета анализатора

    Args:
        capabilities_report: Отчет от ModelCapabilitiesAnalyzer
        config: Конфигурация LLM (если None, создается из переменных окружения)
        model_name: Имя модели из отчета

    Returns:
        Настроенный UniversalLLMClient с оптимальными возможностями
    """
    if config is None:
        config = LLMConfig()

    universal_config = UniversalClientConfig.from_capabilities_report(
        capabilities_report, model_name
    )

    return UniversalLLMClient(config, universal_config)


# Предустановленные конфигурации

def create_basic_client(config: Optional[LLMConfig] = None) -> UniversalLLMClient:
    """Базовый клиент для простых задач"""
    if config is None:
        config = LLMConfig()

    return UniversalLLMClient(config, UniversalClientConfig.basic())


def create_advanced_client(config: Optional[LLMConfig] = None) -> UniversalLLMClient:
    """Продвинутый клиент для сложных задач"""
    if config is None:
        config = LLMConfig()

    return UniversalLLMClient(config, UniversalClientConfig.advanced())


def create_full_client(config: Optional[LLMConfig] = None) -> UniversalLLMClient:
    """Полнофункциональный клиент со всеми возможностями"""
    if config is None:
        config = LLMConfig()

    return UniversalLLMClient(config, UniversalClientConfig.all_capabilities())
