"""
Реализации клиентов Kraken LLM

Этот модуль содержит все реализации LLM клиентов включая:
- BaseLLMClient: Абстрактный базовый класс
- StandardLLMClient: Стандартные chat completions
- StreamingLLMClient: Потоковые ответы
- StructuredLLMClient: Структурированный вывод с валидацией
- ReasoningLLMClient: Рассуждающие модели
- MultimodalLLMClient: Мультимодальные возможности
- AdaptiveLLMClient: Адаптивный клиент
- ASRClient: Автоматическое распознавание речи
- EmbeddingsClient: Векторные представления
- UniversalLLMClient: Универсальный клиент, объединяющий все возможности
- ClientFactory: Фабрика для создания клиентов
"""

from .base import BaseLLMClient
from .standard import StandardLLMClient
from .streaming import StreamingLLMClient
from .structured import StructuredLLMClient
from .reasoning import ReasoningLLMClient
from .multimodal import MultimodalLLMClient
from .adaptive import AdaptiveLLMClient
from .asr import ASRClient
from .embeddings import EmbeddingsLLMClient as EmbeddingsClient
from .completion import CompletionLLMClient
from .universal import (
    UniversalLLMClient,
    UniversalClientConfig,
    UniversalCapability,
    create_universal_client,
    create_universal_client_from_report,
    create_basic_client,
    create_advanced_client,
    create_full_client,
)
from .factory import (
    ClientFactory,
    create_client,
    create_standard_client,
    create_streaming_client,
    create_structured_client,
    create_reasoning_client,
    create_multimodal_client,
    create_adaptive_client,
    create_asr_client,
    create_embeddings_client,
    create_completion_client,
)

__all__ = [
    # Базовые классы
    "BaseLLMClient",
    
    # Основные клиенты
    "StandardLLMClient", 
    "StreamingLLMClient",
    "StructuredLLMClient",
    "CompletionLLMClient",
    
    # Расширенные клиенты
    "ReasoningLLMClient",
    "MultimodalLLMClient",
    "AdaptiveLLMClient",
    "ASRClient",
    "EmbeddingsClient",
    
    # Универсальный клиент
    "UniversalLLMClient",
    "UniversalClientConfig",
    "UniversalCapability",
    
    # Фабрика клиентов
    "ClientFactory",
    
    # Удобные функции создания
    "create_client",
    "create_standard_client",
    "create_streaming_client",
    "create_structured_client",
    "create_reasoning_client",
    "create_multimodal_client",
    "create_adaptive_client",
    "create_asr_client",
    "create_embeddings_client",
    "create_completion_client",
    
    # Функции создания универсального клиента
    "create_universal_client",
    "create_universal_client_from_report",
    "create_basic_client",
    "create_advanced_client",
    "create_full_client",
]