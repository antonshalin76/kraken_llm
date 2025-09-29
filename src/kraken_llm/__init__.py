"""
Kraken LLM Framework

Универсальный LLM фреймворк с полной поддержкой OpenAI API, включая потоковую передачу,
структурированный вывод, вызов функций и вызов инструментов.
"""

from .config.settings import LLMConfig
from .client.base import BaseLLMClient
from .client.standard import StandardLLMClient
from .client.streaming import StreamingLLMClient
from .client.structured import StructuredLLMClient
from .client.reasoning import ReasoningLLMClient, ReasoningConfig
from .client.multimodal import MultimodalLLMClient, MultimodalConfig
from .client.adaptive import AdaptiveLLMClient, AdaptiveConfig
from .client.asr import ASRClient
from .client.embeddings import EmbeddingsLLMClient as EmbeddingsClient
from .client.universal import (
    UniversalLLMClient,
    UniversalClientConfig,
    UniversalCapability,
    create_universal_client,
    create_universal_client_from_report,
    create_basic_client,
    create_advanced_client,
    create_full_client,
)
from .client.factory import (
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

# Поддержка вызова функций и инструментов
from .tools import (
    FunctionRegistry,
    ToolRegistry,
    FunctionToolExecutor,
    ExecutionContext,
    ExecutionResult,
    register_function,
    register_tool,
    default_function_registry,
    default_tool_registry,
    default_executor
)

# Псевдонимы для совместимости
LLMClient = StandardLLMClient  # Основной клиент для демо-приложения

# Информация о версии
from importlib.metadata import PackageNotFoundError, version as _version
try:
    __version__ = _version("kraken_llm")
except PackageNotFoundError:
    __version__ = "0.0.0"
__author__ = "Anton Shalin"
__email__ = "anton.shalin@gmail.com"

# Экспорт публичного API
# Утилиты визуализации уверенности (re-export для удобства)
from .utils.color import (
    colorize_text_ansi,
    colorize_text_html,
    colorize_tokens_ansi,
    colorize_tokens_html,
    get_confidence_legend_ansi,
    get_confidence_legend_html,
)

__all__ = [
    # Основные клиенты
    "BaseLLMClient",
    "StandardLLMClient", 
    "StreamingLLMClient",
    "StructuredLLMClient",
    "ReasoningLLMClient",
    "MultimodalLLMClient", 
    "AdaptiveLLMClient",
    "ASRClient",
    "EmbeddingsClient",
    "LLMClient",  # Compatibility alias
    "LLMConfig",
    
    # Универсальный клиент
    "UniversalLLMClient",
    "UniversalClientConfig",
    "UniversalCapability",
    
    # Конфигурации
    "ReasoningConfig",
    "MultimodalConfig",
    "AdaptiveConfig",
    
    # Фабрика клиентов
    "ClientFactory",
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
    
    # Вызов функций и инструментов
    "FunctionRegistry",
    "ToolRegistry",
    "FunctionToolExecutor",
    "ExecutionContext",
    "ExecutionResult",
    "register_function",
    "register_tool",
    "default_function_registry",
    "default_tool_registry",
    "default_executor",

    # Утилиты визуализации уверенности
    "colorize_text_ansi",
    "colorize_text_html",
    "colorize_tokens_ansi",
    "colorize_tokens_html",
    "get_confidence_legend_ansi",
    "get_confidence_legend_html",
    
    # Версия
    "__version__",
]
