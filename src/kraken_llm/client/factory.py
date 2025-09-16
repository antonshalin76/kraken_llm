"""
Фабрика клиентов для Kraken LLM фреймворка.

Этот модуль предоставляет удобные функции для создания различных типов
LLM клиентов с автоматическим выбором подходящего клиента на основе параметров.
"""

from typing import Optional, Type, Dict, Any, List
import logging
import os

from ..config.settings import LLMConfig
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
from .universal import UniversalLLMClient

logger = logging.getLogger(__name__)


class ClientFactory:
    """
    Фабрика для создания LLM клиентов.

    Предоставляет методы для создания различных типов клиентов
    с автоматическим выбором подходящего типа на основе параметров.
    """

    # Реестр доступных типов клиентов
    _client_registry = {
        'standard': StandardLLMClient,
        'streaming': StreamingLLMClient,
        'structured': StructuredLLMClient,
        'reasoning': ReasoningLLMClient,
        'multimodal': MultimodalLLMClient,
        'adaptive': AdaptiveLLMClient,
        'asr': ASRClient,
        'embeddings': EmbeddingsClient,
        'completion': CompletionLLMClient,
        'universal': UniversalLLMClient,
    }

    @classmethod
    def create_client(
        cls,
        client_type: Optional[str] = None,
        config: Optional[LLMConfig] = None,
        **kwargs
    ) -> BaseLLMClient:
        """
        Создать LLM клиент указанного типа.

        Args:
            client_type: Тип клиента ('standard', 'streaming', 'structured', etc.)
            config: Конфигурация клиента (если не указана, берётся из .env по профилю клиента)
            **kwargs: Дополнительные параметры для конфигурации

        Returns:
            BaseLLMClient: Экземпляр LLM клиента

        Raises:
            ValueError: При неизвестном типе клиента

        Examples:
            >>> # Создание стандартного клиента
            >>> client = ClientFactory.create_client('standard')

            >>> # Создание клиента с кастомной конфигурацией
            >>> config = LLMConfig(endpoint="http://localhost:8080")
            >>> client = ClientFactory.create_client('streaming', config)

            >>> # Создание клиента с параметрами
            >>> client = ClientFactory.create_client(
            ...     'structured',
            ...     endpoint="http://localhost:8080",
            ...     temperature=0.7
            ... )
        """
        # Создаем конфигурацию если не передана: подбираем по типу клиента из .env
        # Выделяем специализированные конфиги, чтобы не попадали в LLMConfig
        reasoning_cfg = kwargs.pop('reasoning_config', None)

        if config is None:
            resolved_type = client_type or cls._auto_detect_client_type(LLMConfig(**kwargs), **kwargs)
            config = cls._create_config_for_client_type(resolved_type, kwargs)
        else:
            # Обновляем конфигурацию переданными параметрами
            if kwargs:
                config_dict = config.model_dump()
                config_dict.update({k: v for k, v in kwargs.items() if k not in {'reasoning_config'}})
                config = LLMConfig(**config_dict)

        # Автоматический выбор типа клиента если не указан
        if client_type is None:
            client_type = cls._auto_detect_client_type(config, **kwargs)

        # Получаем класс клиента
        if client_type not in cls._client_registry:
            available_types = ', '.join(cls._client_registry.keys())
            raise ValueError(
                f"Неизвестный тип клиента: {client_type}. "
                f"Доступные типы: {available_types}"
            )

        client_class = cls._client_registry[client_type]

        logger.info(
            f"Создание клиента типа {client_type} с endpoint: {config.endpoint}")

        # Создаем и возвращаем клиент
        if client_class is ReasoningLLMClient:
            return client_class(config, reasoning_cfg)
        return client_class(config)

    @classmethod
    def _auto_detect_client_type(cls, config: LLMConfig, **kwargs) -> str:
        """
        Автоматическое определение типа клиента на основе параметров.

        Args:
            config: Конфигурация клиента
            **kwargs: Дополнительные параметры

        Returns:
            str: Рекомендуемый тип клиента
        """
        # Проверяем специфичные параметры в kwargs
        if 'response_model' in kwargs or 'structured_output' in kwargs:
            return 'structured'

        if 'stream' in kwargs and kwargs['stream']:
            return 'streaming'

        if 'reasoning_mode' in kwargs or 'chain_of_thought' in kwargs:
            return 'reasoning'

        if 'media_input' in kwargs or 'vision' in kwargs or 'audio' in kwargs:
            return 'multimodal'

        if 'audio_file' in kwargs or 'speech_recognition' in kwargs:
            return 'asr'

        if 'embeddings' in kwargs or 'vector_search' in kwargs:
            return 'embeddings'

        # Проверяем конфигурацию
        if hasattr(config, 'stream') and config.stream:
            return 'streaming'

        if hasattr(config, 'outlines_so_mode') and config.outlines_so_mode:
            return 'structured'

        # По умолчанию возвращаем adaptive клиент
        return 'adaptive'

    @classmethod
    def create_standard_client(cls, config: Optional[LLMConfig] = None, **kwargs) -> StandardLLMClient:
        """
        Создать стандартный LLM клиент.

        Args:
            config: Конфигурация клиента
            **kwargs: Дополнительные параметры

        Returns:
            StandardLLMClient: Стандартный клиент
        """
        return cls.create_client('standard', config, **kwargs)

    @classmethod
    def create_streaming_client(cls, config: Optional[LLMConfig] = None, **kwargs) -> StreamingLLMClient:
        """
        Создать потоковый LLM клиент.

        Args:
            config: Конфигурация клиента
            **kwargs: Дополнительные параметры

        Returns:
            StreamingLLMClient: Потоковый клиент
        """
        return cls.create_client('streaming', config, **kwargs)

    @classmethod
    def create_structured_client(cls, config: Optional[LLMConfig] = None, **kwargs) -> StructuredLLMClient:
        """
        Создать клиент для структурированного вывода.

        Args:
            config: Конфигурация клиента
            **kwargs: Дополнительные параметры

        Returns:
            StructuredLLMClient: Клиент для структурированного вывода
        """
        return cls.create_client('structured', config, **kwargs)

    @classmethod
    def create_reasoning_client(cls, config: Optional[LLMConfig] = None, **kwargs) -> ReasoningLLMClient:
        """
        Создать клиент для рассуждающих моделей.

        Args:
            config: Конфигурация клиента
            **kwargs: Дополнительные параметры

        Returns:
            ReasoningLLMClient: Клиент для рассуждающих моделей
        """
        return cls.create_client('reasoning', config, **kwargs)

    @classmethod
    def create_multimodal_client(cls, config: Optional[LLMConfig] = None, **kwargs) -> MultimodalLLMClient:
        """
        Создать мультимодальный LLM клиент.

        Args:
            config: Конфигурация клиента
            **kwargs: Дополнительные параметры

        Returns:
            MultimodalLLMClient: Мультимодальный клиент
        """
        return cls.create_client('multimodal', config, **kwargs)

    @classmethod
    def create_adaptive_client(cls, config: Optional[LLMConfig] = None, **kwargs) -> AdaptiveLLMClient:
        """
        Создать адаптивный LLM клиент.

        Args:
            config: Конфигурация клиента
            **kwargs: Дополнительные параметры

        Returns:
            AdaptiveLLMClient: Адаптивный клиент
        """
        return cls.create_client('adaptive', config, **kwargs)

    @classmethod
    def create_asr_client(cls, config: Optional[LLMConfig] = None, **kwargs) -> ASRClient:
        """
        Создать ASR клиент для работы с речью.

        Args:
            config: Конфигурация клиента
            **kwargs: Дополнительные параметры

        Returns:
            ASRClient: ASR клиент
        """
        return cls.create_client('asr', config, **kwargs)

    @classmethod
    def create_embeddings_client(cls, config: Optional[LLMConfig] = None, **kwargs) -> EmbeddingsClient:
        """
        Создать клиент для работы с векторными представлениями.

        Args:
            config: Конфигурация клиента
            **kwargs: Дополнительные параметры

        Returns:
            EmbeddingsClient: Клиент для embeddings
        """
        return cls.create_client('embeddings', config, **kwargs)

    @classmethod
    def create_completion_client(cls, config: Optional[LLMConfig] = None, **kwargs) -> CompletionLLMClient:
        """
        Создать completion клиент для работы с /v1/completions endpoint.

        Args:
            config: Конфигурация клиента
            **kwargs: Дополнительные параметры

        Returns:
            CompletionLLMClient: Completion клиент
        """
        return cls.create_client('completion', config, **kwargs)

    @classmethod
    def get_available_client_types(cls) -> Dict[str, Type[BaseLLMClient]]:
        """
        Получить список доступных типов клиентов.

        Returns:
            Dict[str, Type[BaseLLMClient]]: Словарь типов клиентов
        """
        return cls._client_registry.copy()

    @classmethod
    def register_client_type(cls, name: str, client_class: Type[BaseLLMClient]) -> None:
        """
        Зарегистрировать новый тип клиента.

        Args:
            name: Имя типа клиента
            client_class: Класс клиента

        Raises:
            ValueError: Если тип уже зарегистрирован
            ValueError: Если класс не наследуется от BaseLLMClient
        """
        if name in cls._client_registry:
            raise ValueError(f"Тип клиента '{name}' уже зарегистрирован")

        if not issubclass(client_class, BaseLLMClient):
            raise ValueError(
                f"Класс клиента должен наследоваться от BaseLLMClient, "
                f"получен: {client_class}"
            )

        cls._client_registry[name] = client_class
        logger.info(f"Зарегистрирован новый тип клиента: {name}")

    # Вспомогательные методы
    @classmethod
    def _create_config_for_client_type(cls, client_type: str, base_kwargs: Dict[str, Any]) -> LLMConfig:
        """
        Создаёт LLMConfig, подбирая переменные окружения под конкретный тип клиента.
        Предпочитает профильные префиксы (например, EMBEDDING_*, LLM_REASONING_*),
        а затем падает обратно на универсальные LLM_*.
        """
        # Маппинг профилей по типу клиента (в порядке приоритета)
        profile_map: Dict[str, List[str]] = {
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

        # Попробуем собрать endpoint/token/model по профилям
        for prefix in profiles:
            endpoint = os.getenv(f"{prefix}_ENDPOINT")
            model = os.getenv(f"{prefix}_MODEL")
            # Токен может называться по-разному
            token = os.getenv(f"{prefix}_TOKEN") or os.getenv(f"{prefix}_API_KEY") or os.getenv(f"{prefix}_KEY")

            if endpoint and model:
                # Собираем kwargs для LLMConfig
                cfg_kwargs = {
                    **base_kwargs,
                    'endpoint': endpoint,
                    'model': model,
                }
                if token:
                    # LLMConfig примет api_key
                    cfg_kwargs['api_key'] = token
                return LLMConfig(**cfg_kwargs)

        # Ничего профильного не нашли — используем универсальную конфигурацию
        return LLMConfig(**base_kwargs)
        if not issubclass(client_class, BaseLLMClient):
            raise ValueError(
                f"Класс клиента должен наследоваться от BaseLLMClient, "
                f"получен: {client_class}"
            )

        cls._client_registry[name] = client_class
        logger.info(f"Зарегистрирован новый тип клиента: {name}")


# Удобные функции для создания клиентов
def create_client(
    client_type: Optional[str] = None,
    config: Optional[LLMConfig] = None,
    **kwargs
) -> BaseLLMClient:
    """
    Создать LLM клиент (удобная функция).

    Args:
        client_type: Тип клиента
        config: Конфигурация клиента
        **kwargs: Дополнительные параметры

    Returns:
        BaseLLMClient: Экземпляр LLM клиента

    Examples:
        >>> # Простое создание клиента
        >>> client = create_client()

        >>> # Создание с параметрами
        >>> client = create_client(
        ...     client_type='streaming',
        ...     endpoint="http://localhost:8080",
        ...     temperature=0.7
        ... )
    """
    return ClientFactory.create_client(client_type, config, **kwargs)


def create_standard_client(config: Optional[LLMConfig] = None, **kwargs) -> StandardLLMClient:
    """Создать стандартный клиент (удобная функция)."""
    return ClientFactory.create_standard_client(config, **kwargs)


def create_streaming_client(config: Optional[LLMConfig] = None, **kwargs) -> StreamingLLMClient:
    """Создать потоковый клиент (удобная функция)."""
    return ClientFactory.create_streaming_client(config, **kwargs)


def create_structured_client(config: Optional[LLMConfig] = None, **kwargs) -> StructuredLLMClient:
    """Создать клиент для структурированного вывода (удобная функция)."""
    return ClientFactory.create_structured_client(config, **kwargs)


def create_reasoning_client(config: Optional[LLMConfig] = None, **kwargs) -> ReasoningLLMClient:
    """Создать клиент для рассуждающих моделей (удобная функция)."""
    return ClientFactory.create_reasoning_client(config, **kwargs)


def create_multimodal_client(config: Optional[LLMConfig] = None, **kwargs) -> MultimodalLLMClient:
    """Создать мультимодальный клиент (удобная функция)."""
    return ClientFactory.create_multimodal_client(config, **kwargs)


def create_adaptive_client(config: Optional[LLMConfig] = None, **kwargs) -> AdaptiveLLMClient:
    """Создать адаптивный клиент (удобная функция)."""
    return ClientFactory.create_adaptive_client(config, **kwargs)


def create_asr_client(config: Optional[LLMConfig] = None, **kwargs) -> ASRClient:
    """Создать ASR клиент (удобная функция)."""
    return ClientFactory.create_asr_client(config, **kwargs)


def create_embeddings_client(config: Optional[LLMConfig] = None, **kwargs) -> EmbeddingsClient:
    """Создать клиент для embeddings (удобная функция)."""
    return ClientFactory.create_embeddings_client(config, **kwargs)


def create_completion_client(config: Optional[LLMConfig] = None, **kwargs) -> CompletionLLMClient:
    """Создать completion клиент (удобная функция)."""
    return ClientFactory.create_completion_client(config, **kwargs)

    @classmethod
    def create_universal_client(cls, config: Optional[LLMConfig] = None, **kwargs) -> UniversalLLMClient:
        """
        Создать универсальный LLM клиент.

        Args:
            config: Конфигурация клиента
            **kwargs: Дополнительные параметры

        Returns:
            UniversalLLMClient: Универсальный клиент
        """
        return cls.create_client('universal', config, **kwargs)


def create_universal_client(config: Optional[LLMConfig] = None, **kwargs) -> UniversalLLMClient:
    """Создать универсальный клиент (удобная функция)."""
    return ClientFactory.create_universal_client(config, **kwargs)