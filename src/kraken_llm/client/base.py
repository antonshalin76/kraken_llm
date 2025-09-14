"""
Базовый абстрактный LLM клиент для Kraken фреймворка.

Этот модуль содержит абстрактный базовый класс для всех LLM клиентов,
обеспечивающий единый интерфейс и общую функциональность для работы с
языковыми моделями через AsyncOpenAI и Outlines.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel

from ..config.settings import LLMConfig
from ..exceptions.api import APIError
from ..exceptions.network import NetworkError, TimeoutError
from ..exceptions.validation import ValidationError
from ..utils.logging import get_logger

logger = get_logger(__name__)


class BaseLLMClient(ABC):
    """
    Базовый абстрактный класс для всех LLM клиентов.

    Предоставляет общую функциональность для работы с AsyncOpenAI и Outlines,
    включая инициализацию клиентов, обработку параметров и утилитарные методы.

    Все конкретные реализации клиентов должны наследоваться от этого класса
    и реализовывать абстрактные методы для различных типов операций.

    Attributes:
        config: Конфигурация LLM клиента
        openai_client: AsyncOpenAI клиент для API вызовов
        outlines_model: Outlines модель для structured output
    """

    def __init__(self, config: LLMConfig):
        """
        Инициализация базового LLM клиента.

        Args:
            config: Конфигурация клиента с параметрами подключения и генерации

        Raises:
            NetworkError: При ошибках создания HTTP клиента
            ValidationError: При некорректной конфигурации
        """
        logger.info(f"Инициализация BaseLLMClient с endpoint: {config.endpoint}")

        self.config = config
        self._validate_config()

        # Создание AsyncOpenAI клиента
        self.openai_client = self._create_async_openai_client()

        # Создание Outlines модели (ленивая инициализация)
        self._outlines_model = None

        logger.debug(f"BaseLLMClient инициализирован успешно")

    def _validate_config(self) -> None:
        """
        Валидация конфигурации клиента.

        Raises:
            ValidationError: При некорректных параметрах конфигурации
        """
        if not self.config.endpoint:
            raise ValidationError("Endpoint не может быть пустым")

        if not self.config.model:
            raise ValidationError("Model не может быть пустым")

        if self.config.max_tokens <= 0:
            raise ValidationError("max_tokens должен быть больше 0")

        if not (0.0 <= self.config.temperature <= 2.0):
            raise ValidationError("temperature должен быть между 0.0 и 2.0")

    def _build_api_url(self, endpoint_type: str = "chat_completions") -> str:
        """
        Построение полного URL для API endpoint.

        Args:
            endpoint_type: Тип endpoint'а ('chat_completions', 'completions', 'embeddings', 'models')

        Returns:
            Полный URL для API вызова
        """
        base_url = self.config.endpoint.rstrip("/")

        if self.config.api_mode == "direct":
            # Используем endpoint как есть
            return base_url
        elif self.config.api_mode == "custom":
            # Используем кастомные пути
            path_mapping = {
                "chat_completions": self.config.chat_completions_path,
                "completions": self.config.completions_path,
                "embeddings": self.config.embeddings_path,
                "models": self.config.models_path,
            }
            path = path_mapping.get(endpoint_type, self.config.chat_completions_path)
            return f"{base_url}{path}"
        else:  # openai_compatible (default)
            # Используем стандартные OpenAI пути
            path_mapping = {
                "chat_completions": f"/{self.config.api_version}/chat/completions",
                "completions": f"/{self.config.api_version}/completions",
                "embeddings": f"/{self.config.api_version}/embeddings",
                "models": f"/{self.config.api_version}/models",
            }
            path = path_mapping.get(
                endpoint_type, f"/{self.config.api_version}/chat/completions"
            )
            return f"{base_url}{path}"

    def _get_openai_base_url(self) -> str:
        """
        Получение базового URL для AsyncOpenAI клиента.

        Returns:
            Базовый URL без конкретного endpoint'а
        """
        base_url = self.config.endpoint.rstrip("/")

        if self.config.api_mode == "direct":
            return base_url
        else:
            # Для OpenAI-совместимых API возвращаем базовый URL с версией
            return f"{base_url}/{self.config.api_version}"

    def _create_async_openai_client(self) -> AsyncOpenAI:
        """
        Создание AsyncOpenAI клиента с настройками таймаутов.

        Returns:
            Настроенный AsyncOpenAI клиент

        Raises:
            NetworkError: При ошибках создания HTTP клиента
        """
        try:
            # Создание timeout конфигурации
            timeout = httpx.Timeout(
                connect=self.config.connect_timeout,
                read=self.config.read_timeout,
                write=self.config.write_timeout,
                pool=None,  # Добавляем pool параметр
            )

            # Получаем правильный base_url для AsyncOpenAI
            base_url = self._get_openai_base_url()

            # Создание AsyncOpenAI клиента
            # Если api_key не задан, используем фиктивный ключ для совместимости
            api_key = self.config.api_key or "dummy_key_for_local_llm"

            client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=self.config.max_retries,
            )

            logger.debug(
                f"AsyncOpenAI клиент создан с base_url: {base_url} (режим: {self.config.api_mode})"
            )
            return client

        except Exception as e:
            logger.error(f"Ошибка создания AsyncOpenAI клиента: {e}")
            raise NetworkError(
                f"Не удалось создать AsyncOpenAI клиент: {e}",
                context={"endpoint": self.config.endpoint},
                original_error=e,
            )

    @property
    def outlines_model(self):
        """
        Ленивое создание Outlines модели для structured output.

        Returns:
            Outlines модель для structured output

        Raises:
            NetworkError: При ошибках создания Outlines модели
        """
        if self._outlines_model is None:
            try:
                logger.debug(
                    f"Создание Outlines модели для модели: {self.config.model}"
                )
                # Ленивый импорт outlines для избежания проблем с numpy
                import outlines

                # Создаем Outlines модель из AsyncOpenAI клиента
                self._outlines_model = outlines.from_openai(
                    self.openai_client, self.config.model
                )
                logger.debug("Outlines модель создана успешно")
            except Exception as e:
                logger.error(f"Ошибка создания Outlines модели: {e}")
                raise NetworkError(
                    f"Не удалось создать Outlines модель: {e}",
                    context={"model": self.config.model},
                    original_error=e,
                )

        return self._outlines_model

    def _messages_to_outlines_chat(self, messages: List[Dict[str, str]]):
        """
        Конвертация сообщений в формат Outlines Chat.

        Args:
            messages: Список сообщений в формате OpenAI

        Returns:
            Outlines Chat объект
        """
        try:
            # Ленивый импорт outlines
            import outlines

            # Конвертируем сообщения в формат Outlines
            outlines_messages = []
            for msg in messages:
                outlines_messages.append(
                    {"role": msg["role"], "content": msg["content"]}
                )

            # Создаем Chat объект
            chat = outlines.inputs.Chat(outlines_messages)
            logger.debug(f"Конвертировано {len(messages)} сообщений в Outlines Chat")
            return chat

        except Exception as e:
            logger.error(f"Ошибка конвертации сообщений в Outlines Chat: {e}")
            raise ValidationError(
                f"Не удалось конвертировать сообщения в Outlines формат: {e}",
                original_error=e,
            )

    def _prepare_openai_params(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Подготовка параметров для вызова AsyncOpenAI API.

        Args:
            messages: Список сообщений для чата
            model: Переопределение модели (опционально)
            temperature: Переопределение температуры (опционально)
            max_tokens: Переопределение максимального количества токенов (опционально)
            stream: Переопределение режима потока (опционально)
            **kwargs: Дополнительные параметры

        Returns:
            Словарь параметров для AsyncOpenAI API
        """
        # Базовые параметры из конфигурации
        params = self.config.to_openai_params()

        # Обязательные параметры
        params.update(
            {
                "messages": messages,
                "model": model or self.config.model,
            }
        )

        # Переопределения параметров
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if stream is not None:
            params["stream"] = stream

        # Дополнительные параметры
        params.update(kwargs)

        # Удаление None значений
        params = {k: v for k, v in params.items() if v is not None}

        logger.debug(f"Подготовлены параметры OpenAI: {list(params.keys())}")
        return params

    def _messages_to_outlines_chat(self, messages: List[Dict[str, str]]):
        """
        Конвертация сообщений в формат Outlines Chat.

        Args:
            messages: Список сообщений в формате OpenAI

        Returns:
            Outlines Chat объект
        """
        try:
            # Ленивый импорт для избежания проблем с numpy
            from outlines.inputs import Chat

            chat_messages = [
                {"role": msg["role"], "content": msg["content"]} for msg in messages
            ]

            logger.debug(
                f"Конвертировано {len(chat_messages)} сообщений в Outlines Chat"
            )
            return Chat(chat_messages)

        except Exception as e:
            logger.error(f"Ошибка конвертации сообщений в Outlines Chat: {e}")
            raise ValidationError(
                f"Не удалось конвертировать сообщения в Outlines Chat: {e}",
                context={"messages_count": len(messages)},
                original_error=e,
            )

    async def _handle_openai_error(self, error: Exception) -> None:
        """
        Обработка ошибок от AsyncOpenAI API.

        Args:
            error: Исключение от AsyncOpenAI

        Raises:
            APIError: Для ошибок API
            TimeoutError: Для ошибок таймаута
            NetworkError: Для сетевых ошибок
        """
        logger.error(f"Ошибка AsyncOpenAI API: {error}")

        # Обработка различных типов ошибок OpenAI
        if hasattr(error, "status_code"):
            raise APIError(
                message=str(error),
                status_code=error.status_code,
                response_data=getattr(error, "response", None),
            )

        # Обработка таймаутов
        if isinstance(error, (asyncio.TimeoutError, httpx.TimeoutException)):
            raise TimeoutError(
                f"Превышено время ожидания: {error}",
                context={"timeout_config": self.config.timeout_config},
                original_error=error,
            )

        # Обработка сетевых ошибок
        if isinstance(error, (httpx.ConnectError, httpx.NetworkError)):
            raise NetworkError(
                f"Сетевая ошибка: {error}",
                context={"endpoint": self.config.endpoint},
                original_error=error,
            )

        # Общая обработка
        raise NetworkError(
            f"Неизвестная ошибка AsyncOpenAI: {error}",
            original_error=error,
        )

    async def _validate_pydantic_response(
        self, response: str, model: BaseModel
    ) -> BaseModel:
        """
        Валидация ответа по Pydantic модели.

        Args:
            response: JSON строка ответа
            model: Pydantic модель для валидации

        Returns:
            Валидированный Pydantic объект

        Raises:
            ValidationError: При ошибках валидации
        """
        try:
            logger.debug(f"Валидация ответа по модели: {model.__name__}")
            validated = model.model_validate_json(response)
            logger.debug("Валидация ответа прошла успешно")
            return validated

        except Exception as e:
            logger.error(f"Ошибка валидации ответа: {e}")
            raise ValidationError(
                f"Не удалось валидировать ответ по модели {model.__name__}: {e}",
                context={
                    "model": model.__name__,
                    "response_length": len(response),
                    "response_preview": (
                        response[:200] + "..." if len(response) > 200 else response
                    ),
                },
                original_error=e,
            )

    # Абстрактные методы для реализации в наследниках

    @abstractmethod
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = None,
        **kwargs,
    ) -> Union[str, Dict[str, Any]]:
        """
        Основной метод для выполнения chat completion.

        Args:
            messages: Список сообщений чата
            model: Переопределение модели (опционально)
            temperature: Переопределение температуры (опционально)
            max_tokens: Переопределение максимального количества токенов (опционально)
            stream: Переопределение режима потока (опционально)
            **kwargs: Дополнительные параметры

        Returns:
            Ответ модели (строка или словарь)
        """
        pass

    @abstractmethod
    async def chat_completion_stream(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Потоковый chat completion.

        Args:
            messages: Список сообщений чата
            **kwargs: Дополнительные параметры

        Yields:
            Части ответа модели по мере генерации
        """
        pass

    @abstractmethod
    async def chat_completion_structured(
        self, messages: List[Dict[str, str]], response_model: BaseModel, **kwargs
    ) -> BaseModel:
        """
        Chat completion с структурированным выводом.

        Args:
            messages: Список сообщений чата
            response_model: Pydantic модель для структурированного ответа
            **kwargs: Дополнительные параметры

        Returns:
            Валидированный Pydantic объект
        """
        pass

    async def close(self) -> None:
        """
        Закрытие клиента и освобождение ресурсов.
        """
        logger.info("Закрытие BaseLLMClient")

        if self.openai_client:
            await self.openai_client.close()
            logger.debug("AsyncOpenAI клиент закрыт")

        # Outlines модель не требует явного закрытия
        self._outlines_model = None

        logger.debug("BaseLLMClient закрыт успешно")

    async def __aenter__(self):
        """Поддержка async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Поддержка async context manager."""
        await self.close()

    def __repr__(self) -> str:
        """Строковое представление клиента."""
        return (
            f"{self.__class__.__name__}("
            f"endpoint='{self.config.endpoint}', "
            f"model='{self.config.model}'"
            f")"
        )
