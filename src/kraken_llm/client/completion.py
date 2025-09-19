"""
Клиент для работы с completion моделями через /v1/completions endpoint.

Этот модуль предоставляет CompletionLLMClient для работы с моделями,
которые используют простой text completion формат вместо chat формата.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, AsyncGenerator
from pydantic import BaseModel
from openai import AsyncOpenAI

from .base import BaseLLMClient
from ..exceptions.api import APIError
from ..exceptions.validation import ValidationError

logger = logging.getLogger(__name__)


class CompletionLLMClient(BaseLLMClient):
    """
    Клиент для работы с completion моделями.
    
    Особенности:
    - Использует /v1/completions endpoint
    - Работает с простыми текстовыми промптами
    - Поддерживает streaming
    - Не поддерживает function calling и structured output
    """

    def __init__(self, config):
        """Инициализация completion клиента"""
        super().__init__(config)
        logger.info(f"Инициализирован CompletionLLMClient для модели {config.model}")

    def _create_completion_openai_client(self):
        """Создание OpenAI клиента специально для completion endpoint"""
        return AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.endpoint.rstrip('/'),
            timeout=self.config.connect_timeout
        )

    def _build_api_url(self, endpoint_type: str = "completions") -> str:
        """Построение URL для completion API"""
        if endpoint_type == "completions":
            return f"{self.config.endpoint.rstrip('/')}/v1/completions"
        return super()._build_api_url(endpoint_type)

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Конвертация сообщений в простой текстовый промпт"""
        prompt_parts = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            else:
                prompt_parts.append(content)
        
        # Добавляем префикс для ответа ассистента
        prompt_parts.append("Assistant:")
        
        return "\n\n".join(prompt_parts)

    def _prepare_completion_params(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Подготовка параметров для completion запроса"""
        params = {
            "model": model or self.config.model,
            "prompt": prompt,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "top_p": kwargs.get("top_p", self.config.top_p),
            "frequency_penalty": kwargs.get("frequency_penalty", self.config.frequency_penalty),
            "presence_penalty": kwargs.get("presence_penalty", self.config.presence_penalty),
        }
        
        if stream is not None:
            params["stream"] = stream

        # Logprobs для legacy /v1/completions ожидает целочисленное значение
        # Желаемое значение k из kwargs или config
        lp_kw = kwargs.get("logprobs", None)
        top_lp_kw = kwargs.get("top_logprobs", getattr(self.config, "top_logprobs", None))
        if isinstance(lp_kw, int):
            params["logprobs"] = lp_kw
        elif lp_kw is True or (lp_kw is None and getattr(self.config, "logprobs", None)):
            params["logprobs"] = top_lp_kw or 1
            
        # Добавляем дополнительные параметры
        for key, value in kwargs.items():
            if key not in params and value is not None:
                params[key] = value
        
        return params

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = None,
        **kwargs,
    ) -> str:
        """
        Выполнение completion запроса с автоматическим fallback на chat API.
        
        Args:
            messages: Список сообщений (будут конвертированы в промпт)
            model: Название модели
            temperature: Температура генерации
            max_tokens: Максимальное количество токенов
            stream: Использовать ли streaming (игнорируется, всегда True)
            **kwargs: Дополнительные параметры
            
        Returns:
            Сгенерированный текст
        """
        try:
            # Сначала пробуем completion API
            full_text = ""
            async for chunk in self.chat_completion_stream(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            ):
                full_text += chunk
            
            return full_text.strip()
                
        except Exception as e:
            logger.error(f"Ошибка в completion запросе: {e}")
            
            # Если получили 404, пробуем fallback на chat API
            if "404" in str(e) or "not found" in str(e).lower():
                logger.warning("Completion endpoint не найден, пробуем fallback на chat API")
                try:
                    # Используем базовый chat completion
                    return await super().chat_completion(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs
                    )
                except Exception as fallback_error:
                    raise APIError(f"Ошибка и в completion, и в chat API: {e} | {fallback_error}")
            
            elif "400" in str(e):
                raise APIError(f"Некорректный запрос: {e}")
            else:
                raise APIError(f"Ошибка completion API: {e}")

    async def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Streaming completion запрос.
        
        Args:
            messages: Список сообщений (будут конвертированы в промпт)
            model: Название модели
            temperature: Температура генерации
            max_tokens: Максимальное количество токенов
            **kwargs: Дополнительные параметры
            
        Yields:
            Части сгенерированного текста
        """
        try:
            # Конвертируем сообщения в промпт
            prompt = self._messages_to_prompt(messages)
            
            # Подготавливаем параметры
            params = self._prepare_completion_params(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            )
            
            logger.debug(f"Отправка streaming completion запроса: {params}")
            
            # Создаем OpenAI клиент для completion
            openai_client = self._create_completion_openai_client()
            
            # Выполняем streaming запрос
            stream = await openai_client.completions.create(**params)
            
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    text = chunk.choices[0].text
                    if text:
                        yield text
                        
        except Exception as e:
            logger.error(f"Ошибка в streaming completion запросе: {e}")
            if "404" in str(e):
                raise APIError(f"Endpoint не найден: {e}")
            elif "400" in str(e):
                raise APIError(f"Некорректный запрос: {e}")
            else:
                raise APIError(f"Ошибка streaming completion API: {e}")

    async def chat_completion_structured(
        self,
        messages: List[Dict[str, str]],
        response_model: BaseModel,
        **kwargs
    ) -> BaseModel:
        """
        Completion модели не поддерживают structured output нативно.
        Этот метод попытается получить JSON и распарсить его.
        """
        # Добавляем инструкцию для JSON ответа
        json_instruction = f"\n\nОтветь в формате JSON согласно схеме: {response_model.model_json_schema()}"
        
        # Модифицируем последнее сообщение
        modified_messages = messages.copy()
        if modified_messages:
            last_message = modified_messages[-1].copy()
            last_message["content"] += json_instruction
            modified_messages[-1] = last_message
        
        # Получаем ответ
        response_text = await self.chat_completion(modified_messages, **kwargs)
        
        # Пытаемся распарсить JSON
        try:
            import json
            # Ищем JSON в ответе
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                json_data = json.loads(json_str)
                return response_model(**json_data)
            else:
                raise ValidationError(f"Не удалось найти JSON в ответе: {response_text}")
                
        except Exception as e:
            raise ValidationError(f"Ошибка парсинга structured ответа: {e}")

    def text_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Прямой text completion запрос (синхронный).
        
        Args:
            prompt: Текстовый промпт
            model: Название модели
            temperature: Температура генерации
            max_tokens: Максимальное количество токенов
            **kwargs: Дополнительные параметры
            
        Returns:
            Сгенерированный текст
        """
        return asyncio.run(self.text_completion_async(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        ))

    async def text_completion_async(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Прямой text completion запрос через streaming (асинхронный).
        
        Args:
            prompt: Текстовый промпт
            model: Название модели
            temperature: Температура генерации
            max_tokens: Максимальное количество токенов
            **kwargs: Дополнительные параметры
            
        Returns:
            Сгенерированный текст
        """
        try:
            # Используем streaming и собираем результат
            full_text = ""
            async for chunk in self.text_completion_stream(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            ):
                full_text += chunk
            
            return full_text.strip()
                
        except Exception as e:
            logger.error(f"Ошибка в text completion запросе: {e}")
            if "404" in str(e):
                raise APIError(f"Endpoint не найден: {e}")
            elif "400" in str(e):
                raise APIError(f"Некорректный запрос: {e}")
            else:
                raise APIError(f"Ошибка text completion API: {e}")

    async def text_completion_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Streaming text completion запрос.
        
        Args:
            prompt: Текстовый промпт
            model: Название модели
            temperature: Температура генерации
            max_tokens: Максимальное количество токенов
            **kwargs: Дополнительные параметры
            
        Yields:
            Части сгенерированного текста
        """
        try:
            # Подготавливаем параметры
            params = self._prepare_completion_params(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            )
            
            logger.debug(f"Отправка streaming text completion запроса: {params}")
            
            # Создаем OpenAI клиент для completion
            openai_client = self._create_completion_openai_client()
            
            # Выполняем streaming запрос
            stream = await openai_client.completions.create(**params)
            
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    text = chunk.choices[0].text
                    if text:
                        yield text
                        
        except Exception as e:
            logger.error(f"Ошибка в streaming text completion запросе: {e}")
            if "404" in str(e):
                raise APIError(f"Endpoint не найден: {e}")
            elif "400" in str(e):
                raise APIError(f"Некорректный запрос: {e}")
            else:
                raise APIError(f"Ошибка streaming text completion API: {e}")

    def get_supported_capabilities(self) -> List[str]:
        """Получить список поддерживаемых возможностей"""
        return [
            "text_completion",
            "chat_completion", 
            "streaming",
            "basic_structured_output"  # Через промпт инжиниринг
        ]

    def __repr__(self) -> str:
        return f"CompletionLLMClient(model={self.config.model}, endpoint={self.config.endpoint})"