"""
Structured output LLM клиент для Kraken фреймворка.

Этот модуль содержит StructuredLLMClient - специализированную реализацию LLM клиента
для получения структурированных ответов с валидацией через Pydantic модели.
Поддерживает переключение между нативным OpenAI response_format и Outlines для structured output.
"""

import asyncio
import json
import re
from typing import Any, AsyncGenerator, Dict, List, Optional, Type, Union

from pydantic import BaseModel
from openai.types.chat import ChatCompletion

from .base import BaseLLMClient
from ..exceptions.api import APIError
from ..exceptions.validation import ValidationError
from ..structured.validator import StructuredOutputValidator
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ValidationResult:
    """Результат валидации инкрементального JSON парсинга."""
    
    def __init__(self, is_complete: bool = False, is_invalid: bool = False, 
                 parsed_object: Optional[BaseModel] = None, error: Optional[str] = None):
        self.is_complete = is_complete
        self.is_invalid = is_invalid
        self.parsed_object = parsed_object
        self.error = error


class IncrementalJSONParser:
    """
    Инкрементальный JSON парсер для streaming structured output.
    
    Парсит JSON по мере поступления токенов и валидирует против Pydantic модели.
    """
    
    def __init__(self, response_model: Type[BaseModel]):
        self.response_model = response_model
        self.content_buffer = ""
        self.json_started = False
        self.json_depth = 0
        self.in_string = False
        self.escape_next = False
        self.last_validation_attempt = ""
        
    def add_content(self, new_content: str) -> ValidationResult:
        """
        Добавляет новый контент и пытается валидировать JSON.
        
        Args:
            new_content: Новый контент от streaming API
            
        Returns:
            ValidationResult с результатом валидации
        """
        self.content_buffer += new_content
        
        # Обновляем состояние парсера
        self._update_parser_state(new_content)
        
        # Если JSON еще не начался, ищем начало
        if not self.json_started:
            json_start = self.content_buffer.find('{')
            if json_start != -1:
                self.json_started = True
                self.content_buffer = self.content_buffer[json_start:]
                self.json_depth = 1
        
        # Если JSON начался, проверяем завершенность
        if self.json_started and self.json_depth == 0:
            # JSON потенциально завершен, пытаемся валидировать
            return self._attempt_validation()
        
        # Периодически пытаемся валидировать частичный JSON
        if len(self.content_buffer) > len(self.last_validation_attempt) + 10:
            partial_result = self._attempt_partial_validation()
            if partial_result.is_complete:
                return partial_result
        
        return ValidationResult()
    
    def _update_parser_state(self, new_content: str):
        """Обновляет состояние парсера на основе нового контента."""
        for char in new_content:
            if self.escape_next:
                self.escape_next = False
                continue
                
            if char == '\\' and self.in_string:
                self.escape_next = True
                continue
                
            if char == '"' and not self.escape_next:
                self.in_string = not self.in_string
                continue
                
            if not self.in_string:
                if char == '{':
                    self.json_depth += 1
                elif char == '}':
                    self.json_depth -= 1
    
    def _attempt_validation(self) -> ValidationResult:
        """Пытается валидировать полный JSON."""
        try:
            # Извлекаем JSON из буфера
            json_content = self._extract_complete_json()
            
            if json_content:
                parsed_data = json.loads(json_content)
                validated_object = self.response_model.model_validate(parsed_data)
                return ValidationResult(is_complete=True, parsed_object=validated_object)
            
        except json.JSONDecodeError as e:
            return ValidationResult(is_invalid=True, error=f"JSON decode error: {e}")
        except Exception as e:
            return ValidationResult(is_invalid=True, error=f"Validation error: {e}")
        
        return ValidationResult()
    
    def _attempt_partial_validation(self) -> ValidationResult:
        """Пытается валидировать частичный JSON."""
        self.last_validation_attempt = self.content_buffer
        
        # Пытаемся найти и валидировать завершенный JSON в буфере
        potential_json = self._extract_potential_json()
        
        if potential_json:
            try:
                parsed_data = json.loads(potential_json)
                validated_object = self.response_model.model_validate(parsed_data)
                return ValidationResult(is_complete=True, parsed_object=validated_object)
            except:
                pass  # Продолжаем накапливать контент
        
        return ValidationResult()
    
    def _extract_complete_json(self) -> Optional[str]:
        """Извлекает завершенный JSON из буфера."""
        if not self.json_started:
            return None
            
        # Ищем первый полный JSON объект
        json_start = self.content_buffer.find('{')
        if json_start == -1:
            return None
            
        depth = 0
        in_string = False
        escape_next = False
        
        for i, char in enumerate(self.content_buffer[json_start:], json_start):
            if escape_next:
                escape_next = False
                continue
                
            if char == '\\' and in_string:
                escape_next = True
                continue
                
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
                
            if not in_string:
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        return self.content_buffer[json_start:i+1]
        
        return None
    
    def _extract_potential_json(self) -> Optional[str]:
        """Извлекает потенциальный JSON, даже если он не завершен."""
        # Используем существующий метод извлечения JSON
        json_patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Полный JSON объект
            r'\{.*',  # Начало JSON объекта
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, self.content_buffer, re.DOTALL)
            if matches:
                return matches[0]
        
        return None
    
    def finalize(self) -> ValidationResult:
        """Финализирует парсинг и возвращает результат."""
        return self._attempt_validation()
    
    def get_full_content(self) -> str:
        """Возвращает весь накопленный контент."""
        return self.content_buffer
    
    def get_state_info(self) -> Dict[str, Any]:
        """Возвращает информацию о состоянии парсера для отладки."""
        return {
            "content_length": len(self.content_buffer),
            "json_started": self.json_started,
            "json_depth": self.json_depth,
            "in_string": self.in_string,
            "content_preview": self.content_buffer[:100] if self.content_buffer else ""
        }


class StructuredLLMClient(BaseLLMClient):
    """
    LLM клиент для structured output с валидацией.

    Реализует полную поддержку structured output через:
    - Нативный OpenAI response_format для генерации структурированных ответов
    - Outlines библиотеку для альтернативного structured output
    - Переключение между режимами через параметр outlines_so_mode
    - Поддержку как streaming, так и non-streaming режимов
    - Автоматическую валидацию ответов через Pydantic модели
    - Обработку сложных схем с вложенными объектами, массивами и union типами
    """

    def __init__(self, config):
        """
        Инициализация structured output LLM клиента.

        Args:
            config: Конфигурация клиента
        """
        super().__init__(config)

        # Инициализация валидатора
        self.validator = StructuredOutputValidator()

        logger.info(
            f"StructuredLLMClient инициализирован с моделью: {config.model}, "
            f"outlines_so_mode: {config.outlines_so_mode}"
        )

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = None,
        response_model: Optional[Type[BaseModel]] = None,
        **kwargs
    ) -> Union[str, Dict[str, Any], BaseModel, AsyncGenerator[str, None]]:
        """
        Выполнение chat completion с поддержкой structured output.

        Args:
            messages: Список сообщений чата
            model: Переопределение модели (опционально)
            temperature: Переопределение температуры (опционально)
            max_tokens: Переопределение максимального количества токенов (опционально)
            stream: Использовать потоковый режим (опционально)
            response_model: Pydantic модель для structured output (опционально)
            **kwargs: Дополнительные параметры

        Returns:
            Ответ модели (строка, словарь или Pydantic объект)

        Raises:
            ValidationError: При ошибках валидации входных данных или ответа
            APIError: При ошибках API
            NetworkError: При сетевых ошибках
        """
        # Валидация входных параметров
        if not messages:
            raise ValidationError("Список сообщений не может быть пустым")

        # Проверяем, что response_model указан для structured output
        if response_model is None:
            raise ValidationError(
                "response_model обязателен для structured output")

        # Валидация Pydantic модели (проверяем, что это валидная модель)
        try:
            if not issubclass(response_model, BaseModel):
                raise ValidationError(
                    f"response_model должен быть Pydantic BaseModel, получен: {type(response_model)}")
        except TypeError:
            raise ValidationError(
                f"response_model должен быть Pydantic BaseModel, получен: {type(response_model)}")

        # Выполняем structured output
        logger.info(
            f"Выполнение structured output запроса с моделью {response_model.__name__}, "
            f"stream: {stream}, outlines_mode: {self.config.outlines_so_mode}"
        )

        try:
            if stream:
                return await self._structured_stream(
                    messages, response_model, model, temperature, max_tokens, **kwargs
                )
            else:
                return await self._structured_non_stream(
                    messages, response_model, model, temperature, max_tokens, **kwargs
                )
        except Exception as e:
            logger.error(f"Ошибка в structured output запросе: {e}")
            raise

    async def _structured_non_stream(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[BaseModel],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> BaseModel:
        """
        Non-streaming structured output с переключением между OpenAI и Outlines.
        """
        logger.debug(
            f"Выполнение non-streaming structured output с моделью {response_model.__name__}")

        # Проверяем режим structured output
        if self.config.outlines_so_mode:
            logger.debug("Используем Outlines для structured output")
            return await self._structured_non_stream_outlines(
                messages, response_model, model, temperature, max_tokens, **kwargs
            )
        else:
            logger.debug("Используем нативный OpenAI для structured output")
            return await self._structured_non_stream_openai(
                messages, response_model, model, temperature, max_tokens, **kwargs
            )

    async def _structured_non_stream_openai(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[BaseModel],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> BaseModel:
        """
        Non-streaming structured output через нативный OpenAI response_format.
        """
        try:
            # Получаем JSON схему модели
            json_schema = response_model.model_json_schema()

            # Подготовка параметров для AsyncOpenAI
            params = self._prepare_openai_params(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                **kwargs
            )

            # Добавляем response_format для structured output
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": json_schema,
                    "strict": True
                }
            }

            # Сохраняем response_model отдельно для streaming агрегации (не передаем в API)
            # params["_response_model"] = response_model  # Убираем - completion API не поддерживает

            logger.debug(
                f"Параметры OpenAI structured output: {list(params.keys())}")

            # Проверяем, поддерживает ли сервер обычные (не streaming) запросы
            if await self._server_supports_non_streaming():
                # Выполнение запроса через AsyncOpenAI
                response: ChatCompletion = await self.openai_client.chat.completions.create(**params)
                logger.debug(f"Получен ответ от OpenAI API: {response.id}")

                # Извлекаем контент
                if not response.choices:
                    raise APIError(
                        message="Ответ API не содержит choices",
                        status_code=500,
                        response_data={"response_id": response.id}
                    )

                choice = response.choices[0]
                content = choice.message.content

                if not content:
                    raise ValidationError(
                        "Получен пустой ответ от модели",
                        context={"response_id": response.id}
                    )
            else:
                # Сервер поддерживает только streaming, используем агрегацию
                content = await self._structured_via_streaming_aggregation(params)

            logger.debug(f"Получен JSON контент: {len(content)} символов")

            # Валидируем ответ по Pydantic модели
            result = await self.validator.validate_response(content, response_model)
            logger.info(
                f"OpenAI structured output успешно сгенерирован и валидирован")
            return result

        except Exception as e:
            logger.error(f"Ошибка в _structured_non_stream_openai: {e}")
            # Если это ошибка валидации, пробрасываем как есть
            if isinstance(e, ValidationError):
                raise
            # Обрабатываем ошибки OpenAI API
            await self._handle_openai_error(e)

    async def _server_supports_non_streaming(self) -> bool:
        """
        Проверяет, поддерживает ли сервер обычные (не streaming) запросы.

        Кэширует результат для избежания повторных проверок.
        """
        if not hasattr(self, '_non_streaming_support'):
            try:
                # Пробуем простой запрос с stream=false
                import httpx

                headers = {
                    "Authorization": f"Bearer {self.config.api_key or 'dummy'}",
                    "Content-Type": "application/json"
                }

                test_data = {
                    "model": self.config.model,
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 1,
                    "stream": False
                }

                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.config.endpoint}/v1/chat/completions",
                        json=test_data,
                        headers=headers,
                        timeout=5.0
                    )

                    # Если получили JSON ответ, сервер поддерживает non-streaming
                    if response.headers.get("content-type", "").startswith("application/json"):
                        self._non_streaming_support = True
                    else:
                        self._non_streaming_support = False

            except Exception:
                # В случае ошибки считаем, что сервер не поддерживает non-streaming
                self._non_streaming_support = False

        return self._non_streaming_support

    async def _server_supports_streaming_response_format(self) -> bool:
        """
        Проверяет, поддерживает ли сервер response_format в streaming режиме.

        Returns:
            True если поддерживает, False если нет
        """
        # Большинство streaming-only серверов не поддерживают response_format
        # Возвращаем False для использования промптов
        return False

    async def _structured_via_streaming_aggregation(self, params: Dict[str, Any]) -> str:
        """
        Выполняет structured output через streaming агрегацию.

        Args:
            params: Параметры запроса

        Returns:
            Агрегированный контент ответа
        """
        import httpx
        import json

        headers = {
            "Authorization": f"Bearer {self.config.api_key or 'dummy'}",
            "Content-Type": "application/json"
        }

        # Подготавливаем параметры для streaming
        streaming_params = params.copy()
        streaming_params["stream"] = True

        # Получаем response_model для обработки
        response_model = streaming_params.pop("_response_model", None)

        # Проверяем, поддерживает ли сервер response_format в streaming
        # Если нет, используем улучшенные промпты
        if not await self._server_supports_streaming_response_format():
            logger.debug(
                "Сервер не поддерживает response_format в streaming, используем промпты")
            streaming_params.pop("response_format", None)
            # Добавляем улучшенные промпты для JSON генерации
            if response_model:
                streaming_params["messages"] = self._enhance_messages_for_json(
                    streaming_params["messages"], response_model
                )

        logger.debug("Выполняем structured output через streaming агрегацию")

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.config.endpoint}/v1/chat/completions",
                json=streaming_params,
                headers=headers,
                timeout=httpx.Timeout(
                    connect=self.config.connect_timeout,
                    read=self.config.read_timeout,
                    write=self.config.write_timeout,
                    pool=None
                )
            ) as response:

                if response.status_code != 200:
                    error_text = await response.aread()
                    raise Exception(
                        f"HTTP {response.status_code}: {error_text.decode()}")

                # Агрегируем streaming ответ
                content = ""

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue

                    data_str = line[6:]  # Убираем "data: "
                    if data_str.strip() == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)

                        if "choices" in chunk and chunk["choices"]:
                            choice = chunk["choices"][0]
                            delta = choice.get("delta", {})

                            # Агрегируем контент
                            if "content" in delta and delta["content"]:
                                content += delta["content"]

                    except json.JSONDecodeError:
                        continue

                return content.strip()

    async def _structured_non_stream_outlines(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[BaseModel],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> BaseModel:
        """
        Non-streaming structured output через Outlines.

        Для streaming-only серверов используем прямой HTTP запрос
        и обрабатываем ответ через Outlines.
        """
        try:
            logger.debug("Начинаем Outlines structured output")

            # Для streaming-only серверов делаем прямой HTTP запрос
            import httpx
            import json

            # Подготавливаем промпт для JSON генерации
            enhanced_messages = self._enhance_messages_for_json(
                messages, response_model)

            headers = {
                "Authorization": f"Bearer {self.config.api_key or 'dummy'}",
                "Content-Type": "application/json"
            }

            # Параметры запроса
            request_data = {
                "model": model or self.config.model,
                "messages": enhanced_messages,
                "temperature": temperature or self.config.temperature,
                "max_tokens": max_tokens or self.config.max_tokens,
                "top_p": self.config.top_p,
                "frequency_penalty": self.config.frequency_penalty,
                "presence_penalty": self.config.presence_penalty,
            }

            logger.debug("Выполняем HTTP запрос для Outlines")

            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{self.config.endpoint}/v1/chat/completions",
                    json=request_data,
                    headers=headers,
                    timeout=httpx.Timeout(
                        connect=self.config.connect_timeout,
                        read=self.config.read_timeout,
                        write=self.config.write_timeout,
                        pool=None
                    )
                ) as response:

                    if response.status_code != 200:
                        error_text = await response.aread()
                        raise Exception(
                            f"HTTP {response.status_code}: {error_text.decode()}")

                    # Агрегируем streaming ответ
                    content = ""

                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue

                        data_str = line[6:]  # Убираем "data: "
                        if data_str.strip() == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data_str)

                            if "choices" in chunk and chunk["choices"]:
                                choice = chunk["choices"][0]
                                delta = choice.get("delta", {})

                                # Агрегируем контент
                                if "content" in delta and delta["content"]:
                                    content += delta["content"]

                        except json.JSONDecodeError:
                            continue

            logger.debug(f"Получен ответ от сервера: {content[:100]}...")

            # Пытаемся извлечь JSON из ответа
            json_content = self._extract_json_from_response(content)

            # Дополнительная валидация контекста для Outlines режима
            if not self._is_response_contextually_valid(content, messages):
                logger.warning(
                    "Ответ не соответствует контексту запроса, возможно Outlines слишком агрессивен")
                # В строгом режиме выбрасываем ошибку
                raise ValidationError(
                    "Ответ не соответствует контексту запроса",
                    context={
                        "original_request": messages[-1].get("content", "")[:100] if messages else "",
                        "response_preview": content[:100]
                    }
                )

            # Валидируем через Pydantic
            if json_content:
                try:
                    # Парсим JSON
                    parsed_data = json.loads(json_content)

                    # Валидируем через Pydantic модель
                    result = response_model.model_validate(parsed_data)

                    logger.info(
                        f"Outlines structured output успешно сгенерирован и валидирован")
                    return result

                except (json.JSONDecodeError, Exception) as e:
                    logger.warning(f"Ошибка парсинга JSON: {e}")
                    # Fallback к обычной валидации
                    return await self.validator.validate_response(json_content, response_model)
            else:
                # Если JSON не найден, пытаемся валидировать весь ответ
                return await self.validator.validate_response(content, response_model)

        except Exception as e:
            logger.error(f"Ошибка в _structured_non_stream_outlines: {e}")
            # Если это ошибка валидации, пробрасываем как есть
            if isinstance(e, ValidationError):
                raise
            # Иначе оборачиваем в ValidationError
            raise ValidationError(
                f"Не удалось выполнить Outlines structured output: {e}",
                context={
                    "model": response_model.__name__,
                    "messages_count": len(messages),
                },
                original_error=e,
            )

    async def _structured_stream(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[BaseModel],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> BaseModel:
        """
        Потоковый structured output с переключением между OpenAI и Outlines.
        """
        logger.debug(
            f"Выполнение streaming structured output с моделью {response_model.__name__}")

        # Проверяем режим structured output
        if self.config.outlines_so_mode:
            logger.debug("Используем Outlines для streaming structured output")
            # Outlines не поддерживает настоящий streaming, используем _structured_stream_outlines
            # который внутри делает non-streaming запрос
            return await self._structured_stream_outlines(
                messages, response_model, model, temperature, max_tokens, **kwargs
            )
        else:
            logger.debug(
                "Используем нативный OpenAI для streaming structured output")
            return await self._structured_stream_openai(
                messages, response_model, model, temperature, max_tokens, **kwargs
            )

    async def _structured_stream_openai(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[BaseModel],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> BaseModel:
        """
        Потоковый structured output через нативный OpenAI streaming.
        """
        try:
            # Получаем JSON схему модели
            json_schema = response_model.model_json_schema()

            # Подготовка параметров для AsyncOpenAI streaming
            params = self._prepare_openai_params(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            )

            # Добавляем response_format для structured output
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": json_schema,
                    "strict": True
                }
            }

            logger.debug(
                f"Параметры OpenAI streaming structured output: {list(params.keys())}")

            # Создание потокового запроса через AsyncOpenAI
            stream = await self.openai_client.chat.completions.create(**params)
            logger.debug("Потоковый запрос создан, начинаем агрегацию chunks")

            # Агрегация контента из потока
            chunks = []
            chunk_count = 0

            async for chunk in stream:
                chunk_count += 1
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    chunks.append(content)
                    logger.debug(
                        f"Получен streaming chunk #{chunk_count}: {len(content)} символов")

            logger.debug(
                f"Потоковая генерация завершена, получено {chunk_count} chunks")

            # Агрегация всех chunks
            full_response = "".join(chunks)

            if not full_response:
                raise ValidationError(
                    "Получен пустой ответ из потока",
                    context={"chunks_received": chunk_count}
                )

            logger.debug(
                f"Агрегированный ответ: {len(full_response)} символов")

            # Валидация агрегированного ответа
            result = await self.validator.validate_response(full_response, response_model)
            logger.info(
                f"Streaming structured output успешно сгенерирован и валидирован")
            return result

        except Exception as e:
            logger.error(f"Ошибка в _structured_stream_openai: {e}")
            # Если это ошибка валидации, пробрасываем как есть
            if isinstance(e, ValidationError):
                raise
            # Обрабатываем ошибки OpenAI API
            await self._handle_openai_error(e)

    # Реализация абстрактных методов из базового класса
    async def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Потоковый чат completion (не поддерживается для structured output).

        Raises:
            NotImplementedError: Structured output клиент не поддерживает обычный streaming
        """
        raise NotImplementedError(
            "StructuredLLMClient не поддерживает обычный streaming. "
            "Используйте chat_completion с stream=True для structured streaming."
        )

    async def chat_completion_structured(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[BaseModel],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = None,
        **kwargs
    ) -> BaseModel:
        """
        Structured output completion (алиас для chat_completion).

        Args:
            messages: Список сообщений чата
            response_model: Pydantic модель для валидации ответа
            model: Переопределение модели (опционально)
            temperature: Переопределение температуры (опционально)
            max_tokens: Переопределение максимального количества токенов (опционально)
            stream: Использовать потоковый режим (опционально)
            **kwargs: Дополнительные параметры

        Returns:
            Валидированный Pydantic объект
        """
        # Если stream не указан, используем False по умолчанию
        if stream is None:
            stream = False

        return await self.chat_completion(
            messages=messages,
            response_model=response_model,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            **kwargs
        )

    async def structured_completion(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[BaseModel],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[BaseModel, AsyncGenerator[str, None]]:
        """
        Удобный метод для structured output запросов.

        Args:
            messages: Список сообщений чата
            response_model: Pydantic модель для валидации ответа
            model: Переопределение модели (опционально)
            temperature: Переопределение температуры (опционально)
            max_tokens: Переопределение максимального количества токенов (опционально)
            stream: Использовать потоковый режим
            **kwargs: Дополнительные параметры

        Returns:
            Валидированный Pydantic объект или AsyncGenerator для потокового режима
        """
        return await self.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            response_model=response_model,
            **kwargs
        )

    def _enhance_messages_for_json(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[BaseModel]
    ) -> List[Dict[str, str]]:
        """
        Улучшает сообщения для генерации JSON.

        Args:
            messages: Исходные сообщения
            response_model: Pydantic модель для генерации схемы

        Returns:
            Улучшенные сообщения с инструкциями для JSON
        """
        # Генерируем JSON схему из Pydantic модели
        schema = response_model.model_json_schema()

        # Создаем пример JSON на основе схемы
        example_json = self._generate_example_json(schema)

        # Создаем системное сообщение с инструкциями
        system_message = {
            "role": "system",
            "content": f"""Ты ОБЯЗАН вернуть ТОЛЬКО валидный JSON объект. Никакого другого текста!

КРИТИЧЕСКИ ВАЖНО: 
- Используй ТОЛЬКО английские ключи из схемы
- НЕ добавляй объяснений, описаний или дополнительного текста
- НЕ используй markdown форматирование
- Ответ должен начинаться с {{ и заканчиваться }}
- Заполни ВСЕ обязательные поля из схемы
- Для отсутствующих данных используй разумные значения по умолчанию

Схема JSON (используй ТОЧНО эти английские ключи):
{json.dumps(schema, indent=2, ensure_ascii=False)}

Пример ПРАВИЛЬНОГО ответа:
{json.dumps(example_json, indent=2, ensure_ascii=False)}

ОБЯЗАТЕЛЬНЫЕ ПРАВИЛА:
- Все поля из схемы должны присутствовать
- Используй только английские ключи
- Никакого дополнительного текста
- Только валидный JSON

ЗАПРЕЩЕНО:
- Русские ключи (имя, возраст, город)
- Дополнительный текст
- Markdown (```)
- Объяснения
- Пропуск обязательных полей"""
        }

        # Модифицируем сообщения
        enhanced_messages = messages.copy()

        # Добавляем или обновляем системное сообщение
        if enhanced_messages and enhanced_messages[0]["role"] == "system":
            enhanced_messages[0]["content"] += "\n\n" + \
                system_message["content"]
        else:
            enhanced_messages.insert(0, system_message)

        # Улучшаем пользовательское сообщение
        if enhanced_messages:
            last_user_msg = None
            for i, msg in enumerate(enhanced_messages):
                if msg["role"] == "user":
                    last_user_msg = i

            if last_user_msg is not None:
                original_content = enhanced_messages[last_user_msg]["content"]
                enhanced_messages[last_user_msg]["content"] = f"""{original_content}

ОТВЕТ ДОЛЖЕН БЫТЬ ТОЛЬКО JSON:
- Используй английские ключи: name, age, city, occupation
- НЕ используй русские ключи: имя, возраст, город, профессия
- Начни ответ с {{ и закончи }}
- Никакого дополнительного текста!"""

        return enhanced_messages

    def _generate_example_json(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Генерирует пример JSON на основе схемы с осмысленными значениями.

        Args:
            schema: JSON схема

        Returns:
            Пример JSON объекта
        """
        example = {}

        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                prop_type = prop_schema.get("type", "string")
                description = prop_schema.get("description", "")

                # Генерируем осмысленные примеры на основе имени поля
                if prop_type == "string":
                    example[prop_name] = self._get_example_string_value(
                        prop_name, description)
                elif prop_type == "integer":
                    example[prop_name] = self._get_example_integer_value(
                        prop_name, description)
                elif prop_type == "number":
                    example[prop_name] = self._get_example_number_value(
                        prop_name, description)
                elif prop_type == "boolean":
                    example[prop_name] = self._get_example_boolean_value(
                        prop_name, description)
                elif prop_type == "array":
                    example[prop_name] = self._get_example_array_value(
                        prop_name, description)
                elif prop_type == "object":
                    example[prop_name] = {"key": "value"}
                else:
                    example[prop_name] = f"example_{prop_name}"

        return example

    def _get_example_string_value(self, field_name: str, description: str) -> str:
        """Генерирует осмысленное строковое значение для поля"""
        field_lower = field_name.lower()

        if "name" in field_lower:
            return "John Doe"
        elif "city" in field_lower:
            return "New York"
        elif "occupation" in field_lower or "job" in field_lower:
            return "Software Engineer"
        elif "title" in field_lower:
            return "Example Title"
        elif "category" in field_lower:
            return "Electronics"
        elif "description" in field_lower or "bio" in field_lower:
            return "Professional with extensive experience in the field"
        elif "email" in field_lower:
            return "john.doe@example.com"
        elif "phone" in field_lower:
            return "+1-555-123-4567"
        elif "address" in field_lower:
            return "123 Main Street, Apt 4B"
        elif "country" in field_lower:
            return "United States"
        elif "company" in field_lower:
            return "Tech Corporation"
        elif "education" in field_lower:
            return "Bachelor of Science in Computer Science"
        else:
            return f"example_{field_name}"

    def _get_example_integer_value(self, field_name: str, description: str) -> int:
        """Генерирует осмысленное целочисленное значение для поля"""
        field_lower = field_name.lower()

        if "age" in field_lower:
            return 30
        elif "price" in field_lower or "cost" in field_lower:
            return 100
        elif "quantity" in field_lower or "count" in field_lower:
            return 5
        elif "year" in field_lower:
            return 2024
        elif "id" in field_lower:
            return 123
        else:
            return 42

    def _get_example_number_value(self, field_name: str, description: str) -> float:
        """Генерирует осмысленное числовое значение для поля"""
        field_lower = field_name.lower()

        if "price" in field_lower or "cost" in field_lower:
            return 99.99
        elif "weight" in field_lower:
            return 1.5
        elif "height" in field_lower:
            return 180.0
        elif "rating" in field_lower:
            return 4.5
        else:
            return 3.14

    def _get_example_boolean_value(self, field_name: str, description: str) -> bool:
        """Генерирует осмысленное булево значение для поля"""
        field_lower = field_name.lower()

        if "active" in field_lower or "enabled" in field_lower:
            return True
        elif "stock" in field_lower or "available" in field_lower:
            return True
        elif "deleted" in field_lower or "disabled" in field_lower:
            return False
        else:
            return True

    def _get_example_array_value(self, field_name: str, description: str) -> list:
        """Генерирует осмысленное значение массива для поля"""
        field_lower = field_name.lower()

        if "tag" in field_lower:
            return ["web", "development", "programming"]
        elif "skill" in field_lower:
            return ["Python", "JavaScript", "React", "Node.js"]
        elif "language" in field_lower:
            return ["English", "Spanish", "French"]
        elif "hobby" in field_lower or "hobbies" in field_lower:
            return ["reading", "programming", "traveling"]
        elif "color" in field_lower:
            return ["red", "blue", "green"]
        else:
            return ["item1", "item2", "item3"]

    def _is_response_contextually_valid(self, response: str, messages: List[Dict[str, str]]) -> bool:
        """
        Проверяет, соответствует ли ответ контексту запроса.

        Args:
            response: Ответ модели
            messages: Исходные сообщения

        Returns:
            True если ответ контекстуально валиден
        """
        if not messages:
            return True

        last_message = messages[-1].get("content", "").lower()
        response_lower = response.lower()

        # Проверяем на явные признаки неконтекстуального ответа
        non_contextual_indicators = [
            "анекдот", "шутка", "история", "рассказ", "сказка",
            "конечно!", "вот один", "давайте", "можно рассказать",
            "хочешь", "если хочешь", "могу рассказать"
        ]

        # Проверяем на запросы, которые явно НЕ про JSON/данные
        non_data_requests = [
            "анекдот", "шутка", "история", "рассказ", "сказка",
            "расскажи", "поведай", "придумай историю"
        ]

        # Если запрос явно не про данные, а ответ содержит неконтекстуальные индикаторы
        request_is_non_data = any(
            indicator in last_message for indicator in non_data_requests)
        response_has_non_contextual = any(
            indicator in response_lower for indicator in non_contextual_indicators)

        if request_is_non_data and response_has_non_contextual:
            # Дополнительная проверка: если ответ содержит валидный JSON, это подозрительно
            try:
                import json
                # Пытаемся найти JSON в ответе
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    potential_json = response[json_start:json_end]
                    parsed = json.loads(potential_json)
                    # Если JSON валиден и содержит структурированные данные, это подозрительно
                    if isinstance(parsed, dict) and len(parsed) > 0:
                        logger.warning(
                            f"Подозрительный JSON в неконтекстуальном ответе: {potential_json[:50]}...")
                        return False
            except:
                pass

            # Если нет JSON, но есть неконтекстуальные индикаторы - это нормально
            return True

        # Если в запросе есть указание на JSON/данные, ответ должен быть релевантным
        if any(keyword in last_message for keyword in ["json", "данные", "объект", "структура"]):
            # Проверяем, что ответ не содержит явно неконтекстуальный контент
            if response_has_non_contextual and "json" not in response_lower:
                return False

        return True

    def _extract_json_from_response(self, response: str) -> Optional[str]:
        """
        Извлекает JSON из ответа модели.

        Args:
            response: Ответ модели

        Returns:
            JSON строка или None если не найдена
        """
        # Ищем JSON объект в ответе
        json_patterns = [
            r'```json\s*(\{.*?\})\s*```',        # JSON в code block
            r'```\s*(\{.*?\})\s*```',            # JSON в обычном code block
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Простой JSON объект
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                # Если это tuple (из группы), берем первый элемент
                json_candidate = match if isinstance(match, str) else match

                # Проверяем, что это валидный JSON
                try:
                    json.loads(json_candidate)
                    return json_candidate
                except json.JSONDecodeError:
                    continue

        # Если не нашли в паттернах, пробуем весь ответ
        try:
            json.loads(response.strip())
            return response.strip()
        except json.JSONDecodeError:
            pass

        return None

    async def _structured_stream_outlines(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[BaseModel],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> BaseModel:
        """
        Настоящий streaming structured output через Outlines с инкрементальным парсингом.
        
        Реализует реальный streaming с валидацией JSON по мере поступления токенов.
        """
        try:
            logger.debug("Начинаем реальный Outlines streaming structured output")
            
            # Создаем инкрементальный JSON парсер
            json_parser = IncrementalJSONParser(response_model)
            
            # Подготавливаем промпт для JSON генерации
            enhanced_messages = self._enhance_messages_for_json(messages, response_model)
            
            headers = {
                "Authorization": f"Bearer {self.config.api_key or 'dummy'}",
                "Content-Type": "application/json"
            }
            
            # Параметры запроса
            request_data = {
                "model": model or self.config.model,
                "messages": enhanced_messages,
                "temperature": temperature or self.config.temperature,
                "max_tokens": max_tokens or self.config.max_tokens,
                "top_p": self.config.top_p,
                "frequency_penalty": self.config.frequency_penalty,
                "presence_penalty": self.config.presence_penalty,
                "stream": True  # Обязательно включаем streaming
            }
            
            logger.debug("Выполняем реальный streaming HTTP запрос для Outlines")
            
            import httpx
            
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{self.config.endpoint}/v1/chat/completions",
                    json=request_data,
                    headers=headers,
                    timeout=httpx.Timeout(
                        connect=self.config.connect_timeout,
                        read=self.config.read_timeout,
                        write=self.config.write_timeout,
                        pool=None
                    )
                ) as response:
                    
                    if response.status_code != 200:
                        error_text = await response.aread()
                        raise Exception(f"HTTP {response.status_code}: {error_text.decode()}")
                    
                    # Обрабатываем streaming ответ с инкрементальной валидацией
                    chunk_count = 0
                    
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        
                        data_str = line[6:]  # Убираем "data: "
                        if data_str.strip() == "[DONE]":
                            break
                        
                        try:
                            chunk = json.loads(data_str)
                            
                            if "choices" in chunk and chunk["choices"]:
                                choice = chunk["choices"][0]
                                delta = choice.get("delta", {})
                                
                                # Получаем новый контент
                                if "content" in delta and delta["content"]:
                                    new_content = delta["content"]
                                    chunk_count += 1
                                    
                                    logger.debug(f"Streaming chunk #{chunk_count}: '{new_content}'")
                                    
                                    # Добавляем новый контент в парсер
                                    validation_result = json_parser.add_content(new_content)
                                    
                                    # Проверяем результат валидации
                                    if validation_result.is_complete:
                                        logger.info(f"JSON завершен после {chunk_count} chunks")
                                        return validation_result.parsed_object
                                    elif validation_result.is_invalid:
                                        logger.warning(f"Обнаружена ошибка валидации: {validation_result.error}")
                                        # Продолжаем, возможно JSON еще не завершен
                        
                        except json.JSONDecodeError:
                            continue
                    
                    # Если дошли до конца потока, пытаемся финализировать
                    final_result = json_parser.finalize()
                    
                    if final_result.is_complete:
                        logger.info("Outlines streaming structured output успешно завершен")
                        return final_result.parsed_object
                    else:
                        # Fallback к обычной валидации полного контента
                        full_content = json_parser.get_full_content()
                        logger.warning(f"Инкрементальная валидация не удалась, используем fallback. Контент: {full_content[:100]}...")
                        
                        # Пытаемся извлечь JSON из полного контента
                        json_content = self._extract_json_from_response(full_content)
                        
                        if json_content:
                            try:
                                parsed_data = json.loads(json_content)
                                result = response_model.model_validate(parsed_data)
                                logger.info("Fallback валидация успешна")
                                return result
                            except Exception as e:
                                logger.error(f"Fallback валидация не удалась: {e}")
                        
                        raise ValidationError(
                            "Не удалось получить валидный JSON из streaming ответа",
                            context={
                                "chunks_processed": chunk_count,
                                "content_preview": full_content[:200],
                                "parser_state": json_parser.get_state_info()
                            }
                        )
        
        except Exception as e:
            logger.error(f"Ошибка в _structured_stream_outlines: {e}")
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(
                f"Не удалось выполнить реальный Outlines streaming: {e}",
                context={
                    "model": response_model.__name__,
                    "messages_count": len(messages),
                },
                original_error=e,
            )
