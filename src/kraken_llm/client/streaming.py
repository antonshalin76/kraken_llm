"""
Потоковый LLM клиент для Kraken фреймворка.

Этот модуль содержит StreamingLLMClient - специализированную реализацию LLM клиента
для потоковых запросов с поддержкой Server-Sent Events (SSE). Поддерживает
обработку потоковых chunks, агрегацию контента и function/tool calling в потоковом режиме.
"""

import asyncio
import json
import numpy as np
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from openai.types.chat import ChatCompletionChunk
from pydantic import BaseModel

from .base import BaseLLMClient
from ..exceptions.api import APIError
from ..exceptions.validation import ValidationError
from ..utils.logging import get_logger

logger = get_logger(__name__)


class StreamingLLMClient(BaseLLMClient):
    """
    Потоковый LLM клиент для streaming запросов.
    
    Реализует полную поддержку AsyncOpenAI streaming API для:
    - Потоковых chat completion запросов с real-time ответами
    - Function calling с агрегацией данных из потока
    - Tool calling с поддержкой параллельных вызовов в потоке
    - Обработки Server-Sent Events и потоковых chunks
    
    Не поддерживает structured output - для этого используйте
    StructuredLLMClient с поддержкой streaming structured output.
    """
    
    def __init__(self, config):
        """
        Инициализация потокового LLM клиента.
        
        Args:
            config: Конфигурация клиента
        """
        super().__init__(config)
        
        # Реестры для function и tool calling
        self._function_registry = {}
        self._tool_registry = {}
        
        logger.info(f"StreamingLLMClient инициализирован с моделью: {config.model}")
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = None,
        functions: Optional[List[Dict]] = None,
        function_call: Optional[Union[str, Dict]] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[Union[str, Dict]] = None,
        include_confidence: Optional[bool] = None,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """
        Выполняет chat completion с автоматическим определением режима потока.
        
        Если stream=True или не указан, выполняется потоковый запрос с агрегацией.
        Если stream=False, выбрасывается исключение с рекомендацией использовать StandardLLMClient.
        
        Args:
            messages: Список сообщений чата в формате OpenAI
            model: Переопределение модели (опционально)
            temperature: Переопределение температуры (опционально)
            max_tokens: Переопределение максимального количества токенов (опционально)
            stream: Режим потока (True по умолчанию для этого клиента)
            functions: Список функций для function calling
            function_call: Управление вызовом функций
            tools: Список инструментов для tool calling
            tool_choice: Управление выбором инструментов
            **kwargs: Дополнительные параметры для AsyncOpenAI
            
        Returns:
            Полный ответ модели в виде строки (агрегированный из потока)
            
        Raises:
            ValidationError: При некорректных параметрах
            APIError: При ошибках API
            NetworkError: При сетевых ошибках
        """
        logger.info(f"Выполнение chat_completion в потоковом режиме с {len(messages)} сообщениями")
        
        # Валидация параметров
        if stream is False:
            raise ValidationError(
                "StreamingLLMClient предназначен для потоковых запросов. "
                "Для обычных запросов используйте StandardLLMClient."
            )
        
        # Если требуется уверенность, выполняем прямой проход по потоку со сбором logprobs и метрик
        if include_confidence:
            params = self._prepare_openai_params(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs,
            )
            params.setdefault("logprobs", True)
            if "top_logprobs" not in params:
                top_lp = kwargs.get("top_logprobs", getattr(self.config, "top_logprobs", None))
                if top_lp is None:
                    top_lp = 5
                params["top_logprobs"] = top_lp

            stream_gen = await self.openai_client.chat.completions.create(**params)

            content = ""
            all_confidences: List[float] = []
            token_confidences: List[Dict[str, Any]] = []

            try:
                async for chunk in stream_gen:
                    if chunk.choices:
                        ch = chunk.choices[0]
                        if getattr(ch, "delta", None) and getattr(ch.delta, "content", None):
                            content += ch.delta.content
                        # logprobs per token
                        if getattr(ch, "logprobs", None) and getattr(ch.logprobs, "content", None):
                            for item in ch.logprobs.content:
                                # item: ChatCompletionTokenLogprob
                                lp = getattr(item, "logprob", None)
                                if lp is not None:
                                    p = float(np.exp(lp))
                                    all_confidences.append(p)
                                    # альтернативы
                                    alts = []
                                    top = getattr(item, "top_logprobs", None) or []
                                    for alt in top:
                                        alt_lp = getattr(alt, "logprob", None)
                                        if alt_lp is not None:
                                            alts.append(
                                                {
                                                    "token": getattr(alt, "token", "") or "",
                                                    "confidence": float(np.exp(alt_lp)),
                                                    "logprob": alt_lp,
                                                }
                                            )
                                    token_confidences.append(
                                        {
                                            "token": getattr(item, "token", "") or "",
                                            "confidence": p,
                                            "confidence_label": "",  # добавим ниже
                                            "logprob": lp,
                                            "alternatives": alts,
                                        }
                                    )
            except Exception as stream_err:
                if getattr(self.config, "suppress_stream_warnings", False):
                    logger.debug(f"Поток OpenAI прерван в StreamingLLMClient: {stream_err}; возвращаем накопленный контент")
                else:
                    logger.warning(f"Поток OpenAI прерван в StreamingLLMClient: {stream_err}; возвращаем накопленный контент")

            # Если удалось собрать токенные метрики — возвращаем их
            if all_confidences:
                avg = float(np.mean(all_confidences))
                for t in token_confidences:
                    t["confidence_label"] = (
                        "Очень высокая" if t["confidence"] >= 0.9 else
                        "Высокая" if t["confidence"] >= 0.7 else
                        "Средняя" if t["confidence"] >= 0.5 else
                        "Низкая" if t["confidence"] >= 0.3 else
                        "Очень низкая"
                    )
                return {
                    "text": content.strip(),
                    "confidence": avg,
                    "confidence_label": (
                        "Очень высокая" if avg >= 0.9 else
                        "Высокая" if avg >= 0.7 else
                        "Средняя" if avg >= 0.5 else
                        "Низкая" if avg >= 0.3 else
                        "Очень низкая"
                    ),
                    "token_confidences": token_confidences,
                    "total_tokens": len(token_confidences),
                }

            # Если контент частично есть, но метрик нет — возвращаем без logprobs
            if content.strip():
                return {
                    "text": content.strip(),
                    "confidence": 0.5,
                    "confidence_label": "Нет logprobs",
                    "token_confidences": [],
                    "total_tokens": 0,
                }

            # Полный фоллбек: не получили ни контента, ни метрик — пробуем non-stream запрос
            try:
                ns_params = params.copy()
                ns_params["stream"] = False
                # Обеспечим logprobs для расчета уверенности
                ns_params.setdefault("logprobs", True)
                if "top_logprobs" not in ns_params:
                    ns_params["top_logprobs"] = 5
                response = await self.openai_client.chat.completions.create(**ns_params)
                if not response.choices:
                    return {"text": "", "confidence": 0.0, "confidence_label": "Ошибка"}
                choice = response.choices[0]
                msg_text = choice.message.content or ""
                from ..confidence.metrics import confidence_from_chat_logprobs
                metrics = confidence_from_chat_logprobs(getattr(choice, "logprobs", None))
                return {
                    "text": msg_text.strip(),
                    "confidence": metrics.get("average_confidence", 0.5),
                    "confidence_label": metrics.get("confidence_label", "Нет logprobs"),
                    "confidence_metrics": metrics,
                }
            except Exception:
                # Возвращаем пустой безопасный ответ
                return {
                    "text": "",
                    "confidence": 0.5,
                    "confidence_label": "Нет logprobs",
                    "token_confidences": [],
                    "total_tokens": 0,
                }

        # Агрегация потокового ответа по умолчанию
        chunks = []
        async for chunk in self.chat_completion_stream(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            functions=functions,
            function_call=function_call,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs
        ):
            chunks.append(chunk)
        
        result = "".join(chunks)
        logger.debug(f"Агрегирован потоковый ответ длиной {len(result)} символов")
        return result
    
    async def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        functions: Optional[List[Dict]] = None,
        function_call: Optional[Union[str, Dict]] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[Union[str, Dict]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Потоковый chat completion через AsyncOpenAI.
        
        Args:
            messages: Список сообщений чата
            model: Переопределение модели (опционально)
            temperature: Переопределение температуры (опционально)
            max_tokens: Переопределение максимального количества токенов (опционально)
            functions: Список функций для function calling
            function_call: Управление вызовом функций
            tools: Список инструментов для tool calling
            tool_choice: Управление выбором инструментов
            **kwargs: Дополнительные параметры
            
        Yields:
            Части ответа модели по мере генерации
            
        Raises:
            ValidationError: При некорректных параметрах
            APIError: При ошибках API
            NetworkError: При сетевых ошибках
        """
        logger.info(f"Начало потокового chat_completion с {len(messages)} сообщениями")
        
        # Валидация параметров
        self._validate_stream_params(messages, functions, function_call, tools, tool_choice)
        
        try:
            # Подготовка параметров для AsyncOpenAI
            params = self._prepare_openai_params(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,  # Принудительно включаем streaming
                **kwargs
            )
            
            # Добавление function calling параметров
            if functions:
                params["functions"] = functions
                if function_call:
                    params["function_call"] = function_call
            
            # Добавление tool calling параметров
            if tools:
                params["tools"] = tools
                if tool_choice:
                    params["tool_choice"] = tool_choice
            
            logger.debug(f"Параметры потокового запроса: {list(params.keys())}")
            
            # Создание потокового запроса через AsyncOpenAI
            stream = await self.openai_client.chat.completions.create(**params)
            
            logger.debug("Потоковый запрос создан, начинаем обработку chunks")
            
            # Обработка потока
            async for chunk_content in self._process_stream(stream, messages):
                if chunk_content:
                    yield chunk_content
            
            logger.debug("Потоковый запрос завершен")
            
        except Exception as e:
            logger.error(f"Ошибка в chat_completion_stream: {e}")
            await self._handle_openai_error(e)
    
    def _validate_stream_params(
        self,
        messages: List[Dict[str, str]],
        functions: Optional[List[Dict]],
        function_call: Optional[Union[str, Dict]],
        tools: Optional[List[Dict]],
        tool_choice: Optional[Union[str, Dict]]
    ) -> None:
        """
        Валидация параметров потокового запроса.
        
        Args:
            messages: Список сообщений
            functions: Список функций
            function_call: Управление вызовом функций
            tools: Список инструментов
            tool_choice: Управление выбором инструментов
            
        Raises:
            ValidationError: При некорректных параметрах
        """
        if not messages:
            raise ValidationError("Список сообщений не может быть пустым")
        
        # Валидация function calling параметров
        if function_call and not functions:
            raise ValidationError(
                "function_call указан, но functions не предоставлены"
            )
        
        # Валидация tool calling параметров
        if tool_choice and not tools:
            raise ValidationError(
                "tool_choice указан, но tools не предоставлены"
            )
        
        # Нельзя использовать functions и tools одновременно
        if functions and tools:
            raise ValidationError(
                "Нельзя использовать functions и tools одновременно. "
                "Используйте либо functions (устаревший), либо tools (рекомендуемый)."
            )
        
        # Валидация структуры сообщений
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValidationError(f"Сообщение {i} должно быть словарем")
            
            if "role" not in message or "content" not in message:
                raise ValidationError(
                    f"Сообщение {i} должно содержать 'role' и 'content'"
                )
            
            if message["role"] not in ["system", "user", "assistant", "function", "tool"]:
                raise ValidationError(
                    f"Некорректная роль в сообщении {i}: {message['role']}"
                )
    
    async def _process_stream(
        self,
        stream,
        original_messages: List[Dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        """
        Обработка потока chunks от AsyncOpenAI.
        
        Args:
            stream: Поток chunks от AsyncOpenAI
            original_messages: Исходные сообщения для контекста
            
        Yields:
            Обработанные части контента
        """
        # Буферы для агрегации function/tool calls
        function_call_buffer = {"name": "", "arguments": ""}
        tool_calls_buffer = {}
        
        chunk_count = 0
        
        try:
            async for chunk in stream:
                chunk_count += 1
                logger.debug(f"Обработка chunk #{chunk_count}")
                
                # Обработка отдельного chunk
                content = await self._process_stream_chunk(
                    chunk, 
                    function_call_buffer, 
                    tool_calls_buffer,
                    original_messages
                )
                
                if content:
                    yield content
                    
        except Exception as e:
            logger.error(f"Ошибка обработки потока на chunk #{chunk_count}: {e}")
            raise
        
        logger.debug(f"Обработано {chunk_count} chunks")
        
        # Обработка завершенных function/tool calls
        if function_call_buffer["name"]:
            logger.info("Обработка завершенного function call из потока")
            result = await self._execute_aggregated_function_call(
                function_call_buffer, original_messages
            )
            if result:
                yield result
        
        if tool_calls_buffer:
            logger.info(f"Обработка {len(tool_calls_buffer)} завершенных tool calls из потока")
            result = await self._execute_aggregated_tool_calls(
                tool_calls_buffer, original_messages
            )
            if result:
                yield result
    
    async def _process_stream_chunk(
        self,
        chunk: ChatCompletionChunk,
        function_call_buffer: Dict[str, str],
        tool_calls_buffer: Dict[str, Dict],
        original_messages: List[Dict[str, str]]
    ) -> Optional[str]:
        """
        Обработка отдельного chunk из потока AsyncOpenAI.
        
        Args:
            chunk: Chunk от AsyncOpenAI
            function_call_buffer: Буфер для агрегации function call
            tool_calls_buffer: Буфер для агрегации tool calls
            original_messages: Исходные сообщения
            
        Returns:
            Контент для вывода или None
        """
        if not chunk.choices:
            return None
        
        choice = chunk.choices[0]
        delta = choice.delta
        
        # Обработка обычного контента
        if delta.content:
            logger.debug(f"Получен контент: {len(delta.content)} символов")
            return delta.content
        
        # Обработка function call в потоке
        if delta.function_call:
            fc = delta.function_call
            
            if fc.name:
                function_call_buffer["name"] = fc.name
                logger.debug(f"Function call name: {fc.name}")
            
            if fc.arguments:
                function_call_buffer["arguments"] += fc.arguments
                logger.debug(f"Function call arguments chunk: {len(fc.arguments)} символов")
        
        # Обработка tool calls в потоке
        if delta.tool_calls:
            for tool_call in delta.tool_calls:
                call_id = tool_call.id or f"call_{tool_call.index}"
                
                if call_id not in tool_calls_buffer:
                    tool_calls_buffer[call_id] = {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "", "arguments": ""}
                    }
                
                if tool_call.function:
                    if tool_call.function.name:
                        tool_calls_buffer[call_id]["function"]["name"] = tool_call.function.name
                        logger.debug(f"Tool call name: {tool_call.function.name}")
                    
                    if tool_call.function.arguments:
                        tool_calls_buffer[call_id]["function"]["arguments"] += tool_call.function.arguments
                        logger.debug(f"Tool call arguments chunk: {len(tool_call.function.arguments)} символов")
        
        return None
    
    async def _execute_aggregated_function_call(
        self,
        function_call_data: Dict[str, str],
        original_messages: List[Dict[str, str]]
    ) -> Optional[str]:
        """
        Выполнение агрегированного function call из потока.
        
        Args:
            function_call_data: Агрегированные данные function call
            original_messages: Исходные сообщения
            
        Returns:
            Результат выполнения функции
        """
        function_name = function_call_data["name"]
        arguments_str = function_call_data["arguments"]
        
        if not function_name:
            return None
        
        logger.info(f"Выполнение агрегированного function call: {function_name}")
        
        try:
            # Парсинг аргументов
            if arguments_str:
                try:
                    arguments = json.loads(arguments_str)
                except json.JSONDecodeError as e:
                    logger.error(f"Ошибка парсинга аргументов функции: {e}")
                    return f"Ошибка: некорректные аргументы функции - {e}"
            else:
                arguments = {}
            
            # Выполнение функции
            if function_name in self._function_registry:
                function = self._function_registry[function_name]
                logger.debug(f"Выполнение зарегистрированной функции: {function_name}")
                
                try:
                    result = await self._execute_function(function, arguments)
                    logger.info(f"Функция {function_name} выполнена успешно")
                    return f"\n\nРезультат функции {function_name}: {result}"
                except Exception as e:
                    logger.error(f"Ошибка выполнения функции {function_name}: {e}")
                    return f"\n\nОшибка выполнения функции {function_name}: {e}"
            else:
                logger.warning(f"Функция {function_name} не зарегистрирована")
                return f"\n\nОшибка: функция {function_name} не найдена"
                
        except Exception as e:
            logger.error(f"Общая ошибка обработки function call: {e}")
            return f"\n\nОшибка обработки function call: {e}"
    
    async def _execute_aggregated_tool_calls(
        self,
        tool_calls_data: Dict[str, Dict],
        original_messages: List[Dict[str, str]]
    ) -> Optional[str]:
        """
        Выполнение агрегированных tool calls из потока.
        
        Args:
            tool_calls_data: Агрегированные данные tool calls
            original_messages: Исходные сообщения
            
        Returns:
            Результаты выполнения инструментов
        """
        if not tool_calls_data:
            return None
        
        logger.info(f"Выполнение {len(tool_calls_data)} агрегированных tool calls")
        
        results = []
        
        for call_id, tool_call_data in tool_calls_data.items():
            try:
                function_data = tool_call_data["function"]
                function_name = function_data["name"]
                arguments_str = function_data["arguments"]
                
                if not function_name:
                    continue
                
                logger.debug(f"Выполнение tool call: {function_name} (id: {call_id})")
                
                # Парсинг аргументов
                if arguments_str:
                    try:
                        arguments = json.loads(arguments_str)
                    except json.JSONDecodeError as e:
                        logger.error(f"Ошибка парсинга аргументов tool call: {e}")
                        results.append(f"Ошибка: некорректные аргументы - {e}")
                        continue
                else:
                    arguments = {}
                
                # Выполнение инструмента
                if function_name in self._tool_registry:
                    tool = self._tool_registry[function_name]
                    logger.debug(f"Выполнение зарегистрированного инструмента: {function_name}")
                    
                    try:
                        result = await self._execute_tool(tool, arguments)
                        results.append(f"Tool {function_name}: {result}")
                        logger.info(f"Инструмент {function_name} выполнен успешно")
                    except Exception as e:
                        logger.error(f"Ошибка выполнения инструмента {function_name}: {e}")
                        results.append(f"Ошибка выполнения инструмента {function_name}: {e}")
                else:
                    logger.warning(f"Инструмент {function_name} не зарегистрирован")
                    results.append(f"Ошибка: инструмент {function_name} не найден")
                    
            except Exception as e:
                logger.error(f"Общая ошибка обработки tool call: {e}")
                results.append(f"Ошибка обработки tool call: {e}")
        
        if results:
            return "\n\n" + "\n".join(results)
        return None
    
    async def _execute_function(self, function, arguments: Dict[str, Any]) -> Any:
        """
        Выполнение зарегистрированной функции.
        
        Args:
            function: Функция для выполнения
            arguments: Аргументы функции
            
        Returns:
            Результат выполнения функции
        """
        import inspect
        
        if inspect.iscoroutinefunction(function):
            return await function(**arguments)
        else:
            # Выполнение синхронной функции в thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: function(**arguments))
    
    async def _execute_tool(self, tool, arguments: Dict[str, Any]) -> Any:
        """
        Выполнение зарегистрированного инструмента.
        
        Args:
            tool: Инструмент для выполнения
            arguments: Аргументы инструмента
            
        Returns:
            Результат выполнения инструмента
        """
        # Инструменты обрабатываются так же, как функции
        return await self._execute_function(tool, arguments)
    
    def register_function(
        self,
        name: str,
        function,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Регистрация функции для function calling.
        
        Args:
            name: Имя функции
            function: Функция для регистрации
            description: Описание функции
            parameters: JSON Schema параметров функции
        """
        logger.info(f"Регистрация функции: {name}")
        
        self._function_registry[name] = function
        
        logger.debug(f"Функция {name} зарегистрирована успешно")
    
    def register_tool(
        self,
        name: str,
        tool,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Регистрация инструмента для tool calling.
        
        Args:
            name: Имя инструмента
            tool: Инструмент для регистрации
            description: Описание инструмента
            parameters: JSON Schema параметров инструмента
        """
        logger.info(f"Регистрация инструмента: {name}")
        
        self._tool_registry[name] = tool
        
        logger.debug(f"Инструмент {name} зарегистрирован успешно")
    
    def get_registered_functions(self) -> List[str]:
        """
        Получение списка зарегистрированных функций.
        
        Returns:
            Список имен зарегистрированных функций
        """
        return list(self._function_registry.keys())
    
    def get_registered_tools(self) -> List[str]:
        """
        Получение списка зарегистрированных инструментов.
        
        Returns:
            Список имен зарегистрированных инструментов
        """
        return list(self._tool_registry.keys())
    
    # Метод, который не поддерживается в потоковом клиенте
    
    async def chat_completion_structured(
        self,
        messages: List[Dict[str, str]],
        response_model: BaseModel,
        **kwargs
    ) -> BaseModel:
        """
        Structured output не поддерживается в потоковом клиенте.
        
        Raises:
            NotImplementedError: Всегда, так как не поддерживается
        """
        raise NotImplementedError(
            "StreamingLLMClient не поддерживает structured output. "
            "Используйте StructuredLLMClient для структурированных ответов."
        )