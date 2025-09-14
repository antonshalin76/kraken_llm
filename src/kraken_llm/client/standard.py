"""
Стандартный LLM клиент для Kraken фреймворка.

Этот модуль содержит StandardLLMClient - основную реализацию LLM клиента
для обычных запросов без streaming и structured output. Поддерживает
полный набор возможностей AsyncOpenAI API включая function calling и tool calling.
"""

import json
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from openai.types.chat import ChatCompletion, ChatCompletionMessage
from pydantic import BaseModel

from .base import BaseLLMClient
from ..exceptions.api import APIError
from ..exceptions.validation import ValidationError
from ..utils.logging import get_logger

logger = get_logger(__name__)


class StandardLLMClient(BaseLLMClient):
    """
    Стандартный LLM клиент для обычных запросов.
    
    Реализует полную поддержку AsyncOpenAI API для:
    - Обычных chat completion запросов
    - Function calling с автоматическим выполнением функций
    - Tool calling с поддержкой параллельных вызовов
    - Валидации параметров и обработки ошибок
    
    Не поддерживает streaming и structured output - для этого используйте
    соответствующие специализированные клиенты.
    """
    
    def __init__(self, config):
        """
        Инициализация стандартного LLM клиента.
        
        Args:
            config: Конфигурация клиента
        """
        super().__init__(config)
        
        # Реестры для function и tool calling
        self._function_registry = {}
        self._tool_registry = {}
        
        logger.info(f"StandardLLMClient инициализирован с моделью: {config.model}")
    
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
        **kwargs
    ) -> str:
        """
        Выполняет обычный chat completion запрос через AsyncOpenAI.
        
        Args:
            messages: Список сообщений чата в формате OpenAI
            model: Переопределение модели (опционально)
            temperature: Переопределение температуры (опционально)
            max_tokens: Переопределение максимального количества токенов (опционально)
            stream: Переопределение режима потока (должно быть False или None)
            functions: Список функций для function calling
            function_call: Управление вызовом функций ("none", "auto", {"name": "func"})
            tools: Список инструментов для tool calling
            tool_choice: Управление выбором инструментов ("none", "auto", "required", {"type": "function", "function": {"name": "tool"}})
            **kwargs: Дополнительные параметры для AsyncOpenAI
            
        Returns:
            Ответ модели в виде строки
            
        Raises:
            ValidationError: При некорректных параметрах
            APIError: При ошибках API
            NetworkError: При сетевых ошибках
        """
        logger.info(f"Выполнение chat_completion с {len(messages)} сообщениями")
        
        # Валидация параметров
        self._validate_chat_completion_params(
            messages, stream, functions, function_call, tools, tool_choice
        )
        
        try:
            # Подготовка параметров для AsyncOpenAI
            params = self._prepare_openai_params(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,  # Стандартный клиент не поддерживает streaming
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
            
            logger.debug(f"Параметры запроса: {list(params.keys())}")
            
            # Проверяем, поддерживает ли сервер обычные (не streaming) запросы
            if await self._server_supports_non_streaming():
                # Выполнение запроса через AsyncOpenAI
                response: ChatCompletion = await self.openai_client.chat.completions.create(**params)
                
                logger.debug(f"Получен ответ от API: {response.id}")
                
                # Обработка ответа
                return await self._process_chat_completion_response(response, messages)
            else:
                # Сервер поддерживает только streaming, используем прямой HTTP запрос
                return await self._chat_completion_via_streaming(params, messages)
            
        except Exception as e:
            logger.error(f"Ошибка в chat_completion: {e}")
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
    
    async def _chat_completion_via_streaming(
        self, 
        params: Dict[str, Any], 
        original_messages: List[Dict[str, str]]
    ) -> str:
        """
        Выполняет chat completion через streaming и агрегирует результат.
        
        Используется для серверов, которые поддерживают только streaming режим.
        """
        import httpx
        import json
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key or 'dummy'}",
            "Content-Type": "application/json"
        }
        
        # Убираем stream параметр, так как сервер всегда возвращает streaming
        streaming_params = params.copy()
        streaming_params.pop("stream", None)
        
        logger.debug("Выполняем запрос через streaming агрегацию")
        
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
                    raise Exception(f"HTTP {response.status_code}: {error_text.decode()}")
                
                # Агрегируем streaming ответ
                content = ""
                function_call = None
                tool_calls = []
                
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
                            
                            # Обрабатываем function calls
                            if "function_call" in delta and delta["function_call"]:
                                if function_call is None:
                                    function_call = {"name": "", "arguments": ""}
                                
                                fc = delta["function_call"]
                                if "name" in fc:
                                    function_call["name"] += fc["name"]
                                if "arguments" in fc:
                                    function_call["arguments"] += fc["arguments"]
                            
                            # Обрабатываем tool calls
                            if "tool_calls" in delta and delta["tool_calls"]:
                                for tool_call in delta["tool_calls"]:
                                    # Расширяем список tool_calls при необходимости
                                    while len(tool_calls) <= tool_call.get("index", 0):
                                        tool_calls.append({
                                            "id": "",
                                            "type": "function",
                                            "function": {"name": "", "arguments": ""}
                                        })
                                    
                                    idx = tool_call.get("index", 0)
                                    if "id" in tool_call:
                                        tool_calls[idx]["id"] += tool_call["id"]
                                    if "function" in tool_call:
                                        func = tool_call["function"]
                                        if "name" in func:
                                            tool_calls[idx]["function"]["name"] += func["name"]
                                        if "arguments" in func:
                                            tool_calls[idx]["function"]["arguments"] += func["arguments"]
                    
                    except json.JSONDecodeError:
                        continue
                
                # Обрабатываем результат
                if function_call and function_call["name"]:
                    # Выполняем function call
                    return await self._handle_function_call(function_call, original_messages)
                elif tool_calls:
                    # Выполняем tool calls
                    return await self._handle_tool_calls(tool_calls, original_messages)
                else:
                    # Возвращаем обычный контент
                    return content.strip()
    
    def _validate_chat_completion_params(
        self,
        messages: List[Dict[str, str]],
        stream: Optional[bool],
        functions: Optional[List[Dict]],
        function_call: Optional[Union[str, Dict]],
        tools: Optional[List[Dict]],
        tool_choice: Optional[Union[str, Dict]]
    ) -> None:
        """
        Валидация параметров chat completion.
        
        Args:
            messages: Список сообщений
            stream: Режим потока
            functions: Список функций
            function_call: Управление вызовом функций
            tools: Список инструментов
            tool_choice: Управление выбором инструментов
            
        Raises:
            ValidationError: При некорректных параметрах
        """
        if not messages:
            raise ValidationError("Список сообщений не может быть пустым")
        
        if stream is True:
            raise ValidationError(
                "StandardLLMClient не поддерживает streaming. "
                "Используйте StreamingLLMClient для потоковых запросов."
            )
        
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
    
    async def _process_chat_completion_response(
        self,
        response: ChatCompletion,
        original_messages: List[Dict[str, str]]
    ) -> str:
        """
        Обработка ответа от chat completion API.
        
        Args:
            response: Ответ от AsyncOpenAI
            original_messages: Исходные сообщения для контекста
            
        Returns:
            Обработанный ответ в виде строки
        """
        if not response.choices:
            raise APIError(
                message="Ответ API не содержит choices",
                status_code=500,
                response_data={"response_id": response.id}
            )
        
        choice = response.choices[0]
        message = choice.message
        
        logger.debug(f"Обработка ответа: finish_reason={choice.finish_reason}")
        
        # Обработка function calling
        if message.function_call:
            logger.info(f"Обнаружен function call: {message.function_call.name}")
            return await self._handle_function_call(message, original_messages)
        
        # Обработка tool calling
        if message.tool_calls:
            logger.info(f"Обнаружено {len(message.tool_calls)} tool calls")
            return await self._handle_tool_calls(message, original_messages)
        
        # Обычный текстовый ответ
        if message.content:
            logger.debug(f"Получен текстовый ответ длиной {len(message.content)} символов")
            return message.content
        
        # Пустой ответ
        logger.warning("Получен пустой ответ от модели")
        return ""
    
    async def _handle_function_call(
        self,
        message_or_function_call,
        original_messages: List[Dict[str, str]]
    ) -> str:
        """
        Обработка function call от модели.
        
        Args:
            message_or_function_call: Сообщение с function call или словарь function_call
            original_messages: Исходные сообщения
            
        Returns:
            Результат выполнения функции или ошибка
        """
        # Определяем, что нам передали - message объект или function_call словарь
        if isinstance(message_or_function_call, dict):
            # Это function_call словарь из streaming ответа
            function_call = message_or_function_call
            function_name = function_call["name"]
            function_arguments = function_call["arguments"]
        else:
            # Это message объект из обычного ответа
            function_call = message_or_function_call.function_call
            function_name = function_call.name
            function_arguments = function_call.arguments
        
        logger.info(f"Выполнение function call: {function_name}")
        
        try:
            # Парсинг аргументов функции
            if function_arguments:
                try:
                    arguments = json.loads(function_arguments)
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
                    return str(result)
                except Exception as e:
                    logger.error(f"Ошибка выполнения функции {function_name}: {e}")
                    return f"Ошибка выполнения функции {function_name}: {e}"
            else:
                logger.warning(f"Функция {function_name} не зарегистрирована")
                return f"Ошибка: функция {function_name} не найдена"
                
        except Exception as e:
            logger.error(f"Общая ошибка при обработке function call: {e}")
            return f"Ошибка обработки function call: {e}"
    
    async def _handle_tool_calls(
        self,
        message: ChatCompletionMessage,
        original_messages: List[Dict[str, str]]
    ) -> str:
        """
        Обработка tool calls от модели с поддержкой параллельного выполнения.
        
        Args:
            message: Сообщение с tool calls
            original_messages: Исходные сообщения
            
        Returns:
            Результаты выполнения инструментов
        """
        tool_calls = message.tool_calls
        logger.info(f"Выполнение {len(tool_calls)} tool calls")
        
        results = []
        
        # Выполнение всех tool calls (может быть параллельно в будущем)
        for tool_call in tool_calls:
            try:
                if tool_call.type != "function":
                    logger.warning(f"Неподдерживаемый тип tool call: {tool_call.type}")
                    results.append(f"Ошибка: неподдерживаемый тип tool call - {tool_call.type}")
                    continue
                
                function = tool_call.function
                function_name = function.name
                
                logger.debug(f"Выполнение tool call: {function_name} (id: {tool_call.id})")
                
                # Парсинг аргументов
                if function.arguments:
                    try:
                        arguments = json.loads(function.arguments)
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
        
        # Объединение результатов
        return "\n".join(results)
    
    async def _execute_function(self, function, arguments: Dict[str, Any]) -> Any:
        """
        Выполнение зарегистрированной функции.
        
        Args:
            function: Функция для выполнения
            arguments: Аргументы функции
            
        Returns:
            Результат выполнения функции
        """
        import asyncio
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
    
    # Методы, которые не поддерживаются в стандартном клиенте
    
    async def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Fallback streaming через обычный запрос для совместимости.
        
        Для полноценного streaming используйте StreamingLLMClient.
        
        Args:
            messages: Список сообщений
            **kwargs: Дополнительные параметры
            
        Yields:
            str: Полный ответ одним чанком
        """
        logger.warning(
            "StandardLLMClient использует fallback streaming. "
            "Для настоящего streaming используйте StreamingLLMClient."
        )
        
        # Получаем полный ответ и возвращаем его как один чанк
        response = await self.chat_completion(messages, **kwargs)
        
        # Эмулируем streaming формат OpenAI
        try:
            from openai.types.chat import ChatCompletionChunk
            from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
            
            # Создаем chunk в формате OpenAI
            chunk = ChatCompletionChunk(
                id="fallback-stream",
                choices=[
                    Choice(
                        index=0,
                        delta=ChoiceDelta(content=str(response)),
                        finish_reason="stop"
                    )
                ],
                created=int(time.time()),
                model=self.config.model,
                object="chat.completion.chunk"
            )
            
            yield chunk
        except ImportError:
            # Fallback для старых версий OpenAI
            logger.warning("Не удалось импортировать ChatCompletionChunk, возвращаем простой текст")
            yield str(response)
    
    async def chat_completion_structured(
        self,
        messages: List[Dict[str, str]],
        response_model: BaseModel,
        **kwargs
    ) -> BaseModel:
        """
        Structured output не поддерживается в стандартном клиенте.
        
        Raises:
            NotImplementedError: Всегда, так как не поддерживается
        """
        raise NotImplementedError(
            "StandardLLMClient не поддерживает structured output. "
            "Используйте StructuredLLMClient для структурированных ответов."
        )
    

    
    async def _handle_tool_calls(
        self, 
        tool_calls: List[Dict], 
        original_messages: List[Dict[str, str]]
    ) -> str:
        """Обрабатывает tool calls и возвращает результат"""
        from ..tools import default_tool_registry
        from ..tools.tools import ToolCall
        
        logger.info(f"Обрабатываем {len(tool_calls)} tool calls")
        
        # Преобразуем в формат ToolCall
        parsed_tool_calls = []
        for tool_call in tool_calls:
            parsed_tool_calls.append(ToolCall(
                id=tool_call["id"],
                type=tool_call["type"],
                function=tool_call["function"]
            ))
        
        # Выполняем инструменты параллельно
        results = await default_tool_registry.execute_parallel_tools(parsed_tool_calls)
        
        # Формируем новые сообщения
        new_messages = original_messages + [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls
            }
        ]
        
        # Добавляем результаты выполнения инструментов
        for result in results:
            new_messages.append({
                "role": "tool",
                "tool_call_id": result.tool_call_id,
                "content": str(result.result) if result.success else result.error
            })
        
        # Рекурсивно вызываем chat_completion для получения финального ответа
        return await self.chat_completion(new_messages)