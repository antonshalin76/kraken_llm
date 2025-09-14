"""
Обработчик Server-Sent Events (SSE) для потоковых операций Kraken фреймворка.

Этот модуль содержит StreamHandler - специализированный класс для обработки
потоковых ответов от LLM API, включая парсинг SSE событий, агрегацию контента
и обработку различных типов потоковых данных.
"""

import json
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from ..exceptions.api import APIError
from ..exceptions.validation import ValidationError
from ..utils.logging import get_logger

logger = get_logger(__name__)


class StreamHandler:
    """
    Обработчик потоковых ответов Server-Sent Events.
    
    Предоставляет функциональность для:
    - Парсинга SSE событий и извлечения данных
    - Агрегации контента из потоковых chunks
    - Обработки function/tool calls в потоковом режиме
    - Валидации и нормализации потоковых данных
    
    Используется StreamingLLMClient для обработки потоков от AsyncOpenAI.
    """
    
    def __init__(self):
        """Инициализация обработчика потоков."""
        logger.debug("Инициализация StreamHandler")
        
        # Счетчики для мониторинга
        self._processed_chunks = 0
        self._content_chunks = 0
        self._function_call_chunks = 0
        self._tool_call_chunks = 0
    
    async def process_stream(
        self,
        response_stream,
        extract_content: bool = True,
        aggregate_function_calls: bool = True,
        aggregate_tool_calls: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Обработка потока SSE событий.
        
        Args:
            response_stream: Поток ответов от API
            extract_content: Извлекать ли текстовый контент
            aggregate_function_calls: Агрегировать ли function calls
            aggregate_tool_calls: Агрегировать ли tool calls
            
        Yields:
            Словари с обработанными данными из потока
            
        Raises:
            APIError: При ошибках обработки потока
            ValidationError: При некорректных данных в потоке
        """
        logger.info("Начало обработки потока SSE")
        
        self._reset_counters()
        
        # Буферы для агрегации
        function_call_buffer = {"name": "", "arguments": ""} if aggregate_function_calls else None
        tool_calls_buffer = {} if aggregate_tool_calls else None
        
        try:
            async for chunk in response_stream:
                self._processed_chunks += 1
                
                # Обработка отдельного chunk
                processed_data = await self._process_chunk(
                    chunk,
                    extract_content=extract_content,
                    function_call_buffer=function_call_buffer,
                    tool_calls_buffer=tool_calls_buffer
                )
                
                if processed_data:
                    yield processed_data
            
            # Финализация агрегированных данных
            if function_call_buffer and function_call_buffer["name"]:
                yield {
                    "type": "function_call_complete",
                    "data": function_call_buffer.copy()
                }
            
            if tool_calls_buffer:
                yield {
                    "type": "tool_calls_complete", 
                    "data": list(tool_calls_buffer.values())
                }
            
            logger.info(f"Обработка потока завершена: {self._processed_chunks} chunks")
            
        except Exception as e:
            logger.error(f"Ошибка обработки потока на chunk #{self._processed_chunks}: {e}")
            raise APIError(
                message=f"Ошибка обработки потока: {e}",
                status_code=500,
                response_data={
                    "processed_chunks": self._processed_chunks,
                    "error_chunk": self._processed_chunks
                }
            )
    
    async def _process_chunk(
        self,
        chunk,
        extract_content: bool = True,
        function_call_buffer: Optional[Dict[str, str]] = None,
        tool_calls_buffer: Optional[Dict[str, Dict]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Обработка отдельного chunk из потока.
        
        Args:
            chunk: Chunk данных от API
            extract_content: Извлекать ли текстовый контент
            function_call_buffer: Буфер для агрегации function calls
            tool_calls_buffer: Буфер для агрегации tool calls
            
        Returns:
            Обработанные данные или None
        """
        try:
            # Проверка наличия choices
            if not hasattr(chunk, 'choices') or not chunk.choices:
                logger.debug(f"Chunk #{self._processed_chunks} не содержит choices")
                return None
            
            choice = chunk.choices[0]
            delta = choice.delta
            
            # Обработка текстового контента
            if extract_content and delta.content:
                self._content_chunks += 1
                logger.debug(f"Извлечен контент: {len(delta.content)} символов")
                
                return {
                    "type": "content",
                    "data": delta.content,
                    "chunk_id": self._processed_chunks
                }
            
            # Обработка function call
            if delta.function_call and function_call_buffer is not None:
                self._function_call_chunks += 1
                
                fc = delta.function_call
                
                if fc.name:
                    function_call_buffer["name"] = fc.name
                    logger.debug(f"Function call name: {fc.name}")
                
                if fc.arguments:
                    function_call_buffer["arguments"] += fc.arguments
                    logger.debug(f"Function call arguments chunk: {len(fc.arguments)} символов")
                
                return {
                    "type": "function_call_chunk",
                    "data": {
                        "name": fc.name if fc.name else None,
                        "arguments_chunk": fc.arguments if fc.arguments else None
                    },
                    "chunk_id": self._processed_chunks
                }
            
            # Обработка tool calls
            if delta.tool_calls and tool_calls_buffer is not None:
                self._tool_call_chunks += 1
                
                for tool_call in delta.tool_calls:
                    call_id = tool_call.id or f"call_{tool_call.index}"
                    
                    # Инициализация буфера для нового tool call
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
                
                return {
                    "type": "tool_call_chunk",
                    "data": {
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "function": {
                                    "name": tc.function.name if tc.function and tc.function.name else None,
                                    "arguments_chunk": tc.function.arguments if tc.function and tc.function.arguments else None
                                }
                            }
                            for tc in delta.tool_calls
                        ]
                    },
                    "chunk_id": self._processed_chunks
                }
            
            # Обработка завершения потока
            if choice.finish_reason:
                logger.debug(f"Поток завершен с причиной: {choice.finish_reason}")
                
                return {
                    "type": "stream_end",
                    "data": {
                        "finish_reason": choice.finish_reason,
                        "usage": getattr(chunk, 'usage', None)
                    },
                    "chunk_id": self._processed_chunks
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Ошибка обработки chunk #{self._processed_chunks}: {e}")
            raise ValidationError(
                f"Не удалось обработать chunk #{self._processed_chunks}: {e}",
                context={
                    "chunk_id": self._processed_chunks,
                    "chunk_type": type(chunk).__name__
                },
                original_error=e
            )
    
    def _parse_sse_line(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Парсинг строки Server-Sent Events.
        
        Args:
            line: Строка SSE для парсинга
            
        Returns:
            Словарь с данными события или None
        """
        line = line.strip()
        
        if not line or line.startswith(':'):
            # Пустая строка или комментарий
            return None
        
        if line == 'data: [DONE]':
            # Завершение потока
            return {"type": "done"}
        
        if line.startswith('data: '):
            # Данные события
            data_str = line[6:]  # Убираем 'data: '
            
            try:
                data = json.loads(data_str)
                return {"type": "data", "content": data}
            except json.JSONDecodeError as e:
                logger.warning(f"Не удалось распарсить JSON в SSE: {e}")
                return {"type": "raw_data", "content": data_str}
        
        if ':' in line:
            # Другие поля SSE (event, id, retry)
            field, value = line.split(':', 1)
            return {"type": "field", "field": field.strip(), "value": value.strip()}
        
        return None
    
    def _extract_content(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Извлечение текстового контента из данных chunk.
        
        Args:
            data: Данные chunk
            
        Returns:
            Извлеченный контент или None
        """
        try:
            # Стандартная структура OpenAI streaming response
            if isinstance(data, dict):
                choices = data.get('choices', [])
                if choices and len(choices) > 0:
                    delta = choices[0].get('delta', {})
                    content = delta.get('content')
                    
                    if content:
                        logger.debug(f"Извлечен контент: {len(content)} символов")
                        return content
            
            return None
            
        except Exception as e:
            logger.warning(f"Ошибка извлечения контента: {e}")
            return None
    
    def _reset_counters(self) -> None:
        """Сброс счетчиков для новой обработки потока."""
        self._processed_chunks = 0
        self._content_chunks = 0
        self._function_call_chunks = 0
        self._tool_call_chunks = 0
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Получение статистики обработки потока.
        
        Returns:
            Словарь со статистикой
        """
        return {
            "processed_chunks": self._processed_chunks,
            "content_chunks": self._content_chunks,
            "function_call_chunks": self._function_call_chunks,
            "tool_call_chunks": self._tool_call_chunks
        }


class StreamAggregator:
    """
    Агрегатор для сбора и объединения данных из потока.
    
    Используется для накопления частичных данных из потока
    и формирования полных объектов (например, полного ответа,
    завершенных function calls, tool calls).
    """
    
    def __init__(self):
        """Инициализация агрегатора."""
        logger.debug("Инициализация StreamAggregator")
        
        self._content_parts = []
        self._function_calls = {}
        self._tool_calls = {}
        self._metadata = {}
    
    def add_content(self, content: str) -> None:
        """
        Добавление части текстового контента.
        
        Args:
            content: Часть контента для добавления
        """
        if content:
            self._content_parts.append(content)
            logger.debug(f"Добавлена часть контента: {len(content)} символов")
    
    def add_function_call_chunk(
        self,
        name: Optional[str] = None,
        arguments_chunk: Optional[str] = None
    ) -> None:
        """
        Добавление части function call.
        
        Args:
            name: Имя функции (если есть в этом chunk)
            arguments_chunk: Часть аргументов функции
        """
        if name:
            self._function_calls["name"] = name
            logger.debug(f"Установлено имя функции: {name}")
        
        if arguments_chunk:
            if "arguments" not in self._function_calls:
                self._function_calls["arguments"] = ""
            self._function_calls["arguments"] += arguments_chunk
            logger.debug(f"Добавлена часть аргументов функции: {len(arguments_chunk)} символов")
    
    def add_tool_call_chunk(
        self,
        call_id: str,
        name: Optional[str] = None,
        arguments_chunk: Optional[str] = None
    ) -> None:
        """
        Добавление части tool call.
        
        Args:
            call_id: ID вызова инструмента
            name: Имя инструмента (если есть в этом chunk)
            arguments_chunk: Часть аргументов инструмента
        """
        if call_id not in self._tool_calls:
            self._tool_calls[call_id] = {
                "id": call_id,
                "type": "function",
                "function": {"name": "", "arguments": ""}
            }
        
        if name:
            self._tool_calls[call_id]["function"]["name"] = name
            logger.debug(f"Установлено имя инструмента {call_id}: {name}")
        
        if arguments_chunk:
            self._tool_calls[call_id]["function"]["arguments"] += arguments_chunk
            logger.debug(f"Добавлена часть аргументов инструмента {call_id}: {len(arguments_chunk)} символов")
    
    def add_metadata(self, key: str, value: Any) -> None:
        """
        Добавление метаданных.
        
        Args:
            key: Ключ метаданных
            value: Значение метаданных
        """
        self._metadata[key] = value
        logger.debug(f"Добавлены метаданные: {key}")
    
    def get_aggregated_content(self) -> str:
        """
        Получение агрегированного текстового контента.
        
        Returns:
            Полный текстовый контент
        """
        content = "".join(self._content_parts)
        logger.debug(f"Агрегирован контент: {len(content)} символов из {len(self._content_parts)} частей")
        return content
    
    def get_function_call(self) -> Optional[Dict[str, str]]:
        """
        Получение агрегированного function call.
        
        Returns:
            Данные function call или None
        """
        if self._function_calls.get("name"):
            logger.debug(f"Агрегирован function call: {self._function_calls['name']}")
            return self._function_calls.copy()
        return None
    
    def get_tool_calls(self) -> List[Dict[str, Any]]:
        """
        Получение агрегированных tool calls.
        
        Returns:
            Список данных tool calls
        """
        tool_calls = [
            call_data for call_data in self._tool_calls.values()
            if call_data["function"]["name"]
        ]
        
        if tool_calls:
            logger.debug(f"Агрегировано {len(tool_calls)} tool calls")
        
        return tool_calls
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Получение метаданных.
        
        Returns:
            Словарь метаданных
        """
        return self._metadata.copy()
    
    def clear(self) -> None:
        """Очистка агрегатора для повторного использования."""
        logger.debug("Очистка StreamAggregator")
        
        self._content_parts.clear()
        self._function_calls.clear()
        self._tool_calls.clear()
        self._metadata.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Получение сводки агрегированных данных.
        
        Returns:
            Сводка с информацией о собранных данных
        """
        return {
            "content_parts": len(self._content_parts),
            "content_length": len(self.get_aggregated_content()),
            "has_function_call": bool(self._function_calls.get("name")),
            "tool_calls_count": len(self.get_tool_calls()),
            "metadata_keys": list(self._metadata.keys())
        }