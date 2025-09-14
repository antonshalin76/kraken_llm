"""
Модели данных для потоковых операций Kraken LLM фреймворка.

Этот модуль содержит Pydantic модели для работы с потоковыми ответами,
включая агрегацию chunks, обработку Server-Sent Events и управление потоками.
"""

from typing import List, Dict, Any, Optional, Union, AsyncGenerator, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum
from datetime import datetime
import asyncio
import json


class StreamEventType(str, Enum):
    """Типы событий в потоке."""
    CHUNK = "chunk"
    START = "start"
    END = "end"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    METADATA = "metadata"


class StreamChunk(BaseModel):
    """
    Chunk потокового ответа.
    
    Представляет один фрагмент данных в потоковом ответе.
    """
    id: str = Field(..., description="Уникальный идентификатор chunk")
    sequence: int = Field(..., description="Порядковый номер chunk в потоке")
    content: Optional[str] = Field(None, description="Содержимое chunk")
    delta: Optional[Dict[str, Any]] = Field(None, description="Изменения в этом chunk")
    finish_reason: Optional[str] = Field(None, description="Причина завершения (только в последнем chunk)")
    timestamp: datetime = Field(default_factory=datetime.now, description="Время создания chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Метаданные chunk")
    
    @field_validator('sequence')
    
    @classmethod
    def validate_sequence(cls, v):
        """Валидация порядкового номера."""
        if v < 0:
            raise ValueError("Порядковый номер не может быть отрицательным")
        
        return v
    
    @property
    def is_final(self) -> bool:
        """Проверка, является ли chunk финальным."""
        return self.finish_reason is not None
    
    @property
    def has_content(self) -> bool:
        """Проверка наличия содержимого."""
        return bool(self.content and self.content.strip())


class StreamEvent(BaseModel):
    """
    Событие в потоке.
    
    Представляет различные типы событий, происходящих в потоке.
    """
    type: StreamEventType = Field(..., description="Тип события")
    data: Optional[Dict[str, Any]] = Field(None, description="Данные события")
    chunk: Optional[StreamChunk] = Field(None, description="Chunk данных (для типа CHUNK)")
    error: Optional[str] = Field(None, description="Сообщение об ошибке (для типа ERROR)")
    timestamp: datetime = Field(default_factory=datetime.now, description="Время события")
    
    @field_validator('chunk')
    
    @classmethod
    def validate_chunk_for_chunk_event(cls, v, info):
        """Валидация chunk для событий типа CHUNK."""
        data = info.data if info.data else {}
        event_type = data.get('type')
        
        if event_type == StreamEventType.CHUNK and v is None:
            raise ValueError("Для события типа CHUNK обязателен chunk")
        
        if event_type != StreamEventType.CHUNK and v is not None:
            raise ValueError("Chunk может быть указан только для события типа CHUNK")
        
        return v
    
    @field_validator('error')
    
    @classmethod
    def validate_error_for_error_event(cls, v, info):
        """Валидация ошибки для событий типа ERROR."""
        data = info.data if info.data else {}
        event_type = data.get('type')
        
        if event_type == StreamEventType.ERROR and not v:
            raise ValueError("Для события типа ERROR обязательно сообщение об ошибке")
        
        return v


class StreamState(str, Enum):
    """Состояния потока."""
    INITIALIZING = "initializing"
    STREAMING = "streaming"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class StreamMetrics(BaseModel):
    """
    Метрики потока.
    
    Содержит статистику о потоковой передаче данных.
    """
    total_chunks: int = Field(default=0, description="Общее количество chunks")
    total_bytes: int = Field(default=0, description="Общее количество байт")
    total_characters: int = Field(default=0, description="Общее количество символов")
    start_time: Optional[datetime] = Field(None, description="Время начала потока")
    end_time: Optional[datetime] = Field(None, description="Время окончания потока")
    duration: Optional[float] = Field(None, description="Длительность потока в секундах")
    average_chunk_size: Optional[float] = Field(None, description="Средний размер chunk")
    chunks_per_second: Optional[float] = Field(None, description="Chunks в секунду")
    
    def update_with_chunk(self, chunk: StreamChunk) -> None:
        """
        Обновление метрик новым chunk.
        
        Args:
            chunk: Новый chunk для учета в метриках
        """
        self.total_chunks += 1
        
        if chunk.content:
            content_bytes = len(chunk.content.encode('utf-8'))
            self.total_bytes += content_bytes
            self.total_characters += len(chunk.content)
        
        # Обновляем время начала при первом chunk
        if self.total_chunks == 1:
            self.start_time = chunk.timestamp
        
        # Обновляем средний размер chunk
        if self.total_chunks > 0:
            self.average_chunk_size = self.total_characters / self.total_chunks
        
        # Обновляем скорость при завершении
        if chunk.is_final and self.start_time:
            self.end_time = chunk.timestamp
            self.duration = (self.end_time - self.start_time).total_seconds()
            
            if self.duration > 0:
                self.chunks_per_second = self.total_chunks / self.duration
    
    @property
    def is_completed(self) -> bool:
        """Проверка завершенности потока."""
        return self.end_time is not None


class StreamBuffer(BaseModel):
    """
    Буфер для агрегации потоковых данных.
    
    Накапливает chunks и предоставляет методы для их обработки.
    """
    chunks: List[StreamChunk] = Field(default_factory=list, description="Накопленные chunks")
    aggregated_content: str = Field(default="", description="Агрегированное содержимое")
    state: StreamState = Field(default=StreamState.INITIALIZING, description="Состояние буфера")
    metrics: StreamMetrics = Field(default_factory=StreamMetrics, description="Метрики потока")
    
    def add_chunk(self, chunk: StreamChunk) -> None:
        """
        Добавление chunk в буфер.
        
        Args:
            chunk: Chunk для добавления
        """
        self.chunks.append(chunk)
        
        if chunk.content:
            self.aggregated_content += chunk.content
        
        # Обновляем метрики
        self.metrics.update_with_chunk(chunk)
        
        # Обновляем состояние
        if self.state == StreamState.INITIALIZING:
            self.state = StreamState.STREAMING
        
        if chunk.is_final:
            self.state = StreamState.COMPLETED
    
    def get_content_since_sequence(self, sequence: int) -> str:
        """
        Получение содержимого с определенной последовательности.
        
        Args:
            sequence: Начальная последовательность
            
        Returns:
            str: Агрегированное содержимое
        """
        content_parts = []
        
        for chunk in self.chunks:
            if chunk.sequence >= sequence and chunk.content:
                content_parts.append(chunk.content)
        
        return "".join(content_parts)
    
    def get_latest_chunks(self, count: int) -> List[StreamChunk]:
        """
        Получение последних N chunks.
        
        Args:
            count: Количество chunks
            
        Returns:
            List[StreamChunk]: Последние chunks
        """
        return self.chunks[-count:] if count > 0 else []
    
    def clear(self) -> None:
        """Очистка буфера."""
        self.chunks.clear()
        self.aggregated_content = ""
        self.state = StreamState.INITIALIZING
        self.metrics = StreamMetrics()
    
    @property
    def is_empty(self) -> bool:
        """Проверка пустоты буфера."""
        return len(self.chunks) == 0
    
    @property
    def is_completed(self) -> bool:
        """Проверка завершенности потока."""
        return self.state == StreamState.COMPLETED
    
    @property
    def last_chunk(self) -> Optional[StreamChunk]:
        """Получение последнего chunk."""
        return self.chunks[-1] if self.chunks else None


class StreamProcessor(BaseModel):
    """
    Процессор потоковых данных.
    
    Обрабатывает потоковые данные и управляет их агрегацией.
    """
    buffer: StreamBuffer = Field(default_factory=StreamBuffer, description="Буфер данных")
    filters: List[str] = Field(default_factory=list, description="Фильтры для обработки")
    transformers: List[str] = Field(default_factory=list, description="Трансформеры данных")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    async def process_chunk(self, chunk: StreamChunk) -> StreamEvent:
        """
        Обработка одного chunk.
        
        Args:
            chunk: Chunk для обработки
            
        Returns:
            StreamEvent: Событие обработки
        """
        # Применяем фильтры
        if not self._should_process_chunk(chunk):
            return StreamEvent(
                type=StreamEventType.METADATA,
                data={"action": "filtered", "chunk_id": chunk.id}
            )
        
        # Применяем трансформации
        processed_chunk = self._transform_chunk(chunk)
        
        # Добавляем в буфер
        self.buffer.add_chunk(processed_chunk)
        
        # Создаем событие
        event_type = StreamEventType.END if processed_chunk.is_final else StreamEventType.CHUNK
        
        return StreamEvent(
            type=event_type,
            chunk=processed_chunk,
            data={
                "buffer_size": len(self.buffer.chunks),
                "total_content_length": len(self.buffer.aggregated_content)
            }
        )
    
    def _should_process_chunk(self, chunk: StreamChunk) -> bool:
        """
        Проверка, нужно ли обрабатывать chunk.
        
        Args:
            chunk: Chunk для проверки
            
        Returns:
            bool: True если нужно обрабатывать
        """
        # Простая реализация - обрабатываем все chunks
        # В реальной реализации здесь могут быть сложные фильтры
        return True
    
    def _transform_chunk(self, chunk: StreamChunk) -> StreamChunk:
        """
        Трансформация chunk.
        
        Args:
            chunk: Исходный chunk
            
        Returns:
            StreamChunk: Трансформированный chunk
        """
        # Простая реализация - возвращаем как есть
        # В реальной реализации здесь могут быть трансформации
        return chunk
    
    def get_aggregated_result(self) -> str:
        """
        Получение агрегированного результата.
        
        Returns:
            str: Полный агрегированный контент
        """
        return self.buffer.aggregated_content
    
    def get_metrics(self) -> StreamMetrics:
        """
        Получение метрик обработки.
        
        Returns:
            StreamMetrics: Метрики потока
        """
        return self.buffer.metrics


class SSEParser(BaseModel):
    """
    Парсер Server-Sent Events.
    
    Парсит SSE данные и конвертирует их в StreamChunk объекты.
    """
    
    @staticmethod
    def parse_sse_line(line: str) -> Optional[Dict[str, Any]]:
        """
        Парсинг одной строки SSE.
        
        Args:
            line: Строка SSE
            
        Returns:
            Optional[Dict[str, Any]]: Распарсенные данные или None
        """
        line = line.strip()
        
        if not line or line.startswith(':'):
            # Пустая строка или комментарий
            return None
        
        if line == 'data: [DONE]':
            # Специальный маркер завершения
            return {'type': 'done'}
        
        if line.startswith('data: '):
            data_str = line[6:]  # Убираем 'data: '
            
            try:
                data = json.loads(data_str)
                return data
            except json.JSONDecodeError:
                # Если не JSON, возвращаем как текст
                return {'content': data_str}
        
        # Другие типы SSE полей
        if ':' in line:
            field, value = line.split(':', 1)
            return {field.strip(): value.strip()}
        
        return None
    
    @staticmethod
    def parse_openai_chunk(data: Dict[str, Any], sequence: int) -> Optional[StreamChunk]:
        """
        Парсинг chunk от OpenAI API.
        
        Args:
            data: Данные от OpenAI
            sequence: Порядковый номер
            
        Returns:
            Optional[StreamChunk]: Chunk или None
        """
        if data.get('type') == 'done':
            return StreamChunk(
                id=f"final_{sequence}",
                sequence=sequence,
                content="",
                finish_reason="stop"
            )
        
        if 'choices' not in data or not data['choices']:
            return None
        
        choice = data['choices'][0]
        delta = choice.get('delta', {})
        
        content = delta.get('content', '')
        finish_reason = choice.get('finish_reason')
        
        return StreamChunk(
            id=data.get('id', f"chunk_{sequence}"),
            sequence=sequence,
            content=content,
            delta=delta,
            finish_reason=finish_reason,
            metadata={
                'model': data.get('model'),
                'created': data.get('created'),
                'choice_index': choice.get('index', 0)
            }
        )


class StreamingRequest(BaseModel):
    """
    Запрос для потоковой обработки.
    
    Конфигурирует параметры потоковой передачи данных.
    """
    buffer_size: int = Field(default=1024, gt=0, description="Размер буфера")
    timeout: Optional[float] = Field(None, gt=0, description="Таймаут потока")
    heartbeat_interval: Optional[float] = Field(None, gt=0, description="Интервал heartbeat")
    enable_metrics: bool = Field(default=True, description="Включить сбор метрик")
    filters: List[str] = Field(default_factory=list, description="Фильтры обработки")
    transformers: List[str] = Field(default_factory=list, description="Трансформеры данных")
    
    @field_validator('buffer_size')
    
    @classmethod
    def validate_buffer_size(cls, v):
        """Валидация размера буфера."""
        if v > 10240:  # 10KB
            raise ValueError("Размер буфера слишком большой (максимум 10KB)")
        
        return v


class StreamingResponse(BaseModel):
    """
    Ответ потоковой обработки.
    
    Содержит результаты потоковой передачи и метрики.
    """
    stream_id: str = Field(..., description="Идентификатор потока")
    state: StreamState = Field(..., description="Состояние потока")
    content: str = Field(..., description="Агрегированное содержимое")
    metrics: StreamMetrics = Field(..., description="Метрики потока")
    events: List[StreamEvent] = Field(default_factory=list, description="События потока")
    error: Optional[str] = Field(None, description="Ошибка потока")
    
    @property
    def is_successful(self) -> bool:
        """Проверка успешности потока."""
        return self.state == StreamState.COMPLETED and self.error is None
    
    @property
    def duration(self) -> Optional[float]:
        """Длительность потока."""
        return self.metrics.duration
    
    def get_events_by_type(self, event_type: StreamEventType) -> List[StreamEvent]:
        """
        Получение событий определенного типа.
        
        Args:
            event_type: Тип событий
            
        Returns:
            List[StreamEvent]: События указанного типа
        """
        return [event for event in self.events if event.type == event_type]


class StreamingSession(BaseModel):
    """
    Сессия потоковой передачи.
    
    Управляет жизненным циклом потоковой сессии.
    """
    session_id: str = Field(..., description="Идентификатор сессии")
    processor: StreamProcessor = Field(default_factory=StreamProcessor, description="Процессор потока")
    request: StreamingRequest = Field(..., description="Параметры запроса")
    start_time: datetime = Field(default_factory=datetime.now, description="Время начала сессии")
    end_time: Optional[datetime] = Field(None, description="Время окончания сессии")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    async def process_stream_data(self, data: str) -> List[StreamEvent]:
        """
        Обработка потоковых данных.
        
        Args:
            data: Потоковые данные
            
        Returns:
            List[StreamEvent]: События обработки
        """
        events = []
        lines = data.split('\n')
        sequence = len(self.processor.buffer.chunks)
        
        for line in lines:
            parsed_data = SSEParser.parse_sse_line(line)
            
            if parsed_data:
                chunk = SSEParser.parse_openai_chunk(parsed_data, sequence)
                
                if chunk:
                    event = await self.processor.process_chunk(chunk)
                    events.append(event)
                    sequence += 1
        
        return events
    
    def finalize(self) -> StreamingResponse:
        """
        Финализация сессии.
        
        Returns:
            StreamingResponse: Результат потоковой передачи
        """
        self.end_time = datetime.now()
        
        return StreamingResponse(
            stream_id=self.session_id,
            state=self.processor.buffer.state,
            content=self.processor.get_aggregated_result(),
            metrics=self.processor.get_metrics(),
            events=[],  # События можно добавить при необходимости
            error=None
        )
    
    @property
    def duration(self) -> Optional[float]:
        """Длительность сессии."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None