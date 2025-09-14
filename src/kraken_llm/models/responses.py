"""
Модели данных для ответов Kraken LLM фреймворка.

Этот модуль содержит Pydantic модели для различных типов ответов
от LLM API, включая chat completions, embeddings, ASR и другие.
"""

from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import time


class FinishReason(str, Enum):
    """Причины завершения генерации."""
    STOP = "stop"
    LENGTH = "length"
    FUNCTION_CALL = "function_call"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"


class Usage(BaseModel):
    """
    Информация об использовании токенов.
    
    Содержит статистику потребления токенов для запроса.
    """
    prompt_tokens: int = Field(..., description="Количество токенов в промпте")
    completion_tokens: Optional[int] = Field(None, description="Количество токенов в ответе")
    total_tokens: int = Field(..., description="Общее количество токенов")
    
    @field_validator('total_tokens')
    
    @classmethod
    def validate_total_tokens(cls, v, info):
        """Валидация общего количества токенов."""
        data = info.data if info.data else {}
        prompt_tokens = data.get('prompt_tokens', 0)
        completion_tokens = data.get('completion_tokens', 0) or 0
        
        expected_total = prompt_tokens + completion_tokens
        if v != expected_total:
            # Предупреждение, но не ошибка, так как некоторые API могут считать по-разному
            pass
        
        return v


class FunctionCall(BaseModel):
    """
    Вызов функции от модели.
    
    Представляет запрос модели на выполнение функции.
    """
    name: str = Field(..., description="Имя функции для вызова")
    arguments: str = Field(..., description="Аргументы функции в формате JSON")
    
    @field_validator('arguments')
    
    @classmethod
    def validate_arguments(cls, v):
        """Валидация аргументов функции."""
        if not isinstance(v, str):
            raise ValueError("Аргументы должны быть строкой JSON")
        
        # Проверяем, что это валидный JSON
        try:
            import json
            json.loads(v)
        except json.JSONDecodeError:
            raise ValueError("Аргументы должны быть валидным JSON")
        
        return v


class ToolCall(BaseModel):
    """
    Вызов инструмента от модели.
    
    Представляет запрос модели на выполнение инструмента.
    """
    id: str = Field(..., description="Уникальный идентификатор вызова")
    type: Literal["function"] = Field(..., description="Тип инструмента")
    function: FunctionCall = Field(..., description="Детали вызова функции")


class ChatMessage(BaseModel):
    """
    Сообщение в ответе от модели.
    
    Содержит ответ модели или информацию о вызовах функций/инструментов.
    """
    role: Literal["assistant"] = Field(..., description="Роль (всегда assistant для ответов)")
    content: Optional[str] = Field(None, description="Текстовое содержимое ответа")
    function_call: Optional[FunctionCall] = Field(None, description="Вызов функции (устаревший)")
    tool_calls: Optional[List[ToolCall]] = Field(None, description="Вызовы инструментов")
    
    @field_validator('content', 'function_call', 'tool_calls')
    
    @classmethod
    def validate_content_or_calls(cls, v, info):
        """Проверка, что есть либо контент, либо вызовы."""
        # Собираем все поля для проверки
        data = info.data if info.data else {}
        field_name = info.field_name
        
        all_fields = {
            'content': data.get('content'),
            'function_call': data.get('function_call'),
            'tool_calls': data.get('tool_calls')
        }
        all_fields[field_name] = v
        
        # Проверяем, что есть хотя бы одно непустое поле
        has_content = bool(all_fields['content'])
        has_function_call = bool(all_fields['function_call'])
        has_tool_calls = bool(all_fields['tool_calls'])
        
        if not (has_content or has_function_call or has_tool_calls):
            raise ValueError("Сообщение должно содержать либо content, либо function_call, либо tool_calls")
        
        return v


class Choice(BaseModel):
    """
    Вариант ответа от модели.
    
    Представляет один из возможных ответов модели.
    """
    index: int = Field(..., description="Индекс варианта ответа")
    message: ChatMessage = Field(..., description="Сообщение от модели")
    finish_reason: Optional[FinishReason] = Field(None, description="Причина завершения генерации")
    logprobs: Optional[Dict[str, Any]] = Field(None, description="Логарифмы вероятностей токенов")


class ChatCompletionResponse(BaseModel):
    """
    Ответ chat completion.
    
    Основной ответ от LLM модели на запрос chat completion.
    """
    id: str = Field(..., description="Уникальный идентификатор ответа")
    object: Literal["chat.completion"] = Field(..., description="Тип объекта")
    created: int = Field(..., description="Временная метка создания (Unix timestamp)")
    model: str = Field(..., description="Модель, использованная для генерации")
    choices: List[Choice] = Field(..., description="Варианты ответов от модели")
    usage: Optional[Usage] = Field(None, description="Информация об использовании токенов")
    system_fingerprint: Optional[str] = Field(None, description="Отпечаток системы")
    
    @field_validator('choices')
    
    @classmethod
    def validate_choices(cls, v):
        """Валидация вариантов ответов."""
        if not v:
            raise ValueError("Должен быть хотя бы один вариант ответа")
        
        return v
    
    @field_validator('created')
    
    @classmethod
    def validate_created(cls, v):
        """Валидация временной метки."""
        current_time = int(time.time())
        
        # Проверяем, что время не слишком далеко в прошлом или будущем
        if v < current_time - 86400:  # Не старше суток
            raise ValueError("Временная метка слишком старая")
        
        if v > current_time + 3600:  # Не более часа в будущем
            raise ValueError("Временная метка из будущего")
        
        return v


class StreamDelta(BaseModel):
    """
    Дельта для потокового ответа.
    
    Представляет изменения в потоковом ответе.
    """
    role: Optional[Literal["assistant"]] = Field(None, description="Роль (только для первого chunk)")
    content: Optional[str] = Field(None, description="Часть текстового содержимого")
    function_call: Optional[Dict[str, Any]] = Field(None, description="Часть вызова функции")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Части вызовов инструментов")


class StreamChoice(BaseModel):
    """
    Вариант ответа в потоковом режиме.
    
    Представляет один chunk потокового ответа.
    """
    index: int = Field(..., description="Индекс варианта ответа")
    delta: StreamDelta = Field(..., description="Изменения в этом chunk")
    finish_reason: Optional[FinishReason] = Field(None, description="Причина завершения (только в последнем chunk)")
    logprobs: Optional[Dict[str, Any]] = Field(None, description="Логарифмы вероятностей токенов")


class ChatCompletionStreamResponse(BaseModel):
    """
    Chunk потокового ответа chat completion.
    
    Представляет один фрагмент потокового ответа от модели.
    """
    id: str = Field(..., description="Уникальный идентификатор ответа")
    object: Literal["chat.completion.chunk"] = Field(..., description="Тип объекта")
    created: int = Field(..., description="Временная метка создания (Unix timestamp)")
    model: str = Field(..., description="Модель, использованная для генерации")
    choices: List[StreamChoice] = Field(..., description="Варианты ответов в этом chunk")
    system_fingerprint: Optional[str] = Field(None, description="Отпечаток системы")


class EmbeddingData(BaseModel):
    """
    Данные векторного представления.
    
    Содержит embedding для одного входного текста.
    """
    object: Literal["embedding"] = Field(..., description="Тип объекта")
    embedding: List[float] = Field(..., description="Векторное представление")
    index: int = Field(..., description="Индекс входного текста")
    
    @field_validator('embedding')
    
    @classmethod
    def validate_embedding(cls, v):
        """Валидация векторного представления."""
        if not v:
            raise ValueError("Векторное представление не может быть пустым")
        
        # Проверяем, что все элементы - числа
        for i, val in enumerate(v):
            if not isinstance(val, (int, float)):
                raise ValueError(f"Элемент {i} должен быть числом")
        
        return v


class EmbeddingsResponse(BaseModel):
    """
    Ответ для запроса embeddings.
    
    Содержит векторные представления для входных текстов.
    """
    object: Literal["list"] = Field(..., description="Тип объекта")
    data: List[EmbeddingData] = Field(..., description="Список векторных представлений")
    model: str = Field(..., description="Модель, использованная для генерации")
    usage: Usage = Field(..., description="Информация об использовании токенов")
    
    @field_validator('data')
    
    @classmethod
    def validate_data(cls, v):
        """Валидация данных embeddings."""
        if not v:
            raise ValueError("Должно быть хотя бы одно векторное представление")
        
        return v


class ASRSegment(BaseModel):
    """
    Сегмент распознанной речи.
    
    Представляет временной сегмент с распознанным текстом.
    """
    id: int = Field(..., description="Идентификатор сегмента")
    seek: int = Field(..., description="Позиция поиска в аудио")
    start: float = Field(..., description="Время начала сегмента (секунды)")
    end: float = Field(..., description="Время окончания сегмента (секунды)")
    text: str = Field(..., description="Распознанный текст")
    tokens: List[int] = Field(..., description="Токены текста")
    temperature: float = Field(..., description="Температура, использованная для генерации")
    avg_logprob: float = Field(..., description="Средняя логарифмическая вероятность")
    compression_ratio: float = Field(..., description="Коэффициент сжатия")
    no_speech_prob: float = Field(..., description="Вероятность отсутствия речи")
    
    @field_validator('start', 'end')
    
    @classmethod
    def validate_time_bounds(cls, v, info):
        """Валидация временных границ."""
        field_name = info.field_name
        
        if v < 0:
            raise ValueError(f"{field_name} не может быть отрицательным")
        
        if field_name == 'end':
            data = info.data if info.data else {}
            start = data.get('start', 0)
            if v <= start:
                raise ValueError("Время окончания должно быть больше времени начала")
        
        return v


class ASRResponse(BaseModel):
    """
    Ответ для запроса распознавания речи.
    
    Содержит результат распознавания аудио.
    """
    task: Literal["transcribe", "translate"] = Field(..., description="Тип задачи")
    language: str = Field(..., description="Обнаруженный язык")
    duration: float = Field(..., description="Длительность аудио (секунды)")
    text: str = Field(..., description="Полный распознанный текст")
    segments: Optional[List[ASRSegment]] = Field(None, description="Сегменты речи")
    
    @field_validator('duration')
    
    @classmethod
    def validate_duration(cls, v):
        """Валидация длительности аудио."""
        if v <= 0:
            raise ValueError("Длительность должна быть положительной")
        
        return v


class TTSResponse(BaseModel):
    """
    Ответ для запроса синтеза речи.
    
    Содержит синтезированное аудио.
    """
    audio: bytes = Field(..., description="Аудио данные")
    content_type: str = Field(..., description="MIME тип аудио")
    duration: Optional[float] = Field(None, description="Длительность аудио (секунды)")
    
    @field_validator('audio')
    
    @classmethod
    def validate_audio(cls, v):
        """Валидация аудио данных."""
        if not v:
            raise ValueError("Аудио данные не могут быть пустыми")
        
        return v


class ErrorDetail(BaseModel):
    """
    Детали ошибки.
    
    Содержит подробную информацию об ошибке.
    """
    code: str = Field(..., description="Код ошибки")
    message: str = Field(..., description="Сообщение об ошибке")
    param: Optional[str] = Field(None, description="Параметр, вызвавший ошибку")
    type: str = Field(..., description="Тип ошибки")


class ErrorResponse(BaseModel):
    """
    Ответ с ошибкой.
    
    Стандартный формат ответа при возникновении ошибки.
    """
    error: ErrorDetail = Field(..., description="Детали ошибки")


class BatchResult(BaseModel):
    """
    Результат одного запроса в батче.
    
    Содержит результат или ошибку для одного запроса.
    """
    index: int = Field(..., description="Индекс запроса в батче")
    success: bool = Field(..., description="Успешность выполнения")
    response: Optional[ChatCompletionResponse] = Field(None, description="Ответ при успехе")
    error: Optional[ErrorDetail] = Field(None, description="Ошибка при неудаче")
    execution_time: float = Field(..., description="Время выполнения (секунды)")
    
    @field_validator('response', 'error')
    
    @classmethod
    def validate_response_or_error(cls, v, info):
        """Проверка, что есть либо ответ, либо ошибка."""
        data = info.data if info.data else {}
        field_name = info.field_name
        
        success = data.get('success')
        response = data.get('response') if field_name != 'response' else v
        error = data.get('error') if field_name != 'error' else v
        
        if success and not response:
            raise ValueError("При успешном выполнении должен быть ответ")
        
        if not success and not error:
            raise ValueError("При неуспешном выполнении должна быть ошибка")
        
        if success and error:
            raise ValueError("Не может быть одновременно ответа и ошибки при успехе")
        
        return v


class BatchResponse(BaseModel):
    """
    Ответ для батчевого запроса.
    
    Содержит результаты всех запросов в батче.
    """
    results: List[BatchResult] = Field(..., description="Результаты запросов")
    total_requests: int = Field(..., description="Общее количество запросов")
    successful_requests: int = Field(..., description="Количество успешных запросов")
    failed_requests: int = Field(..., description="Количество неудачных запросов")
    total_execution_time: float = Field(..., description="Общее время выполнения (секунды)")
    
    @field_validator('total_requests')
    
    @classmethod
    def validate_total_requests(cls, v, info):
        """Валидация общего количества запросов."""
        data = info.data if info.data else {}
        results = data.get('results', [])
        if v != len(results):
            raise ValueError("Общее количество запросов должно совпадать с количеством результатов")
        
        return v
    
    @field_validator('successful_requests', 'failed_requests')
    
    @classmethod
    def validate_request_counts(cls, v, info):
        """Валидация счетчиков запросов."""
        data = info.data if info.data else {}
        field_name = info.field_name
        results = data.get('results', [])
        
        if field_name == 'successful_requests':
            actual_successful = sum(1 for r in results if r.success)
            if v != actual_successful:
                raise ValueError("Количество успешных запросов не совпадает с фактическим")
        
        if field_name == 'failed_requests':
            actual_failed = sum(1 for r in results if not r.success)
            if v != actual_failed:
                raise ValueError("Количество неудачных запросов не совпадает с фактическим")
        
        return v


class HealthCheckResponse(BaseModel):
    """
    Ответ проверки здоровья системы.
    
    Используется для мониторинга состояния сервиса.
    """
    status: Literal["healthy", "unhealthy", "degraded"] = Field(..., description="Статус системы")
    timestamp: int = Field(..., description="Временная метка проверки")
    version: str = Field(..., description="Версия системы")
    uptime: float = Field(..., description="Время работы (секунды)")
    checks: Dict[str, bool] = Field(..., description="Результаты проверок компонентов")
    
    @field_validator('timestamp')
    
    @classmethod
    def validate_timestamp(cls, v):
        """Валидация временной метки."""
        current_time = int(time.time())
        
        if abs(v - current_time) > 60:  # Не более минуты разницы
            raise ValueError("Временная метка проверки здоровья некорректна")
        
        return v