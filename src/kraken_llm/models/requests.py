"""
Модели данных для запросов Kraken LLM фреймворка.

Этот модуль содержит Pydantic модели для различных типов запросов
к LLM API, включая chat completions, function calling, tool calling и другие.
"""

from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class MessageRole(str, Enum):
    """Роли сообщений в чате."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


class ChatMessage(BaseModel):
    """
    Сообщение чата.
    
    Представляет одно сообщение в диалоге с LLM.
    """
    role: MessageRole = Field(..., description="Роль отправителя сообщения")
    content: Optional[str] = Field(None, description="Содержимое сообщения")
    name: Optional[str] = Field(None, description="Имя отправителя (для function/tool сообщений)")
    function_call: Optional[Dict[str, Any]] = Field(None, description="Вызов функции (устаревший)")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Вызовы инструментов")
    tool_call_id: Optional[str] = Field(None, description="ID вызова инструмента (для tool сообщений)")
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v, info):
        """Валидация содержимого сообщения."""
        role = info.data.get('role') if info.data else None
        
        # Для большинства ролей content обязателен
        if role in [MessageRole.SYSTEM, MessageRole.USER] and not v:
            raise ValueError(f"Содержимое обязательно для роли {role}")
        
        return v
    
    @field_validator('tool_call_id')
    @classmethod
    def validate_tool_call_id(cls, v, info):
        """Валидация ID вызова инструмента."""
        role = info.data.get('role') if info.data else None
        
        if role == MessageRole.TOOL and not v:
            raise ValueError("tool_call_id обязателен для роли tool")
        
        if role != MessageRole.TOOL and v:
            raise ValueError("tool_call_id может быть указан только для роли tool")
        
        return v


class FunctionDefinition(BaseModel):
    """Определение функции для function calling."""
    name: str = Field(..., description="Имя функции")
    description: Optional[str] = Field(None, description="Описание функции")
    parameters: Optional[Dict[str, Any]] = Field(None, description="JSON Schema параметров функции")
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        """Валидация имени функции."""
        if not v or not v.strip():
            raise ValueError("Имя функции не может быть пустым")
        
        # Проверяем, что имя содержит только допустимые символы
        import re
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', v):
            raise ValueError("Имя функции должно содержать только буквы, цифры и подчеркивания")
        
        return v.strip()


class ToolDefinition(BaseModel):
    """Определение инструмента для tool calling."""
    type: Literal["function"] = Field(default="function", description="Тип инструмента")
    function: FunctionDefinition = Field(..., description="Определение функции")


class ResponseFormat(BaseModel):
    """Формат ответа для structured output."""
    type: Literal["text", "json_object", "json_schema"] = Field(..., description="Тип формата ответа")
    json_schema: Optional[Dict[str, Any]] = Field(None, description="JSON схема для structured output")


class ChatCompletionRequest(BaseModel):
    """
    Запрос chat completion.
    
    Основной запрос для получения ответа от LLM модели.
    """
    messages: List[ChatMessage] = Field(..., description="Список сообщений диалога")
    model: Optional[str] = Field(None, description="Модель для использования")
    
    # Параметры генерации
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Температура генерации (0.0-2.0)")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Nucleus sampling параметр (0.0-1.0)")
    max_tokens: Optional[int] = Field(None, gt=0, description="Максимальное количество токенов")
    frequency_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0, description="Штраф за частоту (-2.0 to 2.0)")
    presence_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0, description="Штраф за присутствие (-2.0 to 2.0)")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Последовательности остановки")
    
    # Режимы работы
    stream: Optional[bool] = Field(None, description="Включить потоковый режим")
    response_format: Optional[ResponseFormat] = Field(None, description="Формат ответа")
    
    # Function calling (устаревший)
    functions: Optional[List[FunctionDefinition]] = Field(None, description="Список функций (устаревший)")
    function_call: Optional[Union[str, Dict[str, str]]] = Field(None, description="Управление вызовом функций")
    
    # Tool calling (рекомендуемый)
    tools: Optional[List[ToolDefinition]] = Field(None, description="Список инструментов")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Управление выбором инструментов")
    
    # Logprobs
    logprobs: Optional[bool] = Field(None, description="Возвращать logprobs (если поддерживается)")
    top_logprobs: Optional[int] = Field(None, ge=1, le=5, description="Количество топ-альтернатив (1-5)")
    
    # Дополнительные параметры
    seed: Optional[int] = Field(None, description="Seed для воспроизводимости")
    user: Optional[str] = Field(None, description="Идентификатор пользователя")
    
    @field_validator('messages')
    @classmethod
    def validate_messages(cls, v):
        """Валидация списка сообщений."""
        if not v:
            raise ValueError("Список сообщений не может быть пустым")
        
        # Проверяем, что есть хотя бы одно пользовательское сообщение
        user_messages = [msg for msg in v if msg.role == MessageRole.USER]
        if not user_messages:
            raise ValueError("Должно быть хотя бы одно сообщение от пользователя")
        
        return v
    
    @field_validator('functions', 'tools')
    @classmethod
    def validate_function_tool_conflict(cls, v, info):
        """Проверка конфликта между functions и tools."""
        field_name = info.field_name
        data = info.data if info.data else {}
        
        if field_name == 'tools' and v and data.get('functions'):
            raise ValueError("Нельзя использовать functions и tools одновременно")
        
        if field_name == 'functions' and v and data.get('tools'):
            raise ValueError("Нельзя использовать functions и tools одновременно")
        
        return v

    @field_validator('top_logprobs')
    @classmethod
    def validate_top_logprobs(cls, v, info):
        """Ограничение значений top_logprobs для chat completions."""
        if v is None:
            return v
        # Ограничим безопасным диапазоном 1..5
        if not (1 <= v <= 5):
            raise ValueError("top_logprobs должен быть в диапазоне 1..5")
        return v


class StreamingRequest(ChatCompletionRequest):
    """
    Запрос для потокового chat completion.
    
    Расширяет базовый запрос с принудительным включением streaming.
    """
    stream: bool = Field(True, description="Потоковый режим (всегда True)")


class StructuredOutputRequest(ChatCompletionRequest):
    """
    Запрос для structured output.
    
    Расширяет базовый запрос с обязательным форматом ответа.
    """
    response_format: ResponseFormat = Field(..., description="Формат ответа (обязательный)")
    
    @field_validator('response_format')
    @classmethod
    def validate_response_format(cls, v):
        """Валидация формата ответа для structured output."""
        if v.type == "json_schema" and not v.json_schema:
            raise ValueError("json_schema обязательна для типа json_schema")
        
        return v


class FunctionCallingRequest(ChatCompletionRequest):
    """
    Запрос для function calling.
    
    Расширяет базовый запрос с обязательными функциями.
    """
    functions: List[FunctionDefinition] = Field(..., description="Список функций (обязательный)")
    function_call: Optional[Union[str, Dict[str, str]]] = Field("auto", description="Управление вызовом функций")


class ToolCallingRequest(ChatCompletionRequest):
    """
    Запрос для tool calling.
    
    Расширяет базовый запрос с обязательными инструментами.
    """
    tools: List[ToolDefinition] = Field(..., description="Список инструментов (обязательный)")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field("auto", description="Управление выбором инструментов")


class MultimodalMessage(ChatMessage):
    """
    Мультимодальное сообщение.
    
    Расширяет базовое сообщение поддержкой различных типов контента.
    """
    content: Optional[Union[str, List[Dict[str, Any]]]] = Field(
        None, 
        description="Содержимое сообщения (текст или список контента)"
    )
    
    @field_validator('content')
    @classmethod
    def validate_multimodal_content(cls, v):
        """Валидация мультимодального контента."""
        if isinstance(v, list):
            # Проверяем структуру элементов контента
            for item in v:
                if not isinstance(item, dict):
                    raise ValueError("Элементы контента должны быть словарями")
                
                if 'type' not in item:
                    raise ValueError("Каждый элемент контента должен иметь поле 'type'")
                
                content_type = item['type']
                if content_type == 'text' and 'text' not in item:
                    raise ValueError("Текстовый контент должен иметь поле 'text'")
                elif content_type == 'image_url' and 'image_url' not in item:
                    raise ValueError("Изображение должно иметь поле 'image_url'")
        
        return v


class MultimodalRequest(ChatCompletionRequest):
    """
    Запрос для мультимодального взаимодействия.
    
    Поддерживает сообщения с различными типами контента.
    """
    messages: List[MultimodalMessage] = Field(..., description="Список мультимодальных сообщений")


class EmbeddingsRequest(BaseModel):
    """
    Запрос для получения векторных представлений.
    
    Используется для генерации embeddings из текста.
    """
    input: Union[str, List[str]] = Field(..., description="Входной текст или список текстов")
    model: Optional[str] = Field(None, description="Модель для embeddings")
    encoding_format: Optional[Literal["float", "base64"]] = Field("float", description="Формат кодирования")
    dimensions: Optional[int] = Field(None, gt=0, description="Размерность векторов")
    user: Optional[str] = Field(None, description="Идентификатор пользователя")
    
    @field_validator('input')
    @classmethod
    def validate_input(cls, v):
        """Валидация входных данных."""
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("Входной текст не может быть пустым")
        elif isinstance(v, list):
            if not v:
                raise ValueError("Список входных текстов не может быть пустым")
            
            for text in v:
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("Все элементы списка должны быть непустыми строками")
        
        return v


class ASRRequest(BaseModel):
    """
    Запрос для автоматического распознавания речи.
    
    Используется для преобразования аудио в текст.
    """
    file: Union[str, bytes] = Field(..., description="Аудио файл (путь или байты)")
    model: Optional[str] = Field(None, description="Модель для распознавания")
    language: Optional[str] = Field(None, description="Язык аудио (ISO 639-1)")
    prompt: Optional[str] = Field(None, description="Подсказка для улучшения точности")
    response_format: Optional[Literal["json", "text", "srt", "verbose_json", "vtt"]] = Field(
        "json", 
        description="Формат ответа"
    )
    temperature: Optional[float] = Field(None, ge=0.0, le=1.0, description="Температура генерации")
    
    @field_validator('file')
    @classmethod
    def validate_file(cls, v):
        """Валидация аудио файла."""
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("Путь к файлу не может быть пустым")
        elif isinstance(v, bytes):
            if not v:
                raise ValueError("Аудио данные не могут быть пустыми")
        else:
            raise ValueError("Файл должен быть строкой (путь) или байтами")
        
        return v


class TTSRequest(BaseModel):
    """
    Запрос для синтеза речи.
    
    Используется для преобразования текста в речь.
    """
    input: str = Field(..., description="Текст для синтеза")
    model: Optional[str] = Field(None, description="Модель для синтеза")
    voice: Optional[str] = Field(None, description="Голос для синтеза")
    response_format: Optional[Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]] = Field(
        "mp3", 
        description="Формат аудио"
    )
    speed: Optional[float] = Field(None, ge=0.25, le=4.0, description="Скорость речи (0.25-4.0)")
    
    @field_validator('input')
    @classmethod
    def validate_input(cls, v):
        """Валидация входного текста."""
        if not v or not v.strip():
            raise ValueError("Текст для синтеза не может быть пустым")
        
        if len(v) > 4096:
            raise ValueError("Текст не может быть длиннее 4096 символов")
        
        return v.strip()


class BatchRequest(BaseModel):
    """
    Запрос для батчевой обработки.
    
    Позволяет обрабатывать несколько запросов одновременно.
    """
    requests: List[ChatCompletionRequest] = Field(..., description="Список запросов для обработки")
    max_concurrent: Optional[int] = Field(None, gt=0, description="Максимальное количество одновременных запросов")
    timeout: Optional[float] = Field(None, gt=0, description="Таймаут для каждого запроса")
    
    @field_validator('requests')
    @classmethod
    def validate_requests(cls, v):
        """Валидация списка запросов."""
        if not v:
            raise ValueError("Список запросов не может быть пустым")
        
        if len(v) > 100:
            raise ValueError("Слишком много запросов в батче (максимум 100)")
        
        return v