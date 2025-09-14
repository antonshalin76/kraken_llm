"""
Модели данных для function и tool calling в Kraken LLM фреймворке.

Этот модуль содержит Pydantic модели для работы с функциями и инструментами,
включая их определения, вызовы, результаты выполнения и метаданные.
"""

from typing import List, Dict, Any, Optional, Union, Callable, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum
import inspect
import json
from datetime import datetime


class ParameterType(str, Enum):
    """Типы параметров функций."""
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null"


class FunctionParameter(BaseModel):
    """
    Параметр функции.
    
    Описывает один параметр функции в JSON Schema формате.
    """
    type: ParameterType = Field(..., description="Тип параметра")
    description: Optional[str] = Field(None, description="Описание параметра")
    enum: Optional[List[Any]] = Field(None, description="Допустимые значения")
    default: Optional[Any] = Field(None, description="Значение по умолчанию")
    
    # Для строк
    min_length: Optional[int] = Field(None, ge=0, description="Минимальная длина строки")
    max_length: Optional[int] = Field(None, ge=0, description="Максимальная длина строки")
    pattern: Optional[str] = Field(None, description="Регулярное выражение для валидации")
    
    # Для чисел
    minimum: Optional[Union[int, float]] = Field(None, description="Минимальное значение")
    maximum: Optional[Union[int, float]] = Field(None, description="Максимальное значение")
    exclusive_minimum: Optional[bool] = Field(None, description="Исключить минимальное значение")
    exclusive_maximum: Optional[bool] = Field(None, description="Исключить максимальное значение")
    
    # Для массивов
    items: Optional[Dict[str, Any]] = Field(None, description="Схема элементов массива")
    min_items: Optional[int] = Field(None, ge=0, description="Минимальное количество элементов")
    max_items: Optional[int] = Field(None, ge=0, description="Максимальное количество элементов")
    unique_items: Optional[bool] = Field(None, description="Уникальность элементов")
    
    # Для объектов
    properties: Optional[Dict[str, "FunctionParameter"]] = Field(None, description="Свойства объекта")
    required: Optional[List[str]] = Field(None, description="Обязательные свойства")
    additional_properties: Optional[bool] = Field(None, description="Разрешить дополнительные свойства")
    
    @field_validator('min_length', 'max_length')
    
    @classmethod
    def validate_string_lengths(cls, v, info):
        """Валидация длин строк."""
        data = info.data if info.data else {}
        field_name = info.field_name
        
        if v is not None and data.get('type') != ParameterType.STRING:
            raise ValueError(f"{field_name} применимо только к строковым параметрам")
        
        if field_name == 'max_length':
            min_length = data.get('min_length')
            if min_length is not None and v < min_length:
                raise ValueError("max_length должен быть больше или равен min_length")
        
        return v
    
    @field_validator('minimum', 'maximum')
    
    @classmethod
    def validate_number_bounds(cls, v, info):
        """Валидация границ чисел."""
        data = info.data if info.data else {}
        field_name = info.field_name
        param_type = data.get('type')
        
        if v is not None and param_type not in [ParameterType.NUMBER, ParameterType.INTEGER]:
            raise ValueError(f"{field_name} применимо только к числовым параметрам")
        
        if field_name == 'maximum':
            minimum = data.get('minimum')
            if minimum is not None and v < minimum:
                raise ValueError("maximum должен быть больше или равен minimum")
        
        return v


# Обновляем модель для поддержки рекурсивных ссылок
FunctionParameter.model_rebuild()


class FunctionSchema(BaseModel):
    """
    Схема функции.
    
    Полное описание функции включая параметры и метаданные.
    """
    name: str = Field(..., description="Имя функции")
    description: str = Field(..., description="Описание функции")
    parameters: Dict[str, FunctionParameter] = Field(default_factory=dict, description="Параметры функции")
    required_parameters: List[str] = Field(default_factory=list, description="Обязательные параметры")
    return_type: Optional[ParameterType] = Field(None, description="Тип возвращаемого значения")
    return_description: Optional[str] = Field(None, description="Описание возвращаемого значения")
    examples: Optional[List[Dict[str, Any]]] = Field(None, description="Примеры использования")
    
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
    
    @field_validator('required_parameters')
    
    @classmethod
    def validate_required_parameters(cls, v, info):
        """Валидация обязательных параметров."""
        data = info.data if info.data else {}
        parameters = data.get('parameters', {})
        
        for param_name in v:
            if param_name not in parameters:
                raise ValueError(f"Обязательный параметр '{param_name}' не найден в списке параметров")
        
        return v
    
    def to_openai_format(self) -> Dict[str, Any]:
        """
        Конвертация в формат OpenAI Functions.
        
        Returns:
            Dict[str, Any]: Схема функции в формате OpenAI
        """
        # Конвертируем параметры в JSON Schema
        properties = {}
        for param_name, param in self.parameters.items():
            properties[param_name] = param.model_dump(exclude_none=True)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": self.required_parameters
            }
        }
    
    def to_openai_tool_format(self) -> Dict[str, Any]:
        """
        Конвертация в формат OpenAI Tools.
        
        Returns:
            Dict[str, Any]: Схема инструмента в формате OpenAI
        """
        return {
            "type": "function",
            "function": self.to_openai_format()
        }


class FunctionCall(BaseModel):
    """
    Вызов функции.
    
    Представляет запрос на выполнение функции с конкретными аргументами.
    """
    name: str = Field(..., description="Имя функции")
    arguments: Dict[str, Any] = Field(..., description="Аргументы функции")
    call_id: Optional[str] = Field(None, description="Уникальный идентификатор вызова")
    timestamp: Optional[datetime] = Field(None, description="Время вызова")
    
    @field_validator('arguments')
    
    @classmethod
    def validate_arguments(cls, v):
        """Валидация аргументов функции."""
        if not isinstance(v, dict):
            raise ValueError("Аргументы должны быть словарем")
        
        return v
    
    @classmethod
    def from_openai_format(cls, name: str, arguments_json: str, call_id: Optional[str] = None) -> "FunctionCall":
        """
        Создание из формата OpenAI.
        
        Args:
            name: Имя функции
            arguments_json: Аргументы в формате JSON строки
            call_id: Идентификатор вызова
            
        Returns:
            FunctionCall: Объект вызова функции
        """
        try:
            arguments = json.loads(arguments_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Некорректный JSON в аргументах: {e}")
        
        return cls(
            name=name,
            arguments=arguments,
            call_id=call_id,
            timestamp=datetime.now()
        )


class ToolCall(BaseModel):
    """
    Вызов инструмента.
    
    Представляет запрос на выполнение инструмента (расширенная версия function call).
    """
    id: str = Field(..., description="Уникальный идентификатор вызова")
    type: Literal["function"] = Field(default="function", description="Тип инструмента")
    function: FunctionCall = Field(..., description="Детали вызова функции")
    
    @classmethod
    def from_openai_format(cls, tool_call_data: Dict[str, Any]) -> "ToolCall":
        """
        Создание из формата OpenAI.
        
        Args:
            tool_call_data: Данные вызова инструмента от OpenAI
            
        Returns:
            ToolCall: Объект вызова инструмента
        """
        function_data = tool_call_data["function"]
        function_call = FunctionCall.from_openai_format(
            name=function_data["name"],
            arguments_json=function_data["arguments"],
            call_id=tool_call_data["id"]
        )
        
        return cls(
            id=tool_call_data["id"],
            type=tool_call_data["type"],
            function=function_call
        )


class ExecutionStatus(str, Enum):
    """Статусы выполнения функций/инструментов."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ExecutionResult(BaseModel):
    """
    Результат выполнения функции или инструмента.
    
    Содержит результат выполнения, метаданные и информацию об ошибках.
    """
    call_id: Optional[str] = Field(None, description="Идентификатор вызова")
    function_name: str = Field(..., description="Имя выполненной функции")
    status: ExecutionStatus = Field(..., description="Статус выполнения")
    result: Optional[Any] = Field(None, description="Результат выполнения")
    error: Optional[str] = Field(None, description="Сообщение об ошибке")
    error_type: Optional[str] = Field(None, description="Тип ошибки")
    execution_time: float = Field(..., description="Время выполнения в секундах")
    timestamp: datetime = Field(default_factory=datetime.now, description="Время завершения")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Дополнительные метаданные")
    
    @field_validator('result', 'error')
    
    @classmethod
    def validate_result_or_error(cls, v, info):
        """Проверка, что есть либо результат, либо ошибка."""
        data = info.data if info.data else {}
        field_name = info.field_name
        
        status = data.get('status')
        result = data.get('result') if field_name != 'result' else v
        error = data.get('error') if field_name != 'error' else v
        
        if status == ExecutionStatus.SUCCESS and result is None:
            raise ValueError("При успешном выполнении должен быть результат")
        
        if status == ExecutionStatus.ERROR and not error:
            raise ValueError("При ошибке должно быть сообщение об ошибке")
        
        return v
    
    @field_validator('execution_time')
    
    @classmethod
    def validate_execution_time(cls, v):
        """Валидация времени выполнения."""
        if v < 0:
            raise ValueError("Время выполнения не может быть отрицательным")
        
        return v
    
    @property
    def success(self) -> bool:
        """Проверка успешности выполнения."""
        return self.status == ExecutionStatus.SUCCESS
    
    def to_openai_message(self) -> Dict[str, Any]:
        """
        Конвертация в формат сообщения OpenAI.
        
        Returns:
            Dict[str, Any]: Сообщение для отправки в OpenAI API
        """
        content = str(self.result) if self.success else f"Ошибка: {self.error}"
        
        if self.call_id:
            # Для tool calling
            return {
                "role": "tool",
                "tool_call_id": self.call_id,
                "content": content
            }
        else:
            # Для function calling (устаревший)
            return {
                "role": "function",
                "name": self.function_name,
                "content": content
            }


class FunctionRegistry(BaseModel):
    """
    Реестр функций.
    
    Управляет коллекцией доступных функций и их схем.
    """
    functions: Dict[str, FunctionSchema] = Field(default_factory=dict, description="Зарегистрированные функции")
    handlers: Dict[str, Callable] = Field(default_factory=dict, description="Обработчики функций")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def register_function(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, FunctionParameter]] = None,
        required_parameters: Optional[List[str]] = None
    ) -> FunctionSchema:
        """
        Регистрация функции.
        
        Args:
            func: Функция для регистрации
            name: Имя функции (по умолчанию берется из func.__name__)
            description: Описание функции
            parameters: Параметры функции
            required_parameters: Обязательные параметры
            
        Returns:
            FunctionSchema: Схема зарегистрированной функции
        """
        # Автоматическое извлечение информации из функции
        func_name = name or func.__name__
        func_description = description or func.__doc__ or f"Функция {func_name}"
        
        # Автоматическое извлечение параметров из сигнатуры
        if parameters is None:
            parameters = self._extract_parameters_from_signature(func)
        
        if required_parameters is None:
            required_parameters = self._extract_required_parameters(func)
        
        # Создание схемы функции
        schema = FunctionSchema(
            name=func_name,
            description=func_description,
            parameters=parameters,
            required_parameters=required_parameters
        )
        
        # Регистрация
        self.functions[func_name] = schema
        self.handlers[func_name] = func
        
        return schema
    
    def _extract_parameters_from_signature(self, func: Callable) -> Dict[str, FunctionParameter]:
        """Извлечение параметров из сигнатуры функции."""
        sig = inspect.signature(func)
        parameters = {}
        
        for param_name, param in sig.parameters.items():
            # Определение типа параметра
            param_type = ParameterType.STRING  # По умолчанию
            
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_type = ParameterType.INTEGER
                elif param.annotation == float:
                    param_type = ParameterType.NUMBER
                elif param.annotation == bool:
                    param_type = ParameterType.BOOLEAN
                elif param.annotation == list:
                    param_type = ParameterType.ARRAY
                elif param.annotation == dict:
                    param_type = ParameterType.OBJECT
            
            # Значение по умолчанию
            default_value = None
            if param.default != inspect.Parameter.empty:
                default_value = param.default
            
            parameters[param_name] = FunctionParameter(
                type=param_type,
                description=f"Параметр {param_name}",
                default=default_value
            )
        
        return parameters
    
    def _extract_required_parameters(self, func: Callable) -> List[str]:
        """Извлечение обязательных параметров из сигнатуры функции."""
        sig = inspect.signature(func)
        required = []
        
        for param_name, param in sig.parameters.items():
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        return required
    
    def get_function_schema(self, name: str) -> Optional[FunctionSchema]:
        """Получение схемы функции по имени."""
        return self.functions.get(name)
    
    def get_function_handler(self, name: str) -> Optional[Callable]:
        """Получение обработчика функции по имени."""
        return self.handlers.get(name)
    
    def list_functions(self) -> List[str]:
        """Получение списка зарегистрированных функций."""
        return list(self.functions.keys())
    
    def to_openai_functions_format(self) -> List[Dict[str, Any]]:
        """Конвертация всех функций в формат OpenAI Functions."""
        return [schema.to_openai_format() for schema in self.functions.values()]
    
    def to_openai_tools_format(self) -> List[Dict[str, Any]]:
        """Конвертация всех функций в формат OpenAI Tools."""
        return [schema.to_openai_tool_format() for schema in self.functions.values()]


class ToolRegistry(FunctionRegistry):
    """
    Реестр инструментов.
    
    Расширенная версия реестра функций для работы с OpenAI Tools API.
    """
    
    def register_tool(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, FunctionParameter]] = None,
        required_parameters: Optional[List[str]] = None
    ) -> FunctionSchema:
        """
        Регистрация инструмента (алиас для register_function).
        
        Args:
            func: Функция-инструмент для регистрации
            name: Имя инструмента
            description: Описание инструмента
            parameters: Параметры инструмента
            required_parameters: Обязательные параметры
            
        Returns:
            FunctionSchema: Схема зарегистрированного инструмента
        """
        return self.register_function(func, name, description, parameters, required_parameters)


class ExecutionContext(BaseModel):
    """
    Контекст выполнения функций/инструментов.
    
    Содержит дополнительную информацию для выполнения функций.
    """
    user_id: Optional[str] = Field(None, description="Идентификатор пользователя")
    session_id: Optional[str] = Field(None, description="Идентификатор сессии")
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list, description="История разговора")
    environment: Dict[str, Any] = Field(default_factory=dict, description="Переменные окружения")
    permissions: List[str] = Field(default_factory=list, description="Разрешения пользователя")
    timeout: Optional[float] = Field(None, description="Таймаут выполнения")
    
    def has_permission(self, permission: str) -> bool:
        """Проверка наличия разрешения."""
        return permission in self.permissions
    
    def get_environment_variable(self, key: str, default: Any = None) -> Any:
        """Получение переменной окружения."""
        return self.environment.get(key, default)


class BatchExecutionRequest(BaseModel):
    """
    Запрос на батчевое выполнение функций/инструментов.
    
    Позволяет выполнить несколько функций параллельно.
    """
    calls: List[Union[FunctionCall, ToolCall]] = Field(..., description="Список вызовов для выполнения")
    context: Optional[ExecutionContext] = Field(None, description="Контекст выполнения")
    max_concurrent: Optional[int] = Field(None, gt=0, description="Максимальное количество параллельных выполнений")
    timeout: Optional[float] = Field(None, gt=0, description="Общий таймаут для всех вызовов")
    
    @field_validator('calls')
    
    @classmethod
    def validate_calls(cls, v):
        """Валидация списка вызовов."""
        if not v:
            raise ValueError("Список вызовов не может быть пустым")
        
        if len(v) > 50:
            raise ValueError("Слишком много вызовов в батче (максимум 50)")
        
        return v


class BatchExecutionResponse(BaseModel):
    """
    Ответ на батчевое выполнение функций/инструментов.
    
    Содержит результаты всех выполненных функций.
    """
    results: List[ExecutionResult] = Field(..., description="Результаты выполнения")
    total_calls: int = Field(..., description="Общее количество вызовов")
    successful_calls: int = Field(..., description="Количество успешных вызовов")
    failed_calls: int = Field(..., description="Количество неудачных вызовов")
    total_execution_time: float = Field(..., description="Общее время выполнения")
    
    @field_validator('total_calls')
    
    @classmethod
    def validate_total_calls(cls, v, info):
        """Валидация общего количества вызовов."""
        data = info.data if info.data else {}
        results = data.get('results', [])
        if v != len(results):
            raise ValueError("Общее количество вызовов должно совпадать с количеством результатов")
        
        return v