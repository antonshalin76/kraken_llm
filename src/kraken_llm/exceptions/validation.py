"""
Исключения, связанные с валидацией, для Kraken LLM Framework.

Этот модуль определяет исключения, связанные с валидацией данных, валидацией схем
и ошибками валидации моделей Pydantic.
"""

from typing import Any, Dict, List, Optional, Union

from .base import KrakenError


class ValidationError(KrakenError):
    """
    Исключение, возникающее при неудаче валидации данных.
    
    Это исключение возникает когда:
    - Валидация модели Pydantic не удается
    - Валидация JSON схемы не удается
    - Валидация формата ответа не удается
    - Валидация параметров не удается
    
    Атрибуты:
        validation_errors: Список конкретных ошибок валидации
        invalid_data: Данные, которые не прошли валидацию
        schema_info: Информация об ожидаемой схеме
    """
    
    def __init__(
        self,
        message: str,
        validation_errors: Optional[List[Dict[str, Any]]] = None,
        invalid_data: Optional[Any] = None,
        schema_info: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        """
        Инициализировать ошибку валидации.
        
        Args:
            message: Человекочитаемое сообщение об ошибке
            validation_errors: Список конкретных ошибок валидации
            invalid_data: Данные, которые не прошли валидацию
            schema_info: Информация об ожидаемой схеме
            context: Дополнительная контекстная информация
            original_error: Исходное исключение, которое вызвало эту ошибку
        """
        super().__init__(message, context, original_error)
        self.validation_errors = validation_errors or []
        self.invalid_data = invalid_data
        self.schema_info = schema_info
        
        # Добавляем специфичный для валидации контекст
        if validation_errors:
            self.add_context("validation_errors", validation_errors)
        if schema_info:
            self.add_context("schema_info", schema_info)


class PydanticValidationError(ValidationError):
    """
    Исключение, возникающее при неудаче валидации модели Pydantic.
    
    Это специализированная ошибка валидации для специфичных для Pydantic
    неудач валидации, обычно при преобразовании ответов LLM в структурированные модели.
    
    Атрибуты:
        model_name: Имя модели Pydantic, которая не прошла валидацию
        field_errors: Словарь ошибок, специфичных для полей
    """
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        field_errors: Optional[Dict[str, List[str]]] = None,
        validation_errors: Optional[List[Dict[str, Any]]] = None,
        invalid_data: Optional[Any] = None,
        **kwargs,
    ):
        """
        Инициализировать ошибку валидации Pydantic.
        
        Args:
            message: Человекочитаемое сообщение об ошибке
            model_name: Имя модели Pydantic, которая не прошла валидацию
            field_errors: Словарь ошибок, специфичных для полей
            validation_errors: Список конкретных ошибок валидации
            invalid_data: Данные, которые не прошли валидацию
            **kwargs: Дополнительные аргументы, передаваемые в ValidationError
        """
        super().__init__(
            message,
            validation_errors=validation_errors,
            invalid_data=invalid_data,
            **kwargs,
        )
        self.model_name = model_name
        self.field_errors = field_errors or {}
        
        if model_name:
            self.add_context("model_name", model_name)
        if field_errors:
            self.add_context("field_errors", field_errors)


class JSONValidationError(ValidationError):
    """
    Исключение, возникающее при неудаче парсинга или валидации JSON.
    
    Это происходит когда:
    - Ответ LLM не является валидным JSON
    - JSON не соответствует ожидаемой схеме
    - JSON содержит недопустимые типы данных
    
    Атрибуты:
        json_data: Недопустимые JSON данные (если парсируемые)
        parse_error: Сообщение об ошибке парсинга JSON
    """
    
    def __init__(
        self,
        message: str,
        json_data: Optional[Union[str, Dict[str, Any]]] = None,
        parse_error: Optional[str] = None,
        **kwargs,
    ):
        """
        Инициализировать ошибку валидации JSON.
        
        Args:
            message: Человекочитаемое сообщение об ошибке
            json_data: Недопустимые JSON данные
            parse_error: Сообщение об ошибке парсинга JSON
            **kwargs: Дополнительные аргументы, передаваемые в ValidationError
        """
        super().__init__(message, **kwargs)
        self.json_data = json_data
        self.parse_error = parse_error
        
        if parse_error:
            self.add_context("parse_error", parse_error)


class SchemaValidationError(ValidationError):
    """
    Исключение, возникающее когда данные не соответствуют ожидаемой схеме.
    
    Это происходит когда:
    - Структура ответа не соответствует ожидаемому формату
    - Отсутствуют обязательные поля
    - Типы полей не соответствуют схеме
    
    Атрибуты:
        expected_schema: Ожидаемая схема
        actual_data: Фактические данные, которые были валидированы
        schema_path: Путь в схеме, где валидация не удалась
    """
    
    def __init__(
        self,
        message: str,
        expected_schema: Optional[Dict[str, Any]] = None,
        actual_data: Optional[Any] = None,
        schema_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Инициализировать ошибку валидации схемы.
        
        Args:
            message: Человекочитаемое сообщение об ошибке
            expected_schema: Ожидаемая схема
            actual_data: Фактические данные, которые были валидированы
            schema_path: Путь в схеме, где валидация не удалась
            **kwargs: Дополнительные аргументы, передаваемые в ValidationError
        """
        super().__init__(message, **kwargs)
        self.expected_schema = expected_schema
        self.actual_data = actual_data
        self.schema_path = schema_path
        
        if expected_schema:
            self.add_context("expected_schema", expected_schema)
        if schema_path:
            self.add_context("schema_path", schema_path)


class ParameterValidationError(ValidationError):
    """
    Исключение, возникающее когда параметры функции или метода недопустимы.
    
    Это происходит когда:
    - Отсутствуют обязательные параметры
    - Типы параметров неправильные
    - Значения параметров выходят за пределы допустимого диапазона
    
    Атрибуты:
        parameter_name: Имя недопустимого параметра
        parameter_value: Значение недопустимого параметра
        valid_values: Список допустимых значений (если применимо)
    """
    
    def __init__(
        self,
        message: str,
        parameter_name: Optional[str] = None,
        parameter_value: Optional[Any] = None,
        valid_values: Optional[List[Any]] = None,
        **kwargs,
    ):
        """
        Инициализировать ошибку валидации параметра.
        
        Args:
            message: Человекочитаемое сообщение об ошибке
            parameter_name: Имя недопустимого параметра
            parameter_value: Значение недопустимого параметра
            valid_values: Список допустимых значений (если применимо)
            **kwargs: Дополнительные аргументы, передаваемые в ValidationError
        """
        super().__init__(message, **kwargs)
        self.parameter_name = parameter_name
        self.parameter_value = parameter_value
        self.valid_values = valid_values
        
        if parameter_name:
            self.add_context("parameter_name", parameter_name)
        if parameter_value is not None:
            self.add_context("parameter_value", parameter_value)
        if valid_values:
            self.add_context("valid_values", valid_values)