"""
Исключения, связанные с API, для Kraken LLM Framework.

Этот модуль определяет исключения, связанные с ошибками LLM API, аутентификацией,
ограничением скорости и проблемами, специфичными для сервиса.
"""

from typing import Any, Dict, Optional

from .base import KrakenError


class APIError(KrakenError):
    """
    Исключение, возникающее для ошибок, специфичных для LLM API.
    
    Это исключение возникает, когда LLM API возвращает ответ с ошибкой
    или когда есть проблемы с API коммуникацией, не связанные с сетью.
    
    Атрибуты:
        status_code: HTTP статус код от API
        error_code: Специфичный для API код ошибки (если предоставлен)
        error_type: Тип API ошибки (если предоставлен)
        response_data: Полные данные ответа от API
        request_id: ID запроса для отслеживания (если предоставлен)
    """
    
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        error_code: Optional[str] = None,
        error_type: Optional[str] = None,
        response_data: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        """
        Инициализировать ошибку API.
        
        Args:
            message: Человекочитаемое сообщение об ошибке
            status_code: HTTP статус код от API
            error_code: Специфичный для API код ошибки
            error_type: Тип API ошибки
            response_data: Полные данные ответа от API
            request_id: ID запроса для отслеживания
            context: Дополнительная контекстная информация
            original_error: Исходное исключение, которое вызвало эту ошибку
        """
        super().__init__(message, context, original_error)
        self.status_code = status_code
        self.error_code = error_code
        self.error_type = error_type
        self.response_data = response_data
        self.request_id = request_id
        
        # Добавляем специфичный для API контекст
        if status_code:
            self.add_context("status_code", status_code)
        if error_code:
            self.add_context("error_code", error_code)
        if error_type:
            self.add_context("error_type", error_type)
        if request_id:
            self.add_context("request_id", request_id)


class AuthenticationError(APIError):
    """
    Исключение, возникающее при неудаче аутентификации API.
    
    Это происходит когда:
    - API ключ отсутствует или недействителен
    - Токен аутентификации истек
    - Недостаточно прав для запрашиваемой операции
    
    Атрибуты:
        auth_type: Тип аутентификации, который не удался
    """
    
    def __init__(
        self,
        message: str = "Authentication failed",
        auth_type: Optional[str] = None,
        **kwargs,
    ):
        """
        Инициализировать ошибку аутентификации.
        
        Args:
            message: Человекочитаемое сообщение об ошибке
            auth_type: Тип аутентификации, который не удался
            **kwargs: Дополнительные аргументы, передаваемые в APIError
        """
        super().__init__(message, status_code=401, **kwargs)
        self.auth_type = auth_type
        
        if auth_type:
            self.add_context("auth_type", auth_type)


class RateLimitError(APIError):
    """
    Исключение, возникающее при превышении лимитов скорости API.
    
    Это происходит когда:
    - Слишком много запросов в минуту/час/день
    - Превышены лимиты использования токенов
    - Превышены лимиты одновременных запросов
    
    Атрибуты:
        limit_type: Тип лимита скорости, который был превышен
        limit_value: Значение лимита, которое было превышено
        reset_time: Когда лимит скорости сбрасывается (если предоставлено)
        retry_after: Предлагаемая задержка повтора в секундах
    """
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        limit_type: Optional[str] = None,
        limit_value: Optional[int] = None,
        reset_time: Optional[int] = None,
        retry_after: Optional[int] = None,
        **kwargs,
    ):
        """
        Инициализировать ошибку лимита скорости.
        
        Args:
            message: Человекочитаемое сообщение об ошибке
            limit_type: Тип лимита скорости, который был превышен
            limit_value: Значение лимита, которое было превышено
            reset_time: Когда лимит скорости сбрасывается (временная метка)
            retry_after: Предлагаемая задержка повтора в секундах
            **kwargs: Дополнительные аргументы, передаваемые в APIError
        """
        super().__init__(message, status_code=429, **kwargs)
        self.limit_type = limit_type
        self.limit_value = limit_value
        self.reset_time = reset_time
        self.retry_after = retry_after
        
        if limit_type:
            self.add_context("limit_type", limit_type)
        if limit_value:
            self.add_context("limit_value", limit_value)
        if reset_time:
            self.add_context("reset_time", reset_time)
        if retry_after:
            self.add_context("retry_after", retry_after)


class ModelError(APIError):
    """
    Исключение, возникающее для ошибок, специфичных для модели.
    
    Это происходит когда:
    - Запрашиваемая модель недоступна
    - Модель перегружена или временно недоступна
    - Модель не поддерживает запрашиваемые функции
    
    Атрибуты:
        model_name: Имя модели, которая вызвала ошибку
        available_models: Список доступных моделей (если предоставлен)
    """
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        available_models: Optional[list] = None,
        **kwargs,
    ):
        """
        Инициализировать ошибку модели.
        
        Args:
            message: Человекочитаемое сообщение об ошибке
            model_name: Имя модели, которая вызвала ошибку
            available_models: Список доступных моделей
            **kwargs: Дополнительные аргументы, передаваемые в APIError
        """
        super().__init__(message, **kwargs)
        self.model_name = model_name
        self.available_models = available_models
        
        if model_name:
            self.add_context("model_name", model_name)
        if available_models:
            self.add_context("available_models", available_models)


class ContentFilterError(APIError):
    """
    Исключение, возникающее когда контент фильтруется API.
    
    Это происходит когда:
    - Входной контент нарушает политику контента
    - Выходной контент фильтруется для безопасности
    - Контент содержит запрещенный материал
    
    Атрибуты:
        filter_type: Тип фильтра контента, который был активирован
        filtered_content: Контент, который был отфильтрован (если безопасно включить)
    """
    
    def __init__(
        self,
        message: str = "Content was filtered",
        filter_type: Optional[str] = None,
        filtered_content: Optional[str] = None,
        **kwargs,
    ):
        """
        Инициализировать ошибку фильтра контента.
        
        Args:
            message: Человекочитаемое сообщение об ошибке
            filter_type: Тип фильтра контента, который был активирован
            filtered_content: Контент, который был отфильтрован
            **kwargs: Дополнительные аргументы, передаваемые в APIError
        """
        super().__init__(message, **kwargs)
        self.filter_type = filter_type
        self.filtered_content = filtered_content
        
        if filter_type:
            self.add_context("filter_type", filter_type)


class ServiceUnavailableError(APIError):
    """
    Исключение, возникающее когда API сервис недоступен.
    
    Это происходит когда:
    - API сервис отключен для обслуживания
    - Сервис испытывает высокую нагрузку
    - Временный сбой сервиса
    
    Атрибуты:
        service_status: Текущий статус сервиса (если предоставлен)
        estimated_recovery: Предполагаемое время восстановления (если предоставлено)
    """
    
    def __init__(
        self,
        message: str = "API service is currently unavailable",
        service_status: Optional[str] = None,
        estimated_recovery: Optional[str] = None,
        **kwargs,
    ):
        """
        Инициализировать ошибку недоступности сервиса.
        
        Args:
            message: Человекочитаемое сообщение об ошибке
            service_status: Текущий статус сервиса
            estimated_recovery: Предполагаемое время восстановления
            **kwargs: Дополнительные аргументы, передаваемые в APIError
        """
        super().__init__(message, status_code=503, **kwargs)
        self.service_status = service_status
        self.estimated_recovery = estimated_recovery
        
        if service_status:
            self.add_context("service_status", service_status)
        if estimated_recovery:
            self.add_context("estimated_recovery", estimated_recovery)