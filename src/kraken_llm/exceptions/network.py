"""
Исключения, связанные с сетью, для Kraken LLM Framework.

Этот модуль определяет исключения, связанные с сетевым подключением, таймаутами
и проблемами HTTP-коммуникации.
"""

from typing import Any, Dict, Optional

from .base import KrakenError


class NetworkError(KrakenError):
    """
    Базовый класс для ошибок, связанных с сетью.
    
    Это исключение возникает при проблемах с сетевым подключением,
    HTTP-запросами или коммуникацией с эндпоинтом LLM API.
    
    Атрибуты:
        endpoint: Эндпоинт, к которому происходило обращение при возникновении ошибки
        status_code: HTTP статус код (если применимо)
        response_data: Данные ответа от сервера (если доступны)
    """
    
    def __init__(
        self,
        message: str,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        """
        Инициализировать сетевую ошибку.
        
        Args:
            message: Человекочитаемое сообщение об ошибке
            endpoint: Эндпоинт, к которому происходило обращение
            status_code: HTTP статус код (если применимо)
            response_data: Данные ответа от сервера
            context: Дополнительная контекстная информация
            original_error: Исходное исключение, которое вызвало эту ошибку
        """
        super().__init__(message, context, original_error)
        self.endpoint = endpoint
        self.status_code = status_code
        self.response_data = response_data
        
        # Добавляем специфичный для сети контекст
        if endpoint:
            self.add_context("endpoint", endpoint)
        if status_code:
            self.add_context("status_code", status_code)
        if response_data:
            self.add_context("response_data", response_data)


class ConnectionError(NetworkError):
    """
    Исключение, возникающее при невозможности установить соединение с LLM API.
    
    Обычно это указывает на:
    - Недоступность эндпоинта
    - Неудачу разрешения DNS
    - Проблемы с сетевым подключением
    - Блокировку соединения файрволом
    """
    
    def __init__(
        self,
        message: str = "Не удалось установить соединение с LLM API",
        endpoint: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, endpoint=endpoint, **kwargs)


class TimeoutError(NetworkError):
    """
    Исключение, возникающее при превышении времени ожидания запроса.
    
    Это может произойти во время:
    - Установления соединения (таймаут подключения)
    - Ожидания ответа (таймаут чтения)
    - Отправки данных запроса (таймаут записи)
    
    Атрибуты:
        timeout_type: Тип таймаута, который произошел
        timeout_value: Значение таймаута, которое было превышено
    """
    
    def __init__(
        self,
        message: str = "Превышено время ожидания запроса",
        endpoint: Optional[str] = None,
        timeout_type: Optional[str] = None,
        timeout_value: Optional[float] = None,
        **kwargs,
    ):
        """
        Инициализировать ошибку таймаута.
        
        Args:
            message: Человекочитаемое сообщение об ошибке
            endpoint: Эндпоинт, к которому происходило обращение
            timeout_type: Тип таймаута ('connect', 'read', 'write')
            timeout_value: Значение таймаута, которое было превышено
            **kwargs: Дополнительные аргументы, передаваемые в NetworkError
        """
        super().__init__(message, endpoint=endpoint, **kwargs)
        self.timeout_type = timeout_type
        self.timeout_value = timeout_value
        
        if timeout_type:
            self.add_context("timeout_type", timeout_type)
        if timeout_value:
            self.add_context("timeout_value", timeout_value)


class HTTPError(NetworkError):
    """
    Исключение, возникающее для HTTP-специфичных ошибок.
    
    Это включает:
    - 4xx ошибки клиента (неверный запрос, неавторизован и т.д.)
    - 5xx ошибки сервера (внутренняя ошибка сервера, сервис недоступен и т.д.)
    - Неверные HTTP ответы
    
    Атрибуты:
        method: HTTP метод, который использовался
        url: Полный URL, который запрашивался
        headers: Заголовки запроса, которые были отправлены
    """
    
    def __init__(
        self,
        message: str,
        status_code: int,
        endpoint: Optional[str] = None,
        method: Optional[str] = None,
        url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        response_data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Инициализировать HTTP ошибку.
        
        Args:
            message: Человекочитаемое сообщение об ошибке
            status_code: HTTP статус код
            endpoint: Эндпоинт, к которому происходило обращение
            method: HTTP метод, который использовался
            url: Полный URL, который запрашивался
            headers: Заголовки запроса, которые были отправлены
            response_data: Данные ответа от сервера
            **kwargs: Дополнительные аргументы, передаваемые в NetworkError
        """
        super().__init__(
            message,
            endpoint=endpoint,
            status_code=status_code,
            response_data=response_data,
            **kwargs,
        )
        self.method = method
        self.url = url
        self.headers = headers
        
        if method:
            self.add_context("method", method)
        if url:
            self.add_context("url", url)
        if headers:
            self.add_context("headers", headers)


class SSLError(NetworkError):
    """
    Исключение, возникающее для ошибок, связанных с SSL/TLS.
    
    Это включает:
    - Неудачи проверки сертификата
    - Ошибки SSL handshake
    - Несоответствия версий протокола
    """
    
    def __init__(
        self,
        message: str = "Произошла ошибка SSL/TLS",
        endpoint: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, endpoint=endpoint, **kwargs)