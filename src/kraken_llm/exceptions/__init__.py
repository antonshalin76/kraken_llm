"""
Пользовательские исключения Kraken

Специализированные исключения для различных типов ошибок:
- NetworkError: Проблемы с сетью и соединением
- ValidationError: Проблемы валидации данных
- APIError: Ошибки LLM API
- TimeoutError: Проблемы с таймаутом
"""

from .base import KrakenError
from .network import NetworkError, ConnectionError, TimeoutError, HTTPError, SSLError
from .validation import (
    ValidationError,
    PydanticValidationError,
    JSONValidationError,
    SchemaValidationError,
    ParameterValidationError,
)
from .api import (
    APIError,
    AuthenticationError,
    RateLimitError,
    ModelError,
    ContentFilterError,
    ServiceUnavailableError,
)

__all__ = [
    "KrakenError",
    "NetworkError",
    "ConnectionError",
    "TimeoutError",
    "HTTPError",
    "SSLError",
    "ValidationError",
    "PydanticValidationError",
    "JSONValidationError",
    "SchemaValidationError",
    "ParameterValidationError",
    "APIError",
    "AuthenticationError",
    "RateLimitError",
    "ModelError",
    "ContentFilterError",
    "ServiceUnavailableError",
]
