"""
Утилиты логирования для Kraken LLM Framework.

Этот модуль предоставляет конфигурацию логирования и утилиты для согласованного
логирования во всем фреймворке.
"""

import logging
import sys
from typing import Optional

from ..config.defaults import DEFAULT_LOG_LEVEL, DEFAULT_LOG_FORMAT


def setup_logging(
    level: str = DEFAULT_LOG_LEVEL,
    format_string: str = DEFAULT_LOG_FORMAT,
    logger_name: str = "kraken",
) -> logging.Logger:
    """
    Настроить логирование для Kraken LLM Framework.
    
    Args:
        level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Строка формата сообщений лога
        logger_name: Имя логгера для настройки
        
    Returns:
        Настроенный экземпляр логгера
    """
    # Получаем или создаем логгер
    logger = logging.getLogger(logger_name)
    
    # Не добавляем обработчики, если они уже существуют
    if logger.handlers:
        return logger
    
    # Устанавливаем уровень логирования
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    # Создаем обработчик консоли
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)
    
    # Создаем форматтер
    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)
    
    # Добавляем обработчик к логгеру
    logger.addHandler(handler)
    
    # Предотвращаем распространение к корневому логгеру, чтобы избежать дублирования сообщений
    logger.propagate = False
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Получить экземпляр логгера для компонентов Kraken.
    
    Args:
        name: Имя логгера (будет с префиксом 'kraken.')
        
    Returns:
        Экземпляр логгера
    """
    if name:
        logger_name = f"kraken.{name}"
    else:
        logger_name = "kraken"
    
    return logging.getLogger(logger_name)


def log_exception(
    logger: logging.Logger,
    exception: Exception,
    message: str = "Произошла ошибка",
    level: int = logging.ERROR,
) -> None:
    """
    Логировать исключение с полной информацией трассировки.
    
    Args:
        logger: Экземпляр логгера для использования
        exception: Исключение для логирования
        message: Дополнительное сообщение для включения
        level: Уровень логирования для использования
    """
    logger.log(
        level,
        f"{message}: {exception}",
        exc_info=True,
        extra={
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
        },
    )


def log_api_request(
    logger: logging.Logger,
    method: str,
    url: str,
    headers: Optional[dict] = None,
    data: Optional[dict] = None,
) -> None:
    """
    Логировать API запрос с соответствующими деталями.
    
    Args:
        logger: Экземпляр логгера для использования
        method: HTTP метод
        url: URL запроса
        headers: Заголовки запроса (чувствительные данные будут замаскированы)
        data: Данные запроса (будут обрезаны, если слишком длинные)
    """
    # Маскируем чувствительные заголовки
    safe_headers = {}
    if headers:
        for key, value in headers.items():
            if key.lower() in ("authorization", "api-key", "x-api-key"):
                safe_headers[key] = "***"
            else:
                safe_headers[key] = value
    
    # Обрезаем данные, если они слишком длинные
    safe_data = data
    if data and len(str(data)) > 500:
        safe_data = f"{str(data)[:500]}... (обрезано)"
    
    logger.debug(
        f"API Запрос: {method} {url}",
        extra={
            "method": method,
            "url": url,
            "headers": safe_headers,
            "data": safe_data,
        },
    )


def log_api_response(
    logger: logging.Logger,
    status_code: int,
    response_time: float,
    response_size: Optional[int] = None,
    error: Optional[str] = None,
) -> None:
    """
    Логировать API ответ с соответствующими деталями.
    
    Args:
        logger: Экземпляр логгера для использования
        status_code: HTTP статус код
        response_time: Время ответа в секундах
        response_size: Размер ответа в байтах
        error: Сообщение об ошибке, если запрос не удался
    """
    if error:
        logger.warning(
            f"API Ответ: {status_code} (ошибка: {error})",
            extra={
                "status_code": status_code,
                "response_time": response_time,
                "error": error,
            },
        )
    else:
        logger.debug(
            f"API Ответ: {status_code} ({response_time:.3f}s)",
            extra={
                "status_code": status_code,
                "response_time": response_time,
                "response_size": response_size,
            },
        )