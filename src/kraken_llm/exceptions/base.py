"""
Базовые исключения для Kraken LLM Framework.

Этот модуль определяет базовую иерархию исключений, используемую во всем фреймворке.
Все специфичные для Kraken исключения наследуются от KrakenError.
"""

from typing import Any, Dict, Optional


class KrakenError(Exception):
    """
    Базовое исключение для всех ошибок Kraken LLM Framework.
    
    Это корневой класс исключений, от которого наследуются все остальные исключения Kraken.
    Он предоставляет общую функциональность для обработки ошибок и контекстной информации.
    
    Атрибуты:
        message: Человекочитаемое сообщение об ошибке
        context: Дополнительная контекстная информация об ошибке
        original_error: Исходное исключение, которое вызвало эту ошибку (если есть)
    """
    
    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        """
        Инициализировать ошибку Kraken.
        
        Args:
            message: Человекочитаемое сообщение об ошибке
            context: Дополнительная контекстная информация об ошибке
            original_error: Исходное исключение, которое вызвало эту ошибку
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.original_error = original_error
    
    def add_context(self, key: str, value: Any) -> "KrakenError":
        """
        Добавить контекстную информацию к ошибке.
        
        Args:
            key: Ключ контекста
            value: Значение контекста
            
        Returns:
            Self для цепочки методов
        """
        self.context[key] = value
        return self
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """
        Получить контекстную информацию из ошибки.
        
        Args:
            key: Ключ контекста
            default: Значение по умолчанию, если ключ не найден
            
        Returns:
            Значение контекста или значение по умолчанию
        """
        return self.context.get(key, default)
    
    def __str__(self) -> str:
        """Строковое представление ошибки."""
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (контекст: {context_str})"
        return self.message
    
    def __repr__(self) -> str:
        """Подробное строковое представление ошибки."""
        return (
            f"{self.__class__.__name__}("
            f"message='{self.message}', "
            f"context={self.context}, "
            f"original_error={self.original_error!r}"
            f")"
        )