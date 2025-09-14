"""
Управление конфигурацией Kraken

Этот модуль обрабатывает конфигурацию через Pydantic Settings с поддержкой:
- Переменных окружения
- .env файлов
- Значений по умолчанию с описаниями
"""

from .settings import LLMConfig
from .defaults import *

__all__ = [
    "LLMConfig",
]