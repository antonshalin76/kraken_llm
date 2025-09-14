"""
Structured output модуль для Kraken фреймворка.

Этот модуль предоставляет инструменты для работы со структурированными ответами
от языковых моделей, включая валидацию через Pydantic модели и интеграцию с Outlines.
"""

from .validator import (
    StructuredOutputValidator,
    validator,
    validate_structured_response,
    check_model_compatibility,
    create_model_example,
)

__all__ = [
    "StructuredOutputValidator",
    "validator",
    "validate_structured_response", 
    "check_model_compatibility",
    "create_model_example",
]