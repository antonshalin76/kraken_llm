"""
Модуль для поддержки потоковых операций в Kraken фреймворке.

Этот модуль содержит компоненты для обработки потоковых ответов от LLM API,
включая обработку Server-Sent Events, агрегацию контента и управление
потоковыми данными.
"""

from .handler import StreamHandler, StreamAggregator

__all__ = [
    "StreamHandler",
    "StreamAggregator",
]