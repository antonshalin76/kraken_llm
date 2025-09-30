"""
Модуль для поддержки потоковых операций в Kraken фреймворке.

Этот модуль содержит компоненты для обработки потоковых ответов от LLM API,
включая обработку Server-Sent Events, агрегацию контента и управление
потоковыми данными.
"""

from .handler import StreamHandler, StreamAggregator
from .repair import token_stream_with_shadow_repair

__all__ = [
    "StreamHandler",
    "StreamAggregator",
    "token_stream_with_shadow_repair",
]