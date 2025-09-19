"""
Пакет метрик уверенности и фильтрации.

Публичный API:
- kraken_llm.confidence.metrics
- kraken_llm.confidence.filter
"""

from .metrics import classify_confidence, confidence_from_chat_logprobs, token_confidences_from_stream_logprobs
from .filter import ConfidenceFilterConfig, ensure_confident_chat
