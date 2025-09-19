"""
Фильтрация и перегенерация ответов на основе уверенности (logprobs).

Этот модуль предоставляет высокоуровневые функции, которые делают повторные
запросы к LLM до достижения целевого порога уверенности по всему ответу и/или
по отдельным токенам в потоковом режиме.

Использование:
    from kraken_llm.confidence.filter import ConfidenceFilterConfig, ensure_confident_chat
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..client.base import BaseLLMClient
from .metrics import classify_confidence


@dataclass
class ConfidenceFilterConfig:
    """Конфигурация фильтрации по уверенности.

    - min_confidence: целевой порог по средней уверенности (0..1)
    - max_attempts: максимум попыток перегенерации
    - start_temperature: начальная температура для первой попытки
    - temperature_step: шаг изменения температуры между попытками
    - max_temperature: верхняя граница температуры
    - per_token_threshold: порог низкой уверенности для отдельного токена (для stream)
    - max_low_conf_fraction: допустимая доля низкоуверенных токенов (0..1). Если None — не учитывается
    - max_low_conf_count: допустимое число низкоуверенных токенов. Если None — не учитывается
    - prefer_streaming: попробовать потоковый режим для получения пер-токенных метрик
    """

    min_confidence: float = 0.7
    max_attempts: int = 3

    start_temperature: float = 0.7
    temperature_step: float = 0.1
    max_temperature: float = 1.5

    per_token_threshold: float = 0.4
    max_low_conf_fraction: Optional[float] = None
    max_low_conf_count: Optional[int] = None

    prefer_streaming: bool = False


def _next_temperature(cfg: ConfidenceFilterConfig, attempt_index: int) -> float:
    """Рассчитать температуру для текущей попытки.

    attempt_index начинается с 0.
    """
    t = cfg.start_temperature + attempt_index * cfg.temperature_step
    return min(t, cfg.max_temperature)


def _is_token_profile_acceptable(
    cfg: ConfidenceFilterConfig,
    token_confidences: Optional[List[Dict[str, Any]]],
) -> bool:
    """Проверить, что профиль пер-токенной уверенности укладывается в ограничения.

    - Считаем токены с уверенностью < per_token_threshold как "низкоуверенные".
    - Проверяем долю и/или абсолютное число таких токенов.
    """
    if not token_confidences:
        return True  # Нет данных — считаем допустимым

    total = len(token_confidences)
    if total == 0:
        return True

    low = sum(1 for t in token_confidences if (t.get("confidence") or 0.0) < cfg.per_token_threshold)

    if cfg.max_low_conf_count is not None and low > cfg.max_low_conf_count:
        return False

    if cfg.max_low_conf_fraction is not None:
        frac = low / total if total > 0 else 0.0
        if frac > cfg.max_low_conf_fraction:
            return False

    return True


async def ensure_confident_chat(
    client: BaseLLMClient,
    messages: List[Dict[str, str]],
    cfg: ConfidenceFilterConfig,
    **kwargs,
) -> Dict[str, Any]:
    """Выполнить chat completion с перегенерацией до достижения порога уверенности.

    Возвращает словарь со структурой:
    {
        "text": str,
        "confidence": float,
        "confidence_label": str,
        "attempts_made": int,
        "success": bool,
        "all_attempts": [
            {"attempt": int, "temperature": float, "text": str, "confidence": float, "confidence_label": str}
        ],
        # Для потокового режима дополнительно могут присутствовать поля:
        # "token_confidences": [...], "total_tokens": int
    }

    Замечания:
    - Если prefer_streaming=True, функция попробует получить пер-токенные метрики, но
      фактически выполнит агрегированную генерацию (не по чанкам), чтобы собрать
      метрики из streaming ответа. Это безопасно для последующей фильтрации.
    """
    all_attempts: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    best_conf: float = -1.0

    # Определяем, пробовать ли streaming ради пер‑токенных метрик
    stream_flag = bool(kwargs.get("stream", False) or cfg.prefer_streaming)

    for i in range(cfg.max_attempts):
        temperature = _next_temperature(cfg, i)

        # Параметры запроса: просим вернуть метрики уверенности
        params = {
            # Не передаем "stream" в стандартный клиент, чтобы не ломать non-stream ветку
            **{k: v for k, v in kwargs.items() if k != "stream"},
            "temperature": temperature,
            "include_confidence": True,
            # Если хотим пер‑токенные метрики, предпочтем нативный OpenAI stream у клиента
            "prefer_openai_streaming": True if cfg.prefer_streaming else kwargs.get("prefer_openai_streaming", None),
        }

        # Выполняем запрос
        try:
            result = await client.chat_completion(messages=messages, **params)
        except Exception as e:
            # Ошибка запроса — фиксируем попытку и продолжаем
            attempt_info = {
                "attempt": i + 1,
                "temperature": temperature,
                "text": f"Ошибка: {e}",
                "confidence": 0.0,
                "confidence_label": "Ошибка",
            }
            all_attempts.append(attempt_info)
            continue

        # Ожидаем словарь от include_confidence=True
        if not isinstance(result, dict) or "text" not in result:
            # Нестандартный ответ — приводим к минимальному виду
            text = result if isinstance(result, str) else str(result)
            conf = 0.5
            label = classify_confidence(conf)
            current = {"text": text, "confidence": conf, "confidence_label": label}
        else:
            current = result

        conf = float(current.get("confidence", 0.0))
        label = current.get("confidence_label", classify_confidence(conf))

        # Запоминаем попытку
        attempt_info = {
            "attempt": i + 1,
            "temperature": temperature,
            "text": current.get("text", ""),
            "confidence": conf,
            "confidence_label": label,
        }
        all_attempts.append(attempt_info)

        # Обновляем лучший результат
        if conf > best_conf:
            best_conf = conf
            best = current

        # Проверяем пер‑токенные ограничения, если есть данные
        token_ok = True
        if cfg.prefer_streaming:
            token_ok = _is_token_profile_acceptable(
                cfg, current.get("token_confidences")  # type: ignore[arg-type]
            )

        # Условие успеха
        if conf >= cfg.min_confidence and token_ok:
            return {
                **current,
                "attempts_made": i + 1,
                "success": True,
                "all_attempts": all_attempts,
            }

    # Порог не достигнут — возвращаем лучший из полученных
    final = best or {"text": "", "confidence": 0.0, "confidence_label": classify_confidence(0.0)}
    return {
        **final,
        "attempts_made": cfg.max_attempts,
        "success": False,
        "all_attempts": all_attempts,
    }
