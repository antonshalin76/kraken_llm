"""
Утилиты для вычисления метрик уверенности по logprobs, возвращаемым провайдерами LLM.

Модуль содержит небольшие, нетребовательные по зависимостям функции, работающие с
OpenAI‑совместимой структурой logprobs для chat (choice.logprobs.content), а также с
потоковыми чанками. Сетевые вызовы не выполняются.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def classify_confidence(conf: float) -> str:
    """Сопоставить значение уверенности/вероятности [0,1] человеко‑читаемой метке.

    Пороговые значения выбраны простыми и интуитивными.
    """
    if conf >= 0.9:
        return "Очень высокая"
    elif conf >= 0.7:
        return "Высокая"
    elif conf >= 0.5:
        return "Средняя"
    elif conf >= 0.3:
        return "Низкая"
    else:
        return "Очень низкая"


def confidence_from_chat_logprobs(logprobs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Посчитать агрегированные метрики уверенности по OpenAI chat logprobs.

    Ожидаемая структура:
    choice.logprobs = {
        "content": [
            {"token": "...", "logprob": float, "top_logprobs": [{"token": str, "logprob": float}, ...]},
            ...
        ]
    }

    Возвращает словарь с ключами:
    - average_confidence — средняя уверенность по токенам
    - min_confidence — минимальная уверенность
    - max_confidence — максимальная уверенность
    - confidence_std — стандартное отклонение уверенности
    - confidence_label — текстовая метка уровня уверенности
    - token_count — количество учтённых токенов
    - token_details — подробности по каждому токену (опционально)
    """
    if not logprobs or "content" not in logprobs:
        return {
            "average_confidence": 0.5,
            "min_confidence": 0.5,
            "max_confidence": 0.5,
            "confidence_std": 0.0,
            "confidence_label": "Нет logprobs",
            "token_count": 0,
            "token_details": [],
        }

    probs: List[float] = []
    token_details: List[Dict[str, Any]] = []

    for token_data in logprobs.get("content", []) or []:
        lp = token_data.get("logprob")
        if lp is None:
            continue
        p = float(np.exp(lp))
        probs.append(p)
        token_details.append(
            {
                "token": token_data.get("token", ""),
                "confidence": p,
                "logprob": lp,
            }
        )

    if not probs:
        return {
            "average_confidence": 0.5,
            "min_confidence": 0.5,
            "max_confidence": 0.5,
            "confidence_std": 0.0,
            "confidence_label": "Ошибка расчёта",
            "token_count": 0,
            "token_details": [],
        }

    avg = float(np.mean(probs))
    metrics = {
        "average_confidence": avg,
        "min_confidence": float(np.min(probs)),
        "max_confidence": float(np.max(probs)),
        "confidence_std": float(np.std(probs)),
        "confidence_label": classify_confidence(avg),
        "token_count": len(probs),
        "token_details": token_details,
    }
    return metrics


def token_confidences_from_stream_logprobs(logprobs: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Извлечь пер‑токенную информацию об уверенности из streaming choice.logprobs.

    Возвращает список словарей с полями: token, confidence, confidence_label, logprob, alternatives.
    """
    results: List[Dict[str, Any]] = []
    if not logprobs or "content" not in logprobs:
        return results

    for token_data in logprobs.get("content", []) or []:
        lp = token_data.get("logprob")
        p = float(np.exp(lp)) if lp is not None else 0.0

        # Альтернативные токены (top_logprobs)
        alts: List[Dict[str, Any]] = []
        for alt in token_data.get("top_logprobs", []) or []:
            if "logprob" in alt:
                alts.append(
                    {
                        "token": alt.get("token", ""),
                        "confidence": float(np.exp(alt["logprob"])),
                        "logprob": alt["logprob"],
                    }
                )

        results.append(
            {
                "token": token_data.get("token", ""),
                "confidence": p,
                "confidence_label": classify_confidence(p),
                "logprob": lp,
                "alternatives": alts,
            }
        )

    return results
