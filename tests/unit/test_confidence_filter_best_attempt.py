"""
Юнит‑тест: фильтрация по уверенности возвращает лучшую попытку,
если порог не достигнут за заданное число попыток.

Сценарий:
- Три попытки с уверенностьями: [0.4, 0.6, 0.55]
- Порог min_confidence = 0.7
Ожидаем:
- success=False
- attempts_made=3
- возвращён ответ второй попытки (conf=0.6) как лучший
- all_attempts содержит все 3 попытки в порядке выполнения
"""
import pytest

from kraken_llm.confidence.filter import ConfidenceFilterConfig, ensure_confident_chat
from kraken_llm.confidence.metrics import classify_confidence


class MockClient:
    def __init__(self, confidences):
        self.confidences = confidences
        self.calls = 0

    async def chat_completion(self, messages, **kwargs):
        # Выдаём по очереди заготовленные confidence значения
        idx = self.calls
        self.calls += 1
        conf = self.confidences[idx]
        return {
            "text": f"resp-{idx + 1}",
            "confidence": conf,
            "confidence_label": classify_confidence(conf),
        }


@pytest.mark.asyncio
async def test_confidence_filter_returns_best_when_threshold_not_met():
    # Настраиваем фильтр: 3 попытки, порог 0.7
    cfg = ConfidenceFilterConfig(
        min_confidence=0.7,
        max_attempts=3,
        prefer_streaming=False,
    )

    # Мокаем клиента с тремя попытками: 0.4, 0.6, 0.55
    client = MockClient([0.4, 0.6, 0.55])

    result = await ensure_confident_chat(
        client,
        messages=[{"role": "user", "content": "test"}],
        cfg=cfg,
    )

    assert isinstance(result, dict)
    assert result["success"] is False
    assert result["attempts_made"] == 3

    # Проверяем, что выбрана лучшая попытка (вторая, conf=0.6)
    assert pytest.approx(result["confidence"], rel=1e-6) == 0.6
    assert result["text"] == "resp-2"

    # Проверяем all_attempts
    attempts = result.get("all_attempts", [])
    assert len(attempts) == 3
    assert [round(a["confidence"], 3) for a in attempts] == [0.4, 0.6, 0.55]
