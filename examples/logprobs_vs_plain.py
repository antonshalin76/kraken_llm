#!/usr/bin/env python3
"""
Пример: сравнение обычной генерации, генерации с logprobs (уверенность)
и генерации с фильтрацией/перегенерацией по порогу уверенности.

Запуск:
    python3 examples/logprobs_vs_plain.py

Требования окружения:
    LLM_ENDPOINT, LLM_API_KEY (или LLM_TOKEN), LLM_MODEL
"""
import asyncio
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

from kraken_llm import LLMConfig, create_standard_client
from kraken_llm.confidence.filter import ConfidenceFilterConfig, ensure_confident_chat

load_dotenv()

def _build_messages(user_text: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": user_text}]


async def demo_plain_vs_logprobs():
    print("=== Сравнение: обычная генерация vs logprobs ===")

    # Конфиг без принудительного logprobs (по умолчанию)
    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_API_KEY"),
        model=os.getenv("LLM_MODEL"),
        # Можно включить глобально:
        # logprobs=True,
        # top_logprobs=5,
    )

    async with create_standard_client(config) as client:
        messages = _build_messages("Кратко объясни, что такое машинное обучение")

        # Обычная генерация (строка)
        plain = await client.chat_completion(messages, max_tokens=512)
        print("\n— Обычный ответ —")
        print(plain[:300], ("..." if len(plain) > 300 else ""))

        # Генерация с метриками уверенности (включаем include_confidence)
        with_conf = await client.chat_completion(
            messages,
            include_confidence=True,
            prefer_openai_streaming=True,
            max_tokens=512,
        )
        print("\n— С logprobs (уверенность) —")
        print(with_conf["text"][:300], ("..." if len(with_conf["text"]) > 300 else ""))
        print(
            f"Уверенность: {with_conf['confidence']:.3f} ({with_conf['confidence_label']})"
        )


async def demo_with_confidence_filter():
    print("\n=== Фильтрация/перегенерация по уверенности ===")

    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_API_KEY"),
        model=os.getenv("LLM_MODEL"),
    )

    async with create_standard_client(config) as client:
        messages = _build_messages("Дай краткое объяснение ИИ простыми словами")

        # Настройки фильтрации: целевая уверенность 0.8, до 3 попыток
        filt_cfg = ConfidenceFilterConfig(
            min_confidence=0.95,
            max_attempts=3,
            prefer_streaming=True,       # Попробовать собрать пер‑токенные метрики
            per_token_threshold=0.8,
            max_low_conf_fraction=0.1   # Макс. доля токенов с низкой уверенностью
        )

        result = await ensure_confident_chat(
            client,
            messages=messages,
            cfg=filt_cfg,
            max_tokens=600,
        )

        print("\n— Итог после фильтрации —")
        print(result["text"][:300], ("..." if len(result["text"]) > 300 else ""))
        print(
            f"Уверенность: {result['confidence']:.3f} ({result['confidence_label']});"
            f" попыток: {result['attempts_made']}; успех: {result['success']}"
        )

        if "total_tokens" in result:
            print(f"Токенов учтено: {result['total_tokens']}")


async def main():
    await demo_plain_vs_logprobs()
    await demo_with_confidence_filter()


if __name__ == "__main__":
    asyncio.run(main())
