#!/usr/bin/env python3
"""
Пример: использование UniversalLLMClient с logprobs и фильтрацией по уверенности.

Запуск:
    python3 examples/universal_logprobs_example.py
"""
import asyncio
import os
from typing import List, Dict

from dotenv import load_dotenv
from kraken_llm import LLMConfig, create_universal_client
from kraken_llm.confidence.filter import ConfidenceFilterConfig, ensure_confident_chat


def _msgs(text: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": text}]


async def main():
    load_dotenv(".env")

    # Универсальный клиент автоматически выбирает оптимальный под‑клиент
    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_API_KEY"),
        model=os.getenv("LLM_MODEL"),
    )

    async with create_universal_client(config) as client:
        messages = _msgs("Сформулируй краткое, простое объяснение машинного обучения")

        print("=== UniversalLLMClient: обычная генерация ===")
        plain = await client.chat_completion(messages, max_tokens=200)
        print(plain[:300], ("..." if len(plain) > 300 else ""))

        print("\n=== UniversalLLMClient: с logprobs (уверенность) ===")
        with_conf = await client.chat_completion(messages, include_confidence=True, prefer_openai_streaming=True, max_tokens=200)
        print(with_conf["text"][:300], ("..." if len(with_conf["text"]) > 300 else ""))
        print(f"Уверенность: {with_conf['confidence']:.3f} ({with_conf['confidence_label']})")

        print("\n=== UniversalLLMClient: фильтрация/перегенерация по уверенности ===")
        filt_cfg = ConfidenceFilterConfig(
            min_confidence=0.95,
            max_attempts=5,
            prefer_streaming=True,       # Собирать пер‑токенные метрики
            per_token_threshold=0.8,    # Уверенность на токен
            max_low_conf_fraction=0.1,  # Макс. доля низкой уверенности
        )
        filtered = await ensure_confident_chat(
            client,
            messages=messages,
            cfg=filt_cfg,
            max_tokens=220,
        )
        print(filtered["text"][:300], ("..." if len(filtered["text"]) > 300 else ""))
        print(
            f"Уверенность: {filtered['confidence']:.3f} ({filtered['confidence_label']});"
            f" попыток: {filtered['attempts_made']}; успех: {filtered['success']}"
        )


if __name__ == "__main__":
    asyncio.run(main())
