#!/usr/bin/env python3
"""
Пример: потоковая генерация с logprobs и сравнение с обычной генерацией.

Запуск:
    python3 examples/logprobs_streaming_example.py
"""
import asyncio
import os
from typing import List, Dict

from dotenv import load_dotenv
from kraken_llm import LLMConfig, create_streaming_client


def _msgs(txt: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": txt}]


async def main():
    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT", "http://localhost:8080"),
        api_key=os.getenv("LLM_API_KEY") or os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL", "chat"),
    )

    async with create_streaming_client(config) as client:
        messages = _msgs("Объясни простыми словами, что такое нейросеть")

        print("=== Потоковая генерация: без сбора уверенности ===")
        chunks = []
        async for ch in client.chat_completion_stream(messages, max_tokens=150):
            chunks.append(ch)
        plain_text = "".join(chunks)
        print(plain_text[:300], ("..." if len(plain_text) > 300 else ""))

        print("\n=== Потоковая генерация: с logprobs (агрегация уверенности) ===")
        with_conf = await client.chat_completion(
            messages,
            include_confidence=True,
            max_tokens=150,
        )
        print(with_conf["text"][:300], ("..." if len(with_conf["text"]) > 300 else ""))
        print(
            f"Уверенность: {with_conf['confidence']:.3f} ({with_conf['confidence_label']});"
            f" токенов учтено: {with_conf.get('total_tokens', 0)}"
        )


if __name__ == "__main__":
    load_dotenv(".env")
    asyncio.run(main())
