#!/usr/bin/env python3
"""Простейший чат с LLM"""
import asyncio, sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent / "src"))
from kraken_llm import create_universal_client, UniversalCapability


async def main():
    print("🤖 Kraken Chat (exit для выхода)")

    async with create_universal_client(
        capabilities={
            UniversalCapability.CHAT_COMPLETION,
            UniversalCapability.STREAMING,
            UniversalCapability.REASONING,
        }
    ) as client:

        while True:
            user_input = input("👤 ").strip()
            if user_input.lower() == "exit" or "q" or "й":
                break
            if not user_input:
                continue

            print("🤖 ", end="", flush=True)
            try:
                async for chunk in client.chat_completion_stream(
                    [{"role": "user", "content": user_input}], max_tokens=8192
                ):
                    print(chunk, end="", flush=True)
                print()
            except Exception as e:
                print(f"Ошибка: {e}")


if __name__ == "__main__":
    asyncio.run(main())
