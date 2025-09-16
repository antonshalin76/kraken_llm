#!/usr/bin/env python3
"""
Базовый пример обращения к LLM через клиента библиотеки Kraken LLM.
"""

import asyncio
from kraken_llm import create_universal_client, LLMConfig
import os
from dotenv import load_dotenv

load_dotenv()

config = LLMConfig(
        endpoint=os.getenv("LLM_REASONING_ENDPOINT"),
        api_key=os.getenv("LLM_REASONING_TOKEN"),
        model=os.getenv("LLM_REASONING_MODEL"),    
    )

async def main():
    async with create_universal_client(config) as client:
        response = await client.chat_completion([
            {"role": "user", "content": "Объясни что такое Python"}
        ])
        print(response)

asyncio.run(main())