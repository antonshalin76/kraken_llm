#!/usr/bin/env python3
"""
Простой тест CompletionLLMClient
"""

import asyncio
import os
from dotenv import load_dotenv

from kraken_llm.client.completion import CompletionLLMClient
from kraken_llm.config.settings import LLMConfig

load_dotenv()

async def test_completion_client():
    """Тестирование CompletionLLMClient"""
    print("=== Тестирование CompletionLLMClient ===")
    
    config = LLMConfig(
        endpoint=os.getenv("COMPLETION_ENDPOINT"),
        api_key=os.getenv("COMPLETION_TOKEN"),
        model=os.getenv("COMPLETION_MODEL"),
        temperature=0.7
    )
    
    print(f"Endpoint: {config.endpoint}")
    print(f"Model: {config.model}")
    
    async with CompletionLLMClient(config) as client:
        
        # Тест 1: Прямой text completion
        print("\n1. Тестирование text_completion_async:")
        try:
            response = await client.text_completion_async(
                prompt="Привет! Как дела?",
                max_tokens=50
            )
            print(f"Ответ: {response}")
        except Exception as e:
            print(f"Ошибка: {e}")
        
        # Тест 2: Chat completion (конвертация сообщений в промпт)
        print("\n2. Тестирование chat_completion:")
        try:
            messages = [
                {"role": "user", "content": "Скажи 'Работает'"}
            ]
            response = await client.chat_completion(messages, max_tokens=10)
            print(f"Ответ: {response}")
        except Exception as e:
            print(f"Ошибка: {e}")

if __name__ == "__main__":
    asyncio.run(test_completion_client())