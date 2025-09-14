#!/usr/bin/env python3
"""
Простой тест подключения к LLM серверу.
"""

import asyncio
import os
import httpx

from kraken_llm.config.settings import LLMConfig


async def test_direct_connection():
    """Тест прямого подключения к серверу"""
    config = LLMConfig()
    
    print(f"Тестируем подключение к: {config.endpoint}")
    print(f"Токен: {config.api_key}")
    
    # Тест 1: Простой GET запрос
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{config.endpoint}/")
            print(f"GET /: {response.status_code}")
            if response.status_code != 404:
                print(f"Response: {response.text[:200]}")
        except Exception as e:
            print(f"Ошибка GET /: {e}")
    
    # Тест 2: Проверка эндпоинта chat/completions
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{config.endpoint}/v1/chat/completions")
            print(f"GET /v1/chat/completions: {response.status_code}")
        except Exception as e:
            print(f"Ошибка GET /v1/chat/completions: {e}")
    
    # Тест 3: POST запрос с Bearer токеном
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": config.model,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{config.endpoint}/v1/chat/completions",
                json=data,
                headers=headers,
                timeout=30.0
            )
            print(f"POST /v1/chat/completions (Bearer): {response.status_code}")
            if response.status_code != 200:
                print(f"Response: {response.text[:500]}")
        except Exception as e:
            print(f"Ошибка POST Bearer: {e}")
    
    # Тест 4: POST запрос с токеном в заголовке Authorization без Bearer
    headers = {
        "Authorization": config.api_key,
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{config.endpoint}/v1/chat/completions",
                json=data,
                headers=headers,
                timeout=30.0
            )
            print(f"POST /v1/chat/completions (Direct): {response.status_code}")
            if response.status_code != 200:
                print(f"Response: {response.text[:500]}")
        except Exception as e:
            print(f"Ошибка POST Direct: {e}")
    
    # Тест 5: POST запрос с токеном в параметре api_key
    data_with_key = {
        "model": config.model,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10,
        "api_key": config.api_key
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{config.endpoint}/v1/chat/completions",
                json=data_with_key,
                headers={"Content-Type": "application/json"},
                timeout=30.0
            )
            print(f"POST /v1/chat/completions (api_key param): {response.status_code}")
            if response.status_code == 200:
                print(f"Success! Response: {response.text[:200]}")
            else:
                print(f"Response: {response.text[:500]}")
        except Exception as e:
            print(f"Ошибка POST api_key param: {e}")


async def test_kraken_client():
    """Тест Kraken клиента"""
    print("\n=== Тест Kraken клиента ===")
    
    from kraken_llm.client.standard import StandardLLMClient
    
    config = LLMConfig()
    client = StandardLLMClient(config)
    
    messages = [{"role": "user", "content": "Привет! Как дела?"}]
    
    try:
        response = await client.chat_completion(
            messages=messages,
            max_tokens=50,
            temperature=0.1
        )
        print(f"Kraken client успешно: {response}")
    except Exception as e:
        print(f"Ошибка Kraken client: {e}")


async def main():
    print("Тестирование подключения к LLM серверу")
    print("=" * 50)
    
    await test_direct_connection()
    await test_kraken_client()


if __name__ == "__main__":
    asyncio.run(main())