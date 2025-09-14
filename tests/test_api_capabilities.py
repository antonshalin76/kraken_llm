#!/usr/bin/env python3
"""
Тестирование возможностей API endpoint.
"""

import asyncio
import json
import os
import aiohttp
from dotenv import load_dotenv

load_dotenv()

async def test_api_info():
    """Тестирование информации об API."""
    endpoint = os.getenv("LLM_ENDPOINT")
    token = os.getenv("LLM_TOKEN")
    
    print(f"Тестирование API: {endpoint}")
    
    # Тест 1: Проверка доступности
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{endpoint}/") as response:
                print(f"GET /: {response.status}")
                text = await response.text()
                print(f"Response: {text[:200]}...")
        except Exception as e:
            print(f"GET /: Ошибка - {e}")
        
        # Тест 2: Проверка /v1/models
        try:
            headers = {"Authorization": f"Bearer {token}"}
            async with session.get(f"{endpoint}/v1/models", headers=headers) as response:
                print(f"GET /v1/models: {response.status}")
                if response.status == 200:
                    data = await response.json()
                    print(f"Models: {json.dumps(data, indent=2)}")
                else:
                    text = await response.text()
                    print(f"Response: {text[:200]}...")
        except Exception as e:
            print(f"GET /v1/models: Ошибка - {e}")
        
        # Тест 3: Простой chat completion
        try:
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": os.getenv("LLM_MODEL"),
                "messages": [
                    {"role": "user", "content": "Привет! Как дела?"}
                ],
                "stream": False
            }
            
            async with session.post(f"{endpoint}/v1/chat/completions", 
                                  headers=headers, 
                                  json=payload) as response:
                print(f"POST /v1/chat/completions (non-stream): {response.status}")
                if response.status == 200:
                    data = await response.json()
                    print(f"Response: {json.dumps(data, indent=2)}")
                else:
                    text = await response.text()
                    print(f"Error response: {text[:500]}...")
        except Exception as e:
            print(f"POST /v1/chat/completions: Ошибка - {e}")
        
        # Тест 4: Streaming chat completion
        try:
            payload = {
                "model": os.getenv("LLM_MODEL"),
                "messages": [
                    {"role": "user", "content": "Привет! Как дела?"}
                ],
                "stream": True
            }
            
            async with session.post(f"{endpoint}/v1/chat/completions", 
                                  headers=headers, 
                                  json=payload) as response:
                print(f"POST /v1/chat/completions (stream): {response.status}")
                if response.status == 200:
                    print("Streaming chunks:")
                    chunk_count = 0
                    async for line in response.content:
                        chunk_count += 1
                        line_str = line.decode('utf-8').strip()
                        if line_str and not line_str.startswith('data: [DONE]'):
                            print(f"  Chunk #{chunk_count}: {line_str[:100]}...")
                        if chunk_count >= 5:  # Ограничиваем вывод
                            print("  ... (остальные chunks пропущены)")
                            break
                else:
                    text = await response.text()
                    print(f"Error response: {text[:500]}...")
        except Exception as e:
            print(f"POST /v1/chat/completions (stream): Ошибка - {e}")
        
        # Тест 5: Tool calling
        try:
            payload = {
                "model": os.getenv("LLM_MODEL"),
                "messages": [
                    {"role": "user", "content": "Какая погода в Москве? Используй функцию get_weather."}
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Получить информацию о погоде в указанном городе",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "city": {
                                        "type": "string",
                                        "description": "Название города"
                                    }
                                },
                                "required": ["city"]
                            }
                        }
                    }
                ],
                "tool_choice": "auto",
                "stream": False
            }
            
            async with session.post(f"{endpoint}/v1/chat/completions", 
                                  headers=headers, 
                                  json=payload) as response:
                print(f"POST /v1/chat/completions (tools): {response.status}")
                if response.status == 200:
                    data = await response.json()
                    print(f"Tool calling response: {json.dumps(data, indent=2)}")
                else:
                    text = await response.text()
                    print(f"Tool calling error: {text[:500]}...")
        except Exception as e:
            print(f"POST /v1/chat/completions (tools): Ошибка - {e}")


async def main():
    """Главная функция тестирования."""
    print("🔍 Тестирование возможностей API")
    print("=" * 50)
    
    await test_api_info()
    
    print("\n✅ Тестирование завершено")


if __name__ == "__main__":
    asyncio.run(main())