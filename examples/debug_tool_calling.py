#!/usr/bin/env python3
"""
Отладка Tool Calling в потоковом режиме.
"""

import asyncio
import json
import os
from dotenv import load_dotenv

from kraken_llm.client.streaming import StreamingLLMClient
from kraken_llm.config.settings import LLMConfig

load_dotenv()

async def debug_tool_calling():
    """Отладка tool calling с детальным логированием."""
    print("=== Отладка Tool Calling ===")
    
    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL"),
        temperature=0.3
    )
    
    async with StreamingLLMClient(config) as client:
        
        # Простая функция для тестирования
        def get_weather(city: str) -> str:
            """Получить информацию о погоде в указанном городе."""
            return f"В городе {city} сейчас солнечно, +20°C"
        
        # Регистрируем инструмент
        client.register_tool("get_weather", get_weather, "Получить погоду в городе")
        
        print(f"Зарегистрированные инструменты: {client.get_registered_tools()}")
        
        # Простое определение инструмента
        tools = [
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
        ]
        
        # Простое сообщение, которое должно вызвать инструмент
        messages = [
            {"role": "user", "content": "Какая погода в Москве? Используй функцию get_weather."}
        ]
        
        print("Отправка запроса с tool calling...")
        print("Параметры:")
        print(f"  - tools: {json.dumps(tools, indent=2, ensure_ascii=False)}")
        print(f"  - tool_choice: auto")
        print(f"  - messages: {messages}")
        
        print("\nОтвет:")
        print("-" * 50)
        
        chunk_count = 0
        content_chunks = []
        
        try:
            async for chunk in client.chat_completion_stream(
                messages=messages,
                tools=tools,
                tool_choice="auto"
            ):
                chunk_count += 1
                print(f"Chunk #{chunk_count}: '{chunk}'", end="", flush=True)
                content_chunks.append(chunk)
                
        except Exception as e:
            print(f"\nОшибка: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n{'-' * 50}")
        print(f"Получено {chunk_count} chunks")
        print(f"Общий контент: {''.join(content_chunks)}")


async def debug_simple_request():
    """Отладка простого запроса без tool calling."""
    print("\n=== Отладка простого запроса ===")
    
    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL"),
        temperature=0.3
    )
    
    async with StreamingLLMClient(config) as client:
        messages = [
            {"role": "user", "content": "Привет! Как дела?"}
        ]
        
        print("Отправка простого запроса...")
        print("Ответ:")
        print("-" * 30)
        
        chunk_count = 0
        async for chunk in client.chat_completion_stream(messages=messages):
            chunk_count += 1
            print(chunk, end="", flush=True)
        
        print(f"\n{'-' * 30}")
        print(f"Получено {chunk_count} chunks")


async def debug_function_calling():
    """Отладка function calling (устаревший API)."""
    print("\n=== Отладка Function Calling ===")
    
    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL"),
        temperature=0.3
    )
    
    async with StreamingLLMClient(config) as client:
        
        def get_weather(city: str) -> str:
            """Получить информацию о погоде в указанном городе."""
            return f"В городе {city} сейчас солнечно, +20°C"
        
        client.register_function("get_weather", get_weather, "Получить погоду в городе")
        
        functions = [
            {
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
        ]
        
        messages = [
            {"role": "user", "content": "Какая погода в Москве? Используй функцию get_weather."}
        ]
        
        print("Отправка запроса с function calling...")
        print("Ответ:")
        print("-" * 30)
        
        chunk_count = 0
        async for chunk in client.chat_completion_stream(
            messages=messages,
            functions=functions,
            function_call="auto"
        ):
            chunk_count += 1
            print(chunk, end="", flush=True)
        
        print(f"\n{'-' * 30}")
        print(f"Получено {chunk_count} chunks")


async def main():
    """Главная функция отладки."""
    print("🔍 Отладка Tool/Function Calling")
    print("=" * 50)
    
    try:
        # Сначала проверим простой запрос
        await debug_simple_request()
        
        # Затем function calling
        await debug_function_calling()
        
        # И наконец tool calling
        await debug_tool_calling()
        
        print("\n✅ Отладка завершена")
        
    except Exception as e:
        print(f"\n❌ Ошибка отладки: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())