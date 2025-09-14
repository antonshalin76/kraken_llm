#!/usr/bin/env python3
"""
Детальная отладка Tool Calling с логированием chunks.
"""

import asyncio
import json
import os
from dotenv import load_dotenv

from kraken_llm.client.streaming import StreamingLLMClient
from kraken_llm.config.settings import LLMConfig

load_dotenv()

class DebugStreamingClient(StreamingLLMClient):
    """Клиент с детальным логированием для отладки."""
    
    async def _process_stream_chunk(self, chunk, function_call_buffer, tool_calls_buffer, original_messages):
        """Переопределяем для детального логирования."""
        print(f"\n🔍 DEBUG: Получен chunk: {chunk}")
        
        if hasattr(chunk, 'choices') and chunk.choices:
            choice = chunk.choices[0]
            print(f"🔍 DEBUG: Choice: {choice}")
            
            if hasattr(choice, 'delta'):
                delta = choice.delta
                print(f"🔍 DEBUG: Delta: {delta}")
                
                if hasattr(delta, 'content') and delta.content:
                    print(f"🔍 DEBUG: Content: '{delta.content}'")
                
                if hasattr(delta, 'function_call') and delta.function_call:
                    print(f"🔍 DEBUG: Function call: {delta.function_call}")
                
                if hasattr(delta, 'tool_calls') and delta.tool_calls:
                    print(f"🔍 DEBUG: Tool calls: {delta.tool_calls}")
        
        # Вызываем оригинальный метод
        return await super()._process_stream_chunk(chunk, function_call_buffer, tool_calls_buffer, original_messages)


async def debug_with_detailed_logging():
    """Отладка с детальным логированием chunks."""
    print("=== Детальная отладка Tool Calling ===")
    
    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL"),
        temperature=0.1  # Низкая температура для более предсказуемого поведения
    )
    
    async with DebugStreamingClient(config) as client:
        
        def get_weather(city: str) -> str:
            """Получить информацию о погоде в указанном городе."""
            print(f"🎯 FUNCTION CALLED: get_weather(city='{city}')")
            return f"В городе {city} сейчас солнечно, +20°C"
        
        client.register_tool("get_weather", get_weather, "Получить погоду в городе")
        
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
        
        messages = [
            {"role": "user", "content": "Какая погода в Москве? Обязательно используй функцию get_weather для получения информации."}
        ]
        
        print("Отправка запроса с tool calling...")
        print("Ответ:")
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
                print(f"📦 Chunk #{chunk_count}: '{chunk}'", end="", flush=True)
                content_chunks.append(chunk)
                
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n{'-' * 50}")
        print(f"Получено {chunk_count} chunks")
        print(f"Общий контент: {''.join(content_chunks)}")


async def debug_direct_openai_call():
    """Прямой вызов OpenAI API для сравнения."""
    print("\n=== Прямой вызов OpenAI API ===")
    
    import openai
    
    client = openai.AsyncOpenAI(
        base_url=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN")
    )
    
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
    
    messages = [
        {"role": "user", "content": "Какая погода в Москве? Используй функцию get_weather."}
    ]
    
    print("Прямой вызов OpenAI streaming API...")
    
    try:
        stream = await client.chat.completions.create(
            model=os.getenv("LLM_MODEL"),
            messages=messages,
            tools=tools,
            tool_choice="auto",
            stream=True,
            temperature=0.1
        )
        
        chunk_count = 0
        print("Raw chunks:")
        print("-" * 30)
        
        async for chunk in stream:
            chunk_count += 1
            print(f"Raw chunk #{chunk_count}: {chunk}")
            
            if chunk.choices:
                choice = chunk.choices[0]
                if choice.delta.content:
                    print(f"  Content: '{choice.delta.content}'")
                if choice.delta.tool_calls:
                    print(f"  Tool calls: {choice.delta.tool_calls}")
                if hasattr(choice.delta, 'function_call') and choice.delta.function_call:
                    print(f"  Function call: {choice.delta.function_call}")
        
        print(f"\n{'-' * 30}")
        print(f"Получено {chunk_count} raw chunks")
        
    except Exception as e:
        print(f"Ошибка прямого вызова: {e}")
        import traceback
        traceback.print_exc()
    
    await client.close()


async def main():
    """Главная функция отладки."""
    print("🔍 Детальная отладка Tool/Function Calling")
    print("=" * 60)
    
    try:
        await debug_with_detailed_logging()
        await debug_direct_openai_call()
        
        print("\n✅ Детальная отладка завершена")
        
    except Exception as e:
        print(f"\n❌ Ошибка отладки: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())