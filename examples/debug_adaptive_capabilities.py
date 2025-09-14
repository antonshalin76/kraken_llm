#!/usr/bin/env python3
"""
Детальная отладка определения возможностей в AdaptiveLLMClient.
"""

import asyncio
import os
import logging
from dotenv import load_dotenv

from kraken_llm.client.adaptive import AdaptiveLLMClient, AdaptiveConfig
from kraken_llm.config.settings import LLMConfig

load_dotenv()

# Включаем детальное логирование
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def debug_capability_detection():
    """Детальная отладка определения возможностей."""
    print("=== Детальная отладка AdaptiveLLMClient ===")
    
    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL"),
        temperature=0.7
    )
    
    adaptive_config = AdaptiveConfig(
        capability_detection_timeout=10.0,
        enable_performance_tracking=True
    )
    
    async with AdaptiveLLMClient(config, adaptive_config) as client:
        
        print("1. Тестирование базовых возможностей:")
        basic_caps = await client._test_basic_capabilities()
        print(f"Базовые возможности: {[cap.value for cap in basic_caps]}")
        
        print("\n2. Тестирование продвинутых возможностей:")
        advanced_caps = await client._test_advanced_capabilities()
        print(f"Продвинутые возможности: {[cap.value for cap in advanced_caps]}")
        
        print("\n3. Тестирование прямой проверки API:")
        direct_caps = await client._test_direct_api_capabilities()
        print(f"Прямая проверка API: {[cap.value for cap in direct_caps]}")
        
        print("\n4. Полное определение возможностей:")
        all_caps = await client.get_model_capabilities(force_refresh=True)
        print(f"Все возможности: {[cap.value for cap in all_caps]}")


async def test_manual_function_calling():
    """Ручное тестирование function calling."""
    print("\n=== Ручное тестирование function calling ===")
    
    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL"),
        temperature=0.3
    )
    
    adaptive_config = AdaptiveConfig(capability_detection_timeout=5.0)
    
    async with AdaptiveLLMClient(config, adaptive_config) as client:
        
        # Получаем стандартный клиент
        standard_client = client._get_client("standard")
        
        # Тестируем function calling напрямую
        test_functions = [{
            "name": "get_weather",
            "description": "Get weather information",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        }]
        
        messages = [
            {"role": "user", "content": "What's the weather in Moscow? Use get_weather function."}
        ]
        
        print("Тестирование function calling через StandardLLMClient:")
        
        try:
            response = await standard_client.chat_completion(
                messages=messages,
                functions=test_functions,
                function_call="auto",
                max_tokens=100
            )
            print(f"Ответ: {response}")
            print("✓ Function calling работает!")
            
        except Exception as e:
            print(f"✗ Function calling не работает: {e}")
            print(f"Тип ошибки: {type(e).__name__}")


async def test_manual_tool_calling():
    """Ручное тестирование tool calling."""
    print("\n=== Ручное тестирование tool calling ===")
    
    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL"),
        temperature=0.3
    )
    
    adaptive_config = AdaptiveConfig(capability_detection_timeout=5.0)
    
    async with AdaptiveLLMClient(config, adaptive_config) as client:
        
        standard_client = client._get_client("standard")
        
        test_tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"}
                    },
                    "required": ["city"]
                }
            }
        }]
        
        messages = [
            {"role": "user", "content": "What's the weather in Moscow? Use get_weather tool."}
        ]
        
        print("Тестирование tool calling через StandardLLMClient:")
        
        try:
            response = await standard_client.chat_completion(
                messages=messages,
                tools=test_tools,
                tool_choice="auto",
                max_tokens=100
            )
            print(f"Ответ: {response}")
            print("✓ Tool calling работает!")
            
        except Exception as e:
            print(f"✗ Tool calling не работает: {e}")
            print(f"Тип ошибки: {type(e).__name__}")


async def main():
    """Главная функция отладки."""
    print("🔍 Детальная отладка AdaptiveLLMClient")
    print("=" * 60)
    
    try:
        await debug_capability_detection()
        await test_manual_function_calling()
        await test_manual_tool_calling()
        
        print("\n✅ Отладка завершена")
        
    except Exception as e:
        print(f"\n❌ Ошибка отладки: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())