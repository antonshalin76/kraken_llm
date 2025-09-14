#!/usr/bin/env python3
"""
Тестирование гибридного completion API с prompt + кастомными полями
"""

import asyncio
import os
import aiohttp
import json
from dotenv import load_dotenv

load_dotenv()

async def test_hybrid_completion():
    """Тестирование гибридного completion API"""
    print("🔍 ТЕСТИРОВАНИЕ ГИБРИДНОГО COMPLETION API")
    print("=" * 60)
    
    base_url = os.getenv("COMPLETION_ENDPOINT").rstrip('/')
    token = os.getenv("COMPLETION_TOKEN")
    model = os.getenv("COMPLETION_MODEL")
    
    url = f"{base_url}/v1/completions"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Различные варианты payload
    test_cases = [
        {
            "name": "Стандартный OpenAI формат",
            "payload": {
                "model": model,
                "prompt": "def hello_world():\n    print(",
                "max_tokens": 20,
                "temperature": 0.7
            }
        },
        {
            "name": "Гибридный: prompt + language + segments",
            "payload": {
                "model": model,
                "prompt": "def power_two_numbers(a: int, b: int): result = ",
                "language": "python",
                "segments": {
                    "prefix": "def power_two_numbers(a: int, b: int): result = "
                },
                "max_tokens": 20,
                "temperature": 0.7
            }
        },
        {
            "name": "Только prompt из segments",
            "payload": {
                "model": model,
                "prompt": "#456456sdsds544\ndef power_two_numbers(a: int, b: int): result = ",
                "max_tokens": 20,
                "temperature": 0.7
            }
        },
        {
            "name": "Prompt + кастомные поля",
            "payload": {
                "model": model,
                "prompt": "def calculate(x, y):\n    return ",
                "language": "python",
                "max_tokens": 20,
                "temperature": 0.2
            }
        },
        {
            "name": "Минимальный payload",
            "payload": {
                "prompt": "Hello, world!",
                "max_tokens": 10
            }
        }
    ]
    
    successful_cases = []
    
    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        print(f"Payload: {json.dumps(test_case['payload'], indent=2)}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=test_case['payload'], headers=headers, timeout=15) as response:
                    status = response.status
                    content_type = response.headers.get('content-type', '')
                    
                    print(f"Статус: {status}, Content-Type: {content_type}")
                    
                    if status == 200:
                        try:
                            result = await response.json()
                            print(f"✅ УСПЕХ!")
                            
                            # Извлекаем текст ответа
                            if 'choices' in result and result['choices']:
                                choice = result['choices'][0]
                                if 'text' in choice:
                                    completion_text = choice['text']
                                    print(f"Автодополнение: '{completion_text}'")
                                elif 'message' in choice and 'content' in choice['message']:
                                    completion_text = choice['message']['content']
                                    print(f"Сообщение: '{completion_text}'")
                                else:
                                    print(f"Выбор: {choice}")
                            else:
                                print(f"Полный ответ: {json.dumps(result, indent=2, ensure_ascii=False)}")
                            
                            successful_cases.append(test_case['name'])
                            
                        except Exception as parse_error:
                            text = await response.text()
                            print(f"✅ УСПЕХ (не JSON): {text[:200]}...")
                            successful_cases.append(test_case['name'])
                    else:
                        text = await response.text()
                        print(f"❌ Ошибка {status}: {text}")
                        
        except Exception as e:
            print(f"❌ Исключение: {str(e)}")
    
    return successful_cases

async def test_streaming_completion():
    """Тестирование streaming completion"""
    print("\n🌊 ТЕСТИРОВАНИЕ STREAMING COMPLETION")
    print("=" * 50)
    
    base_url = os.getenv("COMPLETION_ENDPOINT").rstrip('/')
    token = os.getenv("COMPLETION_TOKEN")
    model = os.getenv("COMPLETION_MODEL")
    
    url = f"{base_url}/v1/completions"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "prompt": "def fibonacci(n):\n    if n <= 1:\n        return n\n    else:\n        return ",
        "max_tokens": 50,
        "temperature": 0.3,
        "stream": True
    }
    
    print(f"Streaming payload: {json.dumps(payload, indent=2)}")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers, timeout=30) as response:
                status = response.status
                
                print(f"Статус: {status}")
                
                if status == 200:
                    print("✅ Streaming ответ:")
                    
                    full_text = ""
                    async for line in response.content:
                        line_str = line.decode('utf-8').strip()
                        if line_str:
                            print(f"   Chunk: {line_str}")
                            
                            # Попытка парсинга SSE
                            if line_str.startswith('data: '):
                                data_str = line_str[6:]
                                if data_str != '[DONE]':
                                    try:
                                        data = json.loads(data_str)
                                        if 'choices' in data and data['choices']:
                                            choice = data['choices'][0]
                                            if 'text' in choice:
                                                full_text += choice['text']
                                    except:
                                        pass
                    
                    if full_text:
                        print(f"✅ Полный текст: '{full_text}'")
                        return True
                else:
                    text = await response.text()
                    print(f"❌ Ошибка {status}: {text}")
                    
    except Exception as e:
        print(f"❌ Исключение: {str(e)}")
    
    return False

async def main():
    """Главная функция"""
    try:
        # Тестируем различные форматы
        successful_cases = await test_hybrid_completion()
        
        # Тестируем streaming
        streaming_works = await test_streaming_completion()
        
        print("\n" + "=" * 60)
        print("📊 ИТОГИ ТЕСТИРОВАНИЯ")
        print("=" * 60)
        
        if successful_cases:
            print(f"✅ Успешных форматов: {len(successful_cases)}")
            for case in successful_cases:
                print(f"   • {case}")
        else:
            print("❌ Ни один формат не работает")
        
        if streaming_works:
            print("✅ Streaming поддерживается")
        else:
            print("❌ Streaming не работает")
        
        if successful_cases:
            print("\n💡 РЕКОМЕНДАЦИИ:")
            print("1. Используйте успешный формат для создания CompletionLLMClient")
            print("2. Обязательно включайте поле 'prompt'")
            print("3. Рассмотрите добавление поддержки кастомных полей")
        else:
            print("\n🔧 СЛЕДУЮЩИЕ ШАГИ:")
            print("1. Проверьте правильность токена авторизации")
            print("2. Убедитесь, что сервер поддерживает completion API")
            print("3. Возможно, нужно использовать другой endpoint")
            
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())