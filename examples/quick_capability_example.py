#!/usr/bin/env python3
"""
Быстрая проверка основных возможностей Kraken LLM Framework.
Использует только проверенные рабочие комбинации клиент-модель.
"""

import asyncio
import os
from dotenv import load_dotenv
from pydantic import BaseModel

from kraken_llm.config.settings import LLMConfig
from kraken_llm.client.standard import StandardLLMClient
from kraken_llm.client.streaming import StreamingLLMClient
from kraken_llm.client.structured import StructuredLLMClient
from kraken_llm.client.embeddings import EmbeddingsLLMClient

load_dotenv()

class ResponseModel(BaseModel):
    """Модель ответа для structured output"""
    message: str
    status: str
    code: int

async def test_chat_model():
    """Тестирование chat модели с проверенными клиентами"""
    print("🔄 Тестирование Chat Model...")
    
    config = LLMConfig(
        endpoint=os.getenv("CHAT_ENDPOINT"),
        api_key=os.getenv("CHAT_TOKEN"),
        model=os.getenv("CHAT_MODEL"),
        temperature=0.7
    )
    
    # 1. StandardLLMClient - базовые запросы
    print("  📝 StandardLLMClient...")
    async with StandardLLMClient(config) as client:
        response = await client.chat_completion(
            [{"role": "user", "content": "Привет! Ответь 'Работает'"}],
            max_tokens=10
        )
        print(f"    ✅ Ответ: {response}")
    
    # 2. StreamingLLMClient - потоковые запросы
    print("  🌊 StreamingLLMClient...")
    async with StreamingLLMClient(config) as client:
        full_response = ""
        async for chunk in client.chat_completion_stream(
            [{"role": "user", "content": "Считай от 1 до 3"}],
            max_tokens=20
        ):
            if hasattr(chunk, 'choices') and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                    content = choice.delta.content
                    if content:
                        full_response += content
        print(f"    ✅ Streaming ответ: {full_response.strip()}")
    
    # 3. StructuredLLMClient - структурированные ответы
    print("  📊 StructuredLLMClient...")
    async with StructuredLLMClient(config) as client:
        response = await client.chat_completion_structured(
            [{"role": "user", "content": "Верни JSON: message='Тест', status='ok', code=200"}],
            response_model=ResponseModel,
            max_tokens=50
        )
        print(f"    ✅ Structured ответ: {response}")

async def test_completion_model():
    """Тестирование completion модели с проверенными клиентами"""
    print("\n🔄 Тестирование Completion Model...")
    
    config = LLMConfig(
        endpoint=os.getenv("COMPLETION_ENDPOINT"),
        api_key=os.getenv("COMPLETION_TOKEN"),
        model=os.getenv("COMPLETION_MODEL"),
        temperature=0.7
    )
    
    # 1. StandardLLMClient - базовые запросы
    print("  📝 StandardLLMClient...")
    async with StandardLLMClient(config) as client:
        response = await client.chat_completion(
            [{"role": "user", "content": "Привет! Ответь 'Работает'"}],
            max_tokens=10
        )
        print(f"    ✅ Ответ: {response}")
    
    # 2. StreamingLLMClient - потоковые запросы (работают лучше)
    print("  🌊 StreamingLLMClient...")
    async with StreamingLLMClient(config) as client:
        full_response = ""
        chunk_count = 0
        async for chunk in client.chat_completion_stream(
            [{"role": "user", "content": "Считай от 1 до 3"}],
            max_tokens=20
        ):
            chunk_count += 1
            if hasattr(chunk, 'choices') and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                    content = choice.delta.content
                    if content:
                        full_response += content
            if chunk_count > 10:  # Ограничиваем для теста
                break
        print(f"    ✅ Streaming ответ ({chunk_count} чанков): {full_response.strip()}")

async def test_embedding_model():
    """Тестирование embedding модели"""
    print("\n🔄 Тестирование Embedding Model...")
    
    config = LLMConfig(
        endpoint=os.getenv("EMBEDDING_ENDPOINT"),
        api_key=os.getenv("EMBEDDING_TOKEN"),
        model=os.getenv("EMBEDDING_MODEL")
    )
    
    # EmbeddingsLLMClient - единственный подходящий клиент
    print("  🔢 EmbeddingsLLMClient...")
    async with EmbeddingsLLMClient(config) as client:
        texts = ["Привет мир", "Hello world", "Тестовый текст"]
        embeddings = await client.get_embeddings(texts)
        
        if hasattr(embeddings, 'data') and embeddings.data:
            first_embedding = embeddings.data[0].embedding
            print(f"    ✅ Получены эмбеддинги для {len(texts)} текстов")
            print(f"    📏 Размерность: {len(first_embedding)}")
            print(f"    🔢 Первые 5 значений: {first_embedding[:5]}")
        else:
            print(f"    ✅ Embeddings получены: {type(embeddings)}")

async def main():
    """Основная функция быстрой проверки"""
    print("🚀 БЫСТРАЯ ПРОВЕРКА ВОЗМОЖНОСТЕЙ KRAKEN LLM")
    print("=" * 50)
    print("Тестируем только проверенные рабочие комбинации")
    print("=" * 50)
    
    try:
        # Тестируем chat модель
        if all([os.getenv("CHAT_ENDPOINT"), os.getenv("CHAT_TOKEN"), os.getenv("CHAT_MODEL")]):
            await test_chat_model()
        else:
            print("⚠️ Chat модель не настроена в .env")
        
        # Тестируем completion модель
        if all([os.getenv("COMPLETION_ENDPOINT"), os.getenv("COMPLETION_TOKEN"), os.getenv("COMPLETION_MODEL")]):
            await test_completion_model()
        else:
            print("⚠️ Completion модель не настроена в .env")
        
        # Тестируем embedding модель
        if all([os.getenv("EMBEDDING_ENDPOINT"), os.getenv("EMBEDDING_TOKEN"), os.getenv("EMBEDDING_MODEL")]):
            await test_embedding_model()
        else:
            print("⚠️ Embedding модель не настроена в .env")
        
        print("\n✅ Быстрая проверка завершена!")
        print("\n💡 Рекомендации:")
        print("  • Используйте StreamingLLMClient для потоковых запросов")
        print("  • Используйте StructuredLLMClient для JSON ответов")
        print("  • Используйте EmbeddingsLLMClient только для эмбеддингов")
        print("  • Избегайте CompletionLLMClient с chat endpoints")
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")

if __name__ == "__main__":
    asyncio.run(main())