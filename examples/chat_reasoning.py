#!/usr/bin/env python3
"""
Чат с приоритетом reasoning модели и полностью потоковым выводом
"""
import asyncio
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent / "src"))

from kraken_llm import create_reasoning_client, create_streaming_client, LLMConfig, ReasoningConfig

async def main():
    print("🤖 Kraken Reasoning Chat")
    print("Приоритет: REASONING_MODEL → fallback CHAT_MODEL")
    print("Режим: Полностью потоковый вывод\n")
    
    # Конфигурации
    # Пытаемся собрать конфиг reasoning из окружения; если нет endpoint — пропускаем reasoning
    r_endpoint = os.getenv('REASONING_ENDPOINT') or os.getenv('LLM_REASONING_ENDPOINT')
    r_token = (os.getenv('REASONING_TOKEN') or os.getenv('LLM_REASONING_TOKEN') or
               os.getenv('LLM_REASONING_API_KEY') or os.getenv('REASONING_API_KEY') or
               os.getenv('LLM_API_KEY') or os.getenv('LLM_TOKEN'))
    r_model = (os.getenv('REASONING_MODEL') or os.getenv('LLM_REASONING_MODEL') or
               os.getenv('LLM_MODEL') or 'thinking')
    reasoning_config = None
    if r_endpoint:
        reasoning_config = LLMConfig(
            endpoint=r_endpoint,
            api_key=r_token,
            model=r_model
        )
    
    chat_config = LLMConfig()  # Базовая из .env
    
    print(f"🧠 Reasoning: {r_model} | {r_endpoint or '—'}")
    print(f"💬 Chat: {chat_config.model} | {chat_config.endpoint}\n")
    
    # Создаем клиенты
    reasoning_client = None
    chat_client = None
    
    try:
        # Пробуем подключить reasoning клиент
        if reasoning_config is not None:
            reasoning_client = create_reasoning_client(
                config=reasoning_config,
                reasoning_config=ReasoningConfig(
                    model_type="native_thinking",
                    enable_thinking=True,
                    expose_thinking=True
                )
            )
            await reasoning_client.__aenter__()
            print("✅ Reasoning клиент готов")
        
        # Базовый chat клиент
        chat_client = create_streaming_client(config=chat_config)
        await chat_client.__aenter__()
        print("✅ Chat клиент готов\n")
        
        # Основной цикл
        while True:
            try:
                user_input = input("👤 ").strip()
                if user_input.lower() in ['exit', 'q', 'quit']:
                    break
                if not user_input:
                    continue
                
                print()
                reasoning_used = False
                
                # ПРИОРИТЕТ 1: Reasoning модель для рассуждений
                if reasoning_client:
                    try:
                        print("🧠 Рассуждения (reasoning модель):")
                        print("─" * 50)
                        print("💭 ", end="", flush=True)
                        
                        # Собираем рассуждения для проверки качества
                        reasoning_content = ""
                        chunk_count = 0
                        
                        async for chunk in reasoning_client.chat_completion_stream([
                            {"role": "user", "content": f"Подумай пошагово над вопросом и дай развернутые рассуждения: {user_input}"}
                        ], max_tokens=400):
                            print(chunk, end="", flush=True)
                            reasoning_content += chunk
                            chunk_count += 1
                        
                        # Проверяем качество рассуждений
                        reasoning_quality_ok = (
                            len(reasoning_content.strip()) > 50 and  # Минимальная длина
                            chunk_count > 5 and  # Минимальное количество chunks
                            any(word in reasoning_content.lower() for word in [
                                'потому что', 'поскольку', 'следовательно', 'таким образом',
                                'во-первых', 'во-вторых', 'рассмотрим', 'анализ', 'шаг'
                            ])  # Признаки рассуждений
                        )
                        
                        if reasoning_quality_ok:
                            print("\n" + "─" * 50)
                            reasoning_used = True
                        else:
                            print(f"\n⚠️ Рассуждения некачественные (длина: {len(reasoning_content)}, chunks: {chunk_count})")
                            print("🔄 Переключаюсь на базовую модель...")
                            reasoning_used = False
                        
                    except Exception as e:
                        print(f"\n❌ Reasoning недоступен: {e}")
                        print("🔄 Fallback на chat модель...")
                        reasoning_used = False
                
                # Если reasoning не сработал
                if not reasoning_used:
                    print("🧠 Рассуждения (chat модель):")
                    print("─" * 50)
                    print("💭 [Обрабатываю запрос через базовую модель...]")
                    print("─" * 50)
                
                # ФИНАЛЬНЫЙ ОТВЕТ: Всегда через chat модель
                print("🤖 ", end="", flush=True)
                async for chunk in chat_client.chat_completion_stream([
                    {"role": "user", "content": user_input}
                ], max_tokens=8192):
                    print(chunk, end="", flush=True)
                
                print("\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n❌ {e}\n")
    
    finally:
        if reasoning_client:
            await reasoning_client.__aexit__(None, None, None)
        if chat_client:
            await chat_client.__aexit__(None, None, None)

if __name__ == "__main__":
    asyncio.run(main())