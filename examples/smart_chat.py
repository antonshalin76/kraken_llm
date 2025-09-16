#!/usr/bin/env python3
"""
Умный чат с автоматическим выбором модели и проверкой качества рассуждений
"""
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()
#sys.path.insert(0, str(Path(__file__).parent / "src"))

from kraken_llm import (
    create_reasoning_client,
    create_streaming_client,
    LLMConfig,
    ReasoningConfig,
)


def analyze_reasoning_quality(content: str, chunk_count: int) -> dict:
    """Анализ качества рассуждений"""
    content_clean = content.strip()

    # Критерии качества
    min_length = len(content_clean) > 50
    min_chunks = chunk_count > 5

    # Ключевые слова рассуждений
    reasoning_keywords = [
        "потому что",
        "поскольку",
        "следовательно",
        "таким образом",
        "во-первых",
        "во-вторых",
        "рассмотрим",
        "анализ",
        "шаг",
        "сначала",
        "затем",
        "далее",
        "итак",
        "значит",
        "поэтому",
    ]

    has_reasoning_words = any(
        word in content_clean.lower() for word in reasoning_keywords
    )

    # Проверка структуры (есть ли развернутые предложения)
    sentences = content_clean.split(".")
    has_structure = len(sentences) > 2 and any(len(s.strip()) > 20 for s in sentences)

    quality_score = sum([min_length, min_chunks, has_reasoning_words, has_structure])

    return {
        "is_good": quality_score >= 2,  # Минимум 2 из 4 критериев
        "score": quality_score,
        "length": len(content_clean),
        "chunks": chunk_count,
        "has_keywords": has_reasoning_words,
        "has_structure": has_structure,
    }


async def main():
    print("🧠 Smart Reasoning Chat")
    print("Автоматический выбор лучшей модели для рассуждений")
    print("─" * 60)

    # Конфигурации моделей
    # Конфигурация reasoning из .env (LLM_REASONING_*), с безопасным fallback на LLM_*
    reasoning_config = LLMConfig(
        endpoint=(
            os.getenv("LLM_REASONING_ENDPOINT")
            or os.getenv("REASONING_ENDPOINT")
            or os.getenv("LLM_ENDPOINT")
        ),
        api_key=(
            os.getenv("LLM_REASONING_TOKEN")
            or os.getenv("LLM_REASONING_API_KEY")
            or os.getenv("REASONING_TOKEN")
            or os.getenv("LLM_TOKEN")
        ),
        model=(
            os.getenv("LLM_REASONING_MODEL")
            or os.getenv("REASONING_MODEL")
            or os.getenv("LLM_MODEL")
            or "thinking"
        ),
    )

    chat_config = LLMConfig()

    print(f"🧠 Reasoning: {reasoning_config.model} | {reasoning_config.endpoint}")
    print(f"💬 Chat: {chat_config.model} | {chat_config.endpoint}")
    print()

    # Инициализация клиентов
    reasoning_client = None
    chat_client = None

    try:
        if reasoning_config.endpoint:
            reasoning_client = create_reasoning_client(
                config=reasoning_config,
                reasoning_config=ReasoningConfig(
                    model_type="native_thinking",
                    enable_thinking=True,
                    expose_thinking=True,
                ),
            )
            await reasoning_client.__aenter__()
            print("✅ Reasoning модель подключена")

        chat_client = create_streaming_client(config=chat_config)
        await chat_client.__aenter__()
        print("✅ Chat модель подключена")
        print("\n💡 Введите 'q' для выхода\n")

        while True:
            try:
                user_input = input("👤 ").strip()
                if user_input.lower() in ["q", "quit", "exit", "й"]:
                    break
                if not user_input:
                    continue

                print()
                reasoning_success = False

                # Шаг 1: Пробуем reasoning модель
                if reasoning_client:
                    try:
                        print("🧠 Анализирую через reasoning модель...")
                        print("─" * 50)
                        print("💭 ", end="", flush=True)

                        reasoning_content = ""
                        chunk_count = 0

                        async for chunk in reasoning_client.chat_completion_stream(
                            [
                                {
                                    "role": "user",
                                    "content": f"Подумай пошагово и дай развернутые рассуждения по вопросу: {user_input}",
                                }
                            ],
                            max_tokens=2500,
                        ):
                            print(chunk, end="", flush=True)
                            reasoning_content += chunk
                            chunk_count += 1

                        # Анализируем качество
                        quality = analyze_reasoning_quality(
                            reasoning_content, chunk_count
                        )

                        if quality["is_good"]:
                            print(
                                f"\n✅ Качественные рассуждения (score: {quality['score']}/4)"
                            )
                            print("─" * 50)
                            reasoning_success = True
                        else:
                            print(f"\n⚠️ Низкое качество рассуждений:")
                            print(
                                f"   Длина: {quality['length']} | Chunks: {quality['chunks']}"
                            )
                            print(
                                f"   Ключевые слова: {quality['has_keywords']} | Структура: {quality['has_structure']}"
                            )
                            print("🔄 Переключаюсь на chat модель...")

                    except Exception as e:
                        print(f"\n❌ Ошибка reasoning модели: {e}")
                        print("🔄 Переключаюсь на chat модель...")

                # Шаг 2: Fallback или дополнение через chat модель
                if not reasoning_success:
                    print("\n🧠 Рассуждения через chat модель:")
                    print("─" * 50)
                    print("💭 ", end="", flush=True)

                    async for chunk in chat_client.chat_completion_stream(
                        [
                            {
                                "role": "user",
                                "content": f"Подумай пошагово и дай развернутые рассуждения по вопросу: {user_input}",
                            }
                        ],
                        max_tokens=3000,
                    ):
                        print(chunk, end="", flush=True)

                    print("\n" + "─" * 50)

                # Шаг 3: Финальный ответ
                print("🤖 Финальный ответ: ", end="", flush=True)
                async for chunk in chat_client.chat_completion_stream(
                    [{"role": "user", "content": user_input}], max_tokens=4000
                ):
                    print(chunk, end="", flush=True)

                print("\n")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n❌ Ошибка: {e}\n")

    finally:
        if reasoning_client:
            await reasoning_client.__aexit__(None, None, None)
        if chat_client:
            await chat_client.__aexit__(None, None, None)

    print("👋 До свидания!")


if __name__ == "__main__":
    asyncio.run(main())
