#!/usr/bin/env python3
"""
Простой терминальный чат-бот на базе библиотеки kraken-llm
Использует конфигурацию из .env файла в директории проекта
"""

import asyncio
import sys
import os
from dotenv import load_dotenv

from kraken_llm import create_universal_client, UniversalCapability, LLMConfig

load_dotenv()


class SimpleChatBot:
    """Простой чат-бот с базовым функционалом"""
    
    def __init__(self):
        """Инициализация чат-бота"""
        self.config = LLMConfig(
                endpoint=os.getenv("LLM_ENDPOINT"),
                api_key=os.getenv("LLM_TOKEN"),
                model=os.getenv("LLM_MODEL")            
            )
        self.history = []
        
    async def start(self):
        """Запуск чат-бота"""
        print("🤖 Простой Чат-Бот на базе Kraken LLM")
        print("=" * 50)
        print("Команды:")
        print("  'quit' или 'q' - выход")
        print("  'clear' - очистить историю")
        print("  'history' - показать историю чата")
        print("=" * 50)
        print()
        
        # Создаем универсального клиента с базовыми возможностями
        async with create_universal_client(
            config=self.config,
            capabilities={
                UniversalCapability.CHAT_COMPLETION,
                UniversalCapability.STREAMING
            }
        ) as client:
            
            print("✅ Подключение к модели успешно установлено")
            print("💬 Начните диалог (введите ваше сообщение):")
            print()
            
            while True:
                try:
                    # Получаем ввод от пользователя
                    user_input = input("👤 Вы: ").strip()
                    
                    # Обработка команд
                    if user_input.lower() in ['quit', 'q', 'exit']:
                        print("\n👋 До свидания!")
                        break
                        
                    elif user_input.lower() == 'clear':
                        self.history.clear()
                        print("🗑️  История чата очищена\n")
                        continue
                        
                    elif user_input.lower() == 'history':
                        self.show_history()
                        continue
                        
                    elif not user_input:
                        print("❌ Пустое сообщение. Попробуйте снова.\n")
                        continue
                    
                    # Добавляем сообщение пользователя в историю
                    self.history.append({"role": "user", "content": user_input})
                    
                    # Создаем контекст для запроса (последние 10 сообщений)
                    messages = self.history[-10:]  # Ограничиваем контекст
                    
                    print("🤖 Бот: ", end="", flush=True)
                    
                    # Получаем ответ от модели в потоковом режиме
                    bot_response = ""
                    async for chunk in client.chat_completion_stream(
                        messages=messages,
                        max_tokens=16384,
                        temperature=0.5
                    ):
                        print(chunk, end="", flush=True)
                        bot_response += chunk
                    
                    print("\n")  # Новая строка после ответа
                    
                    # Добавляем ответ бота в историю
                    self.history.append({"role": "assistant", "content": bot_response})
                    
                except KeyboardInterrupt:
                    print("\n\n⚠️  Прерывание. Для выхода введите 'quit'")
                    continue
                    
                except Exception as e:
                    print(f"\n❌ Ошибка: {e}\n")
                    continue
    
    def show_history(self):
        """Показать историю диалога"""
        if not self.history:
            print("📝 История чата пуста\n")
            return
            
        print("\n📝 История чата:")
        print("-" * 40)
        
        for i, message in enumerate(self.history, 1):
            role = "👤 Вы" if message["role"] == "user" else "🤖 Бот"
            content = message["content"]
            
            # Обрезаем длинные сообщения для отображения
            if len(content) > 100:
                content = content[:100] + "..."
                
            print(f"{i:2}. {role}: {content}")
            
        print("-" * 40)
        print(f"Всего сообщений: {len(self.history)}\n")


async def main():
    """Главная функция"""
    bot = SimpleChatBot()
    await bot.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Программа завершена пользователем")
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        sys.exit(1)