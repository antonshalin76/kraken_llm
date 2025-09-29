#!/usr/bin/env python3
"""
Чат-бот с цветовым градиентом для отображения уверенности модели.
Поддерживает три режима работы:
1. RESPONSE - окраска всего ответа на основе общей уверенности 
2. TOKEN - окраска каждого токена на основе его индивидуальной уверенности
3. HYBRID - комбинированный режим с обеими метриками

Запуск:
    python3 examples/chatbot_confidence_colors.py [--mode RESPONSE|TOKEN|HYBRID]

Требования окружения:
    LLM_ENDPOINT, LLM_API_KEY (или LLM_TOKEN), LLM_MODEL
"""

import asyncio
import argparse
import os
import sys
from enum import Enum
from typing import List, Dict, Any, Optional
from pathlib import Path

from dotenv import load_dotenv
from kraken_llm import LLMConfig, create_universal_client
from kraken_llm.utils.color import (
    colorize_text_ansi,
    colorize_tokens_ansi,
    get_confidence_legend_ansi,
    get_confidence_description,
)
from kraken_llm.confidence.filter import ConfidenceFilterConfig, ensure_confident_chat


class DisplayMode(Enum):
    """Режимы отображения цветовой информации."""
    RESPONSE = "response"    # Окраска всего ответа по общей уверенности
    TOKEN = "token"         # Окраска каждого токена по его уверенности  
    HYBRID = "hybrid"       # Комбинированный режим


class ConfidenceChatBot:
    """Чат-бот с отображением цветового градиента уверенности."""
    
    def __init__(self, config: LLMConfig, display_mode: DisplayMode = DisplayMode.RESPONSE,
                 min_confidence: float = 0.8,
                 per_token_threshold: float = 0.4,
                 max_low_conf_fraction: Optional[float] = 0.34):
        self.config = config
        self.display_mode = display_mode
        self.conversation_history: List[Dict[str, str]] = []
        self.min_confidence = float(min_confidence)
        self.per_token_threshold = float(per_token_threshold)
        self.max_low_conf_fraction = max_low_conf_fraction if max_low_conf_fraction is not None else None
        
    async def get_response(self, user_input: str) -> Dict[str, Any]:
        """
        Получает ответ от модели с метриками уверенности.
        
        Args:
            user_input: Ввод пользователя
            
        Returns:
            Словарь с ответом и метриками уверенности
        """
        # Добавляем сообщение пользователя в историю
        self.conversation_history.append({"role": "user", "content": user_input})
        
        async with create_universal_client(self.config) as client:
            # Запрос с включением метрик уверенности
            response = await client.chat_completion(
                messages=self.conversation_history.copy(),
                include_confidence=True,
                max_tokens=512,
                # Принудительно включаем logprobs для получения токен-уровневых метрик
                logprobs=True,
                top_logprobs=5
            )
            
            # Добавляем ответ модели в историю
            response_text = response.get("text", "") if isinstance(response, dict) else str(response)
            self.conversation_history.append({"role": "assistant", "content": response_text})
            
            return response
    
    def display_response_mode(self, response: Dict[str, Any]) -> None:
        """Отображает ответ в режиме RESPONSE (общая уверенность)."""
        text = response.get("text", "")
        confidence = response.get("confidence", 0.5)
        confidence_label = response.get("confidence_label", "средняя")
        
        print(f"\n🤖 Ответ модели (общая уверенность: {confidence:.3f} - {confidence_label}):")
        
        # Окрашиваем весь текст согласно общей уверенности
        colored_text = colorize_text_ansi(text, confidence)
        print(colored_text)
        
        # Дополнительная информация
        if "total_tokens" in response:
            print(f"\n📊 Всего токенов: {response['total_tokens']}")
    
    def display_token_mode(self, response: Dict[str, Any]) -> None:
        """Отображает ответ в режиме TOKEN (индивидуальная уверенность токенов)."""
        text = response.get("text", "")
        token_confidences = response.get("token_confidences", [])
        
        print(f"\n🤖 Ответ модели (токен-уровневые метрики):")
        
        if token_confidences:
            # Окрашиваем каждый токен индивидуально
            colored_text = colorize_tokens_ansi(token_confidences)
            print(colored_text)
            
            # Статистика по токенам
            avg_conf = sum(t.get("confidence", 0) for t in token_confidences) / len(token_confidences)
            min_conf = min(t.get("confidence", 1) for t in token_confidences)
            max_conf = max(t.get("confidence", 0) for t in token_confidences)
            
            print(f"\n📊 Статистика токенов:")
            print(f"   • Всего токенов: {len(token_confidences)}")
            print(f"   • Средняя уверенность: {avg_conf:.3f} ({get_confidence_description(avg_conf)})")
            print(f"   • Мин. уверенность: {min_conf:.3f} ({get_confidence_description(min_conf)})")
            print(f"   • Макс. уверенность: {max_conf:.3f} ({get_confidence_description(max_conf)})")
            
            # Показываем токены с самой низкой уверенностью
            low_conf_tokens = [t for t in token_confidences if t.get("confidence", 1) < 0.5]
            if low_conf_tokens:
                print(f"   • Токены с низкой уверенностью ({len(low_conf_tokens)}):")
                for token_info in sorted(low_conf_tokens, key=lambda x: x.get("confidence", 1))[:5]:
                    token = token_info.get("token", "").replace('Ġ', ' ').replace('▁', ' ')
                    conf = token_info.get("confidence", 0)
                    print(f"     - '{token.strip()}': {conf:.3f}")
        else:
            # Fallback - окрашиваем весь текст по общей уверенности
            confidence = response.get("confidence", 0.5)
            colored_text = colorize_text_ansi(text, confidence)
            print(colored_text)
            print("\n⚠️  Токен-уровневые метрики недоступны, использую общую уверенность")
    
    def display_hybrid_mode(self, response: Dict[str, Any]) -> None:
        """Отображает ответ в комбинированном режиме HYBRID."""
        text = response.get("text", "")
        confidence = response.get("confidence", 0.5)
        confidence_label = response.get("confidence_label", "средняя")
        token_confidences = response.get("token_confidences", [])
        
        print(f"\n🤖 Ответ модели (комбинированный режим):")
        print(f"📈 Общая уверенность: {confidence:.3f} ({confidence_label})")
        
        if token_confidences:
            # Показываем токены с индивидуальным окрашиванием
            print("🎨 Текст с токен-уровневым окрашиванием:")
            colored_text = colorize_tokens_ansi(token_confidences)
            print(colored_text)
            
            # Анализ распределения уверенности
            confidences = [t.get("confidence", 0) for t in token_confidences]
            avg_conf = sum(confidences) / len(confidences)
            
            # Разбиваем на категории
            very_high = sum(1 for c in confidences if c >= 0.9)
            high = sum(1 for c in confidences if 0.7 <= c < 0.9) 
            medium = sum(1 for c in confidences if 0.5 <= c < 0.7)
            low = sum(1 for c in confidences if 0.3 <= c < 0.5)
            very_low = sum(1 for c in confidences if c < 0.3)
            
            print(f"\n📊 Подробная статистика:")
            print(f"   • Всего токенов: {len(token_confidences)}")
            print(f"   • Средняя уверенность токенов: {avg_conf:.3f}")
            print(f"   • Распределение по категориям:")
            print(f"     - Очень высокая (≥0.9): {very_high} токенов")
            print(f"     - Высокая (0.7-0.9): {high} токенов")
            print(f"     - Средняя (0.5-0.7): {medium} токенов")
            print(f"     - Низкая (0.3-0.5): {low} токенов")
            print(f"     - Очень низкая (<0.3): {very_low} токенов")
            
            # Показываем разницу между общей и средней токенной уверенностью
            diff = abs(confidence - avg_conf)
            if diff > 0.1:
                print(f"   ⚠️  Расхождение общей и токенной уверенности: {diff:.3f}")
        else:
            # Fallback к режиму RESPONSE
            colored_text = colorize_text_ansi(text, confidence)
            print(colored_text)
            print("\n⚠️  Токен-уровневые метрики недоступны")
    
    def _token_profile_violates(self, token_confidences: List[Dict[str, Any]], per_token_threshold: float, max_low_conf_fraction: Optional[float]) -> bool:
        if not token_confidences:
            return False
        total = len(token_confidences)
        low = sum(1 for t in token_confidences if float(t.get("confidence", 0) or 0.0) < per_token_threshold)
        if max_low_conf_fraction is None:
            return False
        return (low / total) > max_low_conf_fraction if total > 0 else False

    def display_response(self, response: Dict[str, Any]) -> None:
        """Отображает ответ согласно выбранному режиму."""
        if self.display_mode == DisplayMode.RESPONSE:
            self.display_response_mode(response)
        elif self.display_mode == DisplayMode.TOKEN:
            self.display_token_mode(response)
        elif self.display_mode == DisplayMode.HYBRID:
            self.display_hybrid_mode(response)

    async def maybe_regenerate(self, response: Dict[str, Any], min_confidence: float, per_token_threshold: float, max_low_conf_fraction: Optional[float]) -> Optional[Dict[str, Any]]:
        """Если порог уверенности не достигнут, выполняет перегенерацию с фильтром уверенности."""
        overall_ok = float(response.get("confidence", 0.0) or 0.0) >= min_confidence
        token_ok = True
        if "token_confidences" in response:
            token_ok = not self._token_profile_violates(
                response.get("token_confidences") or [], per_token_threshold, max_low_conf_fraction
            )
        if overall_ok and token_ok:
            return None

        print("\n🔁 Порог уверенности не достигнут — запускаю перегенерацию...")
        cfg = ConfidenceFilterConfig(
            min_confidence=min_confidence,
            max_attempts=3,
            prefer_streaming=True,  # собирать пер‑токенные метрики
            per_token_threshold=per_token_threshold,
            max_low_conf_fraction=max_low_conf_fraction,
        )
        # Используем историю без последнего ответа ассистента
        messages = self.conversation_history[:-1]
        async with create_universal_client(self.config) as client:
            regen = await ensure_confident_chat(
                client,
                messages=messages,
                cfg=cfg,
                max_tokens=512,
            )
        return regen
    
    def print_welcome_message(self) -> None:
        """Выводит приветственное сообщение."""
        print("=" * 70)
        print("🌈 KRAKEN CONFIDENCE CHATBOT")
        print("=" * 70)
        print(f"Режим отображения: {self.display_mode.value.upper()}")
        
        if self.display_mode == DisplayMode.RESPONSE:
            print("• Весь ответ окрашивается по общей уверенности модели")
        elif self.display_mode == DisplayMode.TOKEN:
            print("• Каждый токен окрашивается по своей индивидуальной уверенности")
        elif self.display_mode == DisplayMode.HYBRID:
            print("• Комбинированный режим: общая + токенная уверенность")
        
        print("\n" + get_confidence_legend_ansi())
        print("\nКоманды:")
        print("• 'exit' или 'quit' - выход")
        print("• 'clear' - очистка истории диалога") 
        print("• 'mode' - смена режима отображения")
        print("=" * 70)
    
    def change_mode(self) -> None:
        """Позволяет сменить режим отображения."""
        print("\nВыберите режим отображения:")
        print("1. RESPONSE - общая уверенность")
        print("2. TOKEN - токенная уверенность")
        print("3. HYBRID - комбинированный")
        
        while True:
            try:
                choice = input("\nВведите номер (1-3): ").strip()
                if choice == "1":
                    self.display_mode = DisplayMode.RESPONSE
                    break
                elif choice == "2":
                    self.display_mode = DisplayMode.TOKEN
                    break
                elif choice == "3":
                    self.display_mode = DisplayMode.HYBRID
                    break
                else:
                    print("Неверный выбор. Введите 1, 2 или 3.")
            except KeyboardInterrupt:
                break
        
        print(f"\n✅ Режим изменен на: {self.display_mode.value.upper()}")
    
    def clear_history(self) -> None:
        """Очищает историю диалога."""
        self.conversation_history = []
        print("✅ История диалога очищена.")
    
    async def run(self) -> None:
        """Запускает интерактивный чат-бот."""
        self.print_welcome_message()
        
        try:
            while True:
                # Получаем ввод пользователя
                try:
                    user_input = input("\n👤 Вы: ").strip()
                except KeyboardInterrupt:
                    print("\n\n👋 До свидания!")
                    break
                
                # Обработка команд
                if user_input.lower() in ["exit", "quit", "выход", "й", "q"]:
                    print("👋 До свидания!")
                    break
                elif user_input.lower() in ["clear", "очистить"]:
                    self.clear_history()
                    continue
                elif user_input.lower() in ["mode", "режим"]:
                    self.change_mode()
                    continue
                elif not user_input:
                    continue
                
                # Получаем и отображаем ответ
                try:
                    print("\n⏳ Генерирую ответ...")
                    response = await self.get_response(user_input)
                    self.display_response(response)

                    # Проверяем пороги и демонстрируем перегенерацию при необходимости
                    regen = await self.maybe_regenerate(
                        response,
                        min_confidence=self.min_confidence,
                        per_token_threshold=self.per_token_threshold,
                        max_low_conf_fraction=self.max_low_conf_fraction,
                    )
                    if regen is not None:
                        print("\n✅ Перегенерация выполнена. Итоговый ответ:")
                        self.display_response(regen)
                    
                except Exception as e:
                    print(f"\n❌ Ошибка при получении ответа: {e}")
                    print("Попробуйте еще раз.")
        
        except Exception as e:
            print(f"\n💥 Критическая ошибка: {e}")


async def main():
    """Главная функция."""
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description="Чат-бот с цветовым градиентом уверенности")
    parser.add_argument(
        "--mode", 
        choices=["response", "token", "hybrid"],
        default="response",
        help="Режим отображения цветов (по умолчанию: response)"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.8,
        help="Минимально допустимая уверенность ответа для автоперегенерации"
    )
    parser.add_argument(
        "--per-token-threshold",
        type=float,
        default=0.4,
        help="Порог низкой уверенности токена (для перегенерации по профилю токенов)"
    )
    parser.add_argument(
        "--max-low-conf-fraction",
        type=float,
        default=0.34,
        help="Макс. доля токенов ниже порога, иначе перегенерация"
    )
    parser.add_argument(
        "--model",
        help="Модель для использования (переопределяет переменную окружения)"
    )
    parser.add_argument(
        "--endpoint",
        help="Endpoint API (переопределяет переменную окружения)"
    )
    
    args = parser.parse_args()
    
    # Загружаем конфигурацию
    load_dotenv()
    
    config = LLMConfig(
        endpoint=args.endpoint or os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_API_KEY") or os.getenv("LLM_TOKEN"),
        model=args.model or os.getenv("LLM_MODEL"),
        temperature=0.7,
        # Включаем logprobs для получения токен-уровневых метрик
        logprobs=True,
        top_logprobs=5
    )
    
    # Проверяем конфигурацию
    if not config.api_key:
        print("❌ Ошибка: не задан API ключ. Установите LLM_API_KEY или LLM_TOKEN")
        return
    
    # Создаем и запускаем чат-бот
    display_mode = DisplayMode(args.mode)
    chatbot = ConfidenceChatBot(
        config,
        display_mode,
        min_confidence=float(args.min_confidence),
        per_token_threshold=float(args.per_token_threshold),
        max_low_conf_fraction=float(args.max_low_conf_fraction) if args.max_low_conf_fraction is not None else None,
    )
    
    await chatbot.run()


if __name__ == "__main__":
    asyncio.run(main())