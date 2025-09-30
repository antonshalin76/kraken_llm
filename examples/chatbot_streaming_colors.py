#!/usr/bin/env python3
"""
Стриминговый чат-бот с цветовым градиентом для отображения уверенности токенов.
Поддерживает два режима работы:
1. REALTIME - окраска токенов в реальном времени по мере их получения
2. AGGREGATED - сбор всех токенов и окраска с финальной статистикой

Запуск:
    python3 examples/chatbot_streaming_colors.py [--mode REALTIME|AGGREGATED]

Требования окружения:
    LLM_ENDPOINT, LLM_API_KEY (или LLM_TOKEN), LLM_MODEL
"""

import asyncio
import argparse
import os
import sys
import time
from enum import Enum
from typing import List, Dict, Any, Optional, AsyncGenerator
from pathlib import Path

from dotenv import load_dotenv
from kraken_llm import LLMConfig, create_streaming_client, create_universal_client
from kraken_llm.utils.color import (
    colorize_text_ansi,
    colorize_tokens_ansi,
    get_confidence_legend_ansi,
    get_confidence_description,
)
from kraken_llm.streaming import token_stream_with_shadow_repair
from kraken_llm.confidence.filter import ConfidenceFilterConfig, ensure_confident_chat


class StreamingMode(Enum):
    """Режимы стримингового отображения."""
    REALTIME = "realtime"        # Отображение токенов в реальном времени
    AGGREGATED = "aggregated"    # Сбор всех токенов и итоговая статистика


class StreamingConfidenceChatBot:
    """Стриминговый чат-бот с цветовым градиентом уверенности."""
    
    def __init__(self, config: LLMConfig, streaming_mode: StreamingMode = StreamingMode.REALTIME, *, min_confidence: float = 0.8, per_token_threshold: float = 0.4, max_low_conf_fraction: float = 0.34, no_rollback_marker: bool = False):
        self.config = config
        self.streaming_mode = streaming_mode
        # Пороговые значения уверенности
        self.min_confidence = float(min_confidence)
        self.per_token_threshold = float(per_token_threshold)
        self.max_low_conf_fraction = float(max_low_conf_fraction)
        self.no_rollback_marker = bool(no_rollback_marker)
        self.conversation_history: List[Dict[str, str]] = []
        
    async def _maybe_regenerate(self, base_messages: List[Dict[str, str]], min_confidence: float, per_token_threshold: float, max_low_conf_fraction: float) -> None:
        cfg = ConfidenceFilterConfig(
            min_confidence=min_confidence,
            max_attempts=3,
            prefer_streaming=True,
            per_token_threshold=per_token_threshold,
            max_low_conf_fraction=max_low_conf_fraction,
        )
        async with create_universal_client(self.config) as client:
            regen = await ensure_confident_chat(
                client,
                messages=base_messages,
                cfg=cfg,
                max_tokens=2048,
            )
        print("\n\n🔁 Результат перегенерации (фильтр уверенности):")
        if isinstance(regen, dict):
            if regen.get("token_confidences"):
                print(colorize_tokens_ansi(regen["token_confidences"]))
            else:
                print(colorize_text_ansi(regen.get("text", ""), float(regen.get("confidence", 0.5) or 0.5)))
            print(f"\n📈 Уверенность: {regen.get('confidence', 0.0):.3f} ({regen.get('confidence_label', '')}); попыток: {regen.get('attempts_made', 0)}; успех: {regen.get('success', False)}")
        
    async def stream_response_realtime(self, user_input: str, min_confidence: float, per_token_threshold: float, max_low_conf_fraction: float) -> None:
        """
        Стримит ответ в реальном времени с окраской токенов.
        
        Args:
            user_input: Ввод пользователя
        """
        # Добавляем сообщение пользователя в историю
        self.conversation_history.append({"role": "user", "content": user_input})
        
        print("\n🤖 Ответ модели (в реальном времени):")
        
        collected_tokens = []
        total_text = ""
        start_time = time.time()
        
        # Выбор режима ремонта
        repair_mode = (self.config.repair_mode or "off").lower() if hasattr(self.config, "repair_mode") else "off"
        # Режим с теневым ремонтом или серверный режим: стримим токены в реальном времени
        if repair_mode in ("shadow", "server"):
            client = create_streaming_client(self.config)
            try:
                enable_cutover = (repair_mode == "shadow")
                suppress_rollback = self.no_rollback_marker and enable_cutover
                # Буферы отображения и стек видимых токенов (используем в обоих режимах)
                display_text_buffer = ""
                visible_tokens_stack: List[Dict[str, Any]] = []  # для итоговой окраски токенов и вычисления длины

                def _erase_last_chars(n: int) -> None:
                    # Мягкое стирание n последних видимых символов
                    if n <= 0:
                        return
                    try:
                        sys.stdout.write("\b" * n + " " * n + "\b" * n)
                        sys.stdout.flush()
                    except Exception:
                        # В безпечном режиме просто ничего не делаем
                        pass

                async for ev in token_stream_with_shadow_repair(
                    client,
                    self.conversation_history.copy(),
                    per_token_threshold=per_token_threshold,
                    max_tokens=512,
                    enable_cutover=enable_cutover,
                ):
                    if ev.get("type") == "token":
                        token = ev.get("token", "")
                        conf = float(ev.get("confidence", 0.5) or 0.5)
                        clean_token = token.replace('Ġ', ' ').replace('▁', ' ')
                        if suppress_rollback:
                            # Копим, не показываем до завершения
                            display_text_buffer += clean_token
                        else:
                            # Печатаем с окраской
                            print(colorize_text_ansi(clean_token, conf), end="", flush=True)
                        # Ведём стек для обоих режимов
                        visible_tokens_stack.append({"token": token, "confidence": conf, "visible_len": len(clean_token)})
                        collected_tokens.append({"token": token, "confidence": conf})
                        total_text += clean_token
                    elif ev.get("type") == "rollback":
                        count = int(ev.get("count", 0) or 0)
                        if suppress_rollback:
                            # Удаляем последний токен из локального буфера
                            if count > 0 and display_text_buffer:
                                display_text_buffer = display_text_buffer[:-count]
                            if visible_tokens_stack:
                                visible_tokens_stack.pop()
                        else:
                            # Мягкое стирание с экрана без маркера
                            if count > 0:
                                _erase_last_chars(count)
                                total_text = total_text[:-count] if total_text else total_text
                            if visible_tokens_stack:
                                visible_tokens_stack.pop()
                    elif ev.get("type") == "done":
                        # Если скрывали откаты — выведем финальный чистый результат разом
                        if suppress_rollback:
                            if visible_tokens_stack:
                                final_colored = colorize_tokens_ansi(visible_tokens_stack)
                                print(final_colored, end="", flush=True)
                                total_text = display_text_buffer  # перезапишем финальным текстом
                        break
            except Exception as e:
                print(f"\n❌ Ошибка стриминга: {e}")
                return
            finally:
                try:
                    await client.close()
                except Exception:
                    pass

            # Добавляем собранный текст в историю
            self.conversation_history.append({"role": "assistant", "content": total_text})
        else:
            async with create_streaming_client(self.config) as client:
                try:
                    # Пробуем получить стрим с confidence информацией
                    response = await client.chat_completion(
                        messages=self.conversation_history.copy(),
                        include_confidence=True,
                        max_tokens=2048,
                        logprobs=True,
                        top_logprobs=5
                    )
                    
                    # Если response содержит token_confидences, используем их
                    if isinstance(response, dict) and "token_confidences" in response:
                        token_confidences = response["token_confidences"]
                        text = response.get("text", "")
                        
                        print("🎨 Токены с индивидуальными метриками:")
                        
                        # Отображаем токены по мере их "поступления" с небольшой задержкой для эффекта
                        for i, token_info in enumerate(token_confidences):
                            token = token_info.get("token", "")
                            confidence = token_info.get("confidence", 0.5)
                            
                            # Очищаем токен от специальных символов
                            clean_token = token.replace('Ġ', ' ').replace('▁', ' ')
                            
                            # Окрашиваем и выводим токен
                            colored_token = colorize_text_ansi(clean_token, confidence)
                            print(colored_token, end="", flush=True)
                            
                            collected_tokens.append(token_info)
                            total_text += clean_token
                            
                            # Небольшая задержка для имитации реального стрима
                            await asyncio.sleep(0.05)
                        
                        # Добавляем ответ в историю
                        self.conversation_history.append({"role": "assistant", "content": text})

                        # Проверяем пороги и при необходимости выполняем перегенерацию
                        overall_ok = float(response.get("confidence", 0.0) or 0.0) >= min_confidence
                        confidences = [float(t.get("confidence", 0.0) or 0.0) for t in token_confidences]
                        low_frac = (sum(1 for c in confidences if c < per_token_threshold) / len(confidences)) if confidences else 0.0
                        if not overall_ok or low_frac > max_low_conf_fraction:
                            base_messages = self.conversation_history[:-1]
                            await self._maybe_regenerate(base_messages, min_confidence, per_token_threshold, max_low_conf_fraction)
                        
                    else:
                        # Fallback на обычный стрим без confidence
                        print("⚡ Обычный стрим (без токен-уровневых метрик):")
                        
                        async for chunk in client.chat_completion_stream(
                            messages=self.conversation_history.copy(),
                            max_tokens=2048
                        ):
                            # Окрашиваем chunk средней уверенностью
                            colored_chunk = colorize_text_ansi(chunk, 0.5)
                            print(colored_chunk, end="", flush=True)
                            total_text += chunk
                            await asyncio.sleep(0.02)
                        
                        # Добавляем собранный текст в историю
                        self.conversation_history.append({"role": "assistant", "content": total_text})
                
                except Exception as e:
                    print(f"\n❌ Ошибка стриминга: {e}")
                    return
        
        end_time = time.time()
        
        # Статистика
        print(f"\n\n⏱️  Время генерации: {end_time - start_time:.2f} сек")
        if collected_tokens:
            avg_confidence = sum(t.get("confidence", 0) for t in collected_tokens) / len(collected_tokens)
            print(f"📊 Средняя уверенность: {avg_confidence:.3f} ({get_confidence_description(avg_confidence)})")
            print(f"📝 Всего токенов: {len(collected_tokens)}")
    
    async def stream_response_aggregated(self, user_input: str, min_confidence: float, per_token_threshold: float, max_low_conf_fraction: float) -> None:
        """
        Собирает весь стрим и показывает результат с детальной статистикой.
        
        Args:
            user_input: Ввод пользователя
        """
        # Добавляем сообщение пользователя в историю
        self.conversation_history.append({"role": "user", "content": user_input})
        
        print("\n⏳ Генерирую ответ (режим агрегации)...")
        
        start_time = time.time()
        
        async with create_streaming_client(self.config) as client:
            try:
                # Используем метод с include_confidence для получения агрегированной информации
                response = await client.chat_completion(
                    messages=self.conversation_history.copy(),
                    include_confidence=True,
                    max_tokens=2048,
                    logprobs=True,
                    top_logprobs=5
                )
                
                if isinstance(response, dict):
                    text = response.get("text", "")
                    confidence = response.get("confidence", 0.5)
                    confidence_label = response.get("confidence_label", "средняя")
                    token_confidences = response.get("token_confidences", [])
                    
                    # Добавляем ответ в историю
                    self.conversation_history.append({"role": "assistant", "content": text})
                    
                    end_time = time.time()
                    
                    print("\n🤖 Ответ модели (агрегированный режим):")
                    print(f"📈 Общая уверенность: {confidence:.3f} ({confidence_label})")
                    
                    if token_confidences:
                        print("\n🎨 Текст с токен-уровневым окрашиванием:")
                        colored_text = colorize_tokens_ansi(token_confidences)
                        print(colored_text)
                        
                        # Детальная статистика
                        self._show_detailed_statistics(token_confidences, end_time - start_time)

                        # Проверяем пороги и при необходимости выполняем перегенерацию
                        overall_ok = float(confidence or 0.0) >= min_confidence
                        confidences = [float(t.get("confidence", 0.0) or 0.0) for t in token_confidences]
                        low_frac = (sum(1 for c in confidences if c < per_token_threshold) / len(confidences)) if confidences else 0.0
                        if not overall_ok or low_frac > max_low_conf_fraction:
                            base_messages = self.conversation_history[:-1]
                            await self._maybe_regenerate(base_messages, min_confidence, per_token_threshold, max_low_conf_fraction)
                        
                    else:
                        # Fallback - окраска всего текста по общей уверенности
                        colored_text = colorize_text_ansi(text, confidence)
                        print(f"\n{colored_text}")
                        print("⚠️  Токен-уровневые метрики недоступны")
                        print(f"⏱️  Время генерации: {end_time - start_time:.2f} сек")
                
                else:
                    # Простой текстовый ответ
                    text = str(response)
                    self.conversation_history.append({"role": "assistant", "content": text})
                    
                    end_time = time.time()
                    
                    # Окрашиваем средней уверенностью
                    colored_text = colorize_text_ansi(text, 0.5)
                    print(f"\n🤖 Ответ модели:\n{colored_text}")
                    print(f"⏱️  Время генерации: {end_time - start_time:.2f} сек")
                
            except Exception as e:
                print(f"\n❌ Ошибка получения ответа: {e}")
    
    def _show_detailed_statistics(self, token_confidences: List[Dict[str, Any]], generation_time: float) -> None:
        """Показывает детальную статистику по токенам."""
        confidences = [t.get("confidence", 0) for t in token_confidences]
        
        # Базовая статистика
        avg_conf = sum(confidences) / len(confidences)
        min_conf = min(confidences)
        max_conf = max(confidences)
        
        # Категоризация токенов
        very_high = sum(1 for c in confidences if c >= 0.9)
        high = sum(1 for c in confidences if 0.7 <= c < 0.9)
        medium = sum(1 for c in confidences if 0.5 <= c < 0.7)
        low = sum(1 for c in confidences if 0.3 <= c < 0.5)
        very_low = sum(1 for c in confidences if c < 0.3)
        
        print(f"\n📊 Детальная статистика:")
        print(f"   • Время генерации: {generation_time:.2f} сек")
        print(f"   • Всего токенов: {len(token_confidences)}")
        print(f"   • Скорость: {len(token_confidences) / generation_time:.1f} токенов/сек")
        print(f"   • Средняя уверенность: {avg_conf:.3f} ({get_confidence_description(avg_conf)})")
        print(f"   • Диапазон уверенности: {min_conf:.3f} - {max_conf:.3f}")
        
        print(f"\n🎯 Распределение по категориям:")
        print(f"   • Очень высокая (≥0.9): {very_high} токенов ({very_high/len(confidences)*100:.1f}%)")
        print(f"   • Высокая (0.7-0.9): {high} токенов ({high/len(confidences)*100:.1f}%)")
        print(f"   • Средняя (0.5-0.7): {medium} токенов ({medium/len(confidences)*100:.1f}%)")
        print(f"   • Низкая (0.3-0.5): {low} токенов ({low/len(confidences)*100:.1f}%)")
        print(f"   • Очень низкая (<0.3): {very_low} токенов ({very_low/len(confidences)*100:.1f}%)")
        
        # Показываем самые неуверенные токены
        low_conf_tokens = [t for t in token_confidences if t.get("confidence", 1) < 0.5]
        if low_conf_tokens:
            print(f"\n⚠️  Токены с низкой уверенностью (топ 5):")
            sorted_low = sorted(low_conf_tokens, key=lambda x: x.get("confidence", 1))
            for token_info in sorted_low[:5]:
                token = token_info.get("token", "").replace('Ġ', ' ').replace('▁', ' ')
                conf = token_info.get("confidence", 0)
                colored_token = colorize_text_ansi(f"'{token.strip()}'", conf)
                print(f"     - {colored_token}: {conf:.3f}")
    
    def print_welcome_message(self) -> None:
        """Выводит приветственное сообщение."""
        print("=" * 70)
        print("🌊 KRAKEN STREAMING CONFIDENCE CHATBOT")
        print("=" * 70)
        print(f"Режим стриминга: {self.streaming_mode.value.upper()}")
        
        if self.streaming_mode == StreamingMode.REALTIME:
            print("• Токены отображаются в реальном времени с индивидуальными цветами")
        elif self.streaming_mode == StreamingMode.AGGREGATED:
            print("• Сбор стрима и отображение с детальной статистикой")
        
        print("\n" + get_confidence_legend_ansi())
        print("\nКоманды:")
        print("• 'exit' или 'quit' - выход")
        print("• 'clear' - очистка истории диалога")
        print("• 'mode' - смена режима стриминга")
        print("=" * 70)
    
    def change_mode(self) -> None:
        """Позволяет сменить режим стриминга."""
        print("\nВыберите режим стриминга:")
        print("1. REALTIME - отображение в реальном времени")
        print("2. AGGREGATED - агрегированная статистика")
        
        while True:
            try:
                choice = input("\nВведите номер (1-2): ").strip()
                if choice == "1":
                    self.streaming_mode = StreamingMode.REALTIME
                    break
                elif choice == "2":
                    self.streaming_mode = StreamingMode.AGGREGATED
                    break
                else:
                    print("Неверный выбор. Введите 1 или 2.")
            except KeyboardInterrupt:
                break
        
        print(f"\n✅ Режим изменен на: {self.streaming_mode.value.upper()}")
    
    def clear_history(self) -> None:
        """Очищает историю диалога."""
        self.conversation_history = []
        print("✅ История диалога очищена.")
    
    async def process_user_input(self, user_input: str, min_confidence: float, per_token_threshold: float, max_low_conf_fraction: float) -> None:
        """Обрабатывает ввод пользователя согласно выбранному режиму."""
        if self.streaming_mode == StreamingMode.REALTIME:
            await self.stream_response_realtime(user_input, min_confidence, per_token_threshold, max_low_conf_fraction)
        elif self.streaming_mode == StreamingMode.AGGREGATED:
            await self.stream_response_aggregated(user_input, min_confidence, per_token_threshold, max_low_conf_fraction)
    
    async def run(self) -> None:
        """Запускает интерактивный стриминговый чат-бот."""
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
                if user_input.lower() in ["exit", "quit", "выход", "q", "й"]:
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
                
                # Обрабатываем ввод пользователя
                try:
                    await self.process_user_input(
                        user_input,
                        min_confidence=self.min_confidence,
                        per_token_threshold=self.per_token_threshold,
                        max_low_conf_fraction=self.max_low_conf_fraction,
                    )
                    
                except Exception as e:
                    print(f"\n❌ Ошибка при обработке ввода: {e}")
                    print("Попробуйте еще раз.")
        
        except Exception as e:
            print(f"\n💥 Критическая ошибка: {e}")


async def main():
    """Главная функция."""
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description="Стриминговый чат-бот с цветовым градиентом уверенности")
    parser.add_argument(
        "--mode",
        choices=["realtime", "aggregated"],
        default="realtime",
        help="Режим стриминга (по умолчанию: realtime)"
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
        "--no-rollback-marker",
        action="store_true",
        help="Скрывать индикатор отката и промежуточные неудачные токены; выводить только финальный чистый ответ"
    )
    parser.add_argument(
        "--endpoint",
        help="Endpoint API (переопределяет переменную окружения)"
    )
    parser.add_argument(
        "--repair-mode",
        choices=["off", "shadow", "server"],
        default=os.getenv("LLM_REPAIR_MODE", "off"),
        help="Режим ремонта токенов: off | shadow (теневой форк) | server (серверный порог min_p)"
    )
    parser.add_argument(
        "--server-min-p",
        type=float,
        default=None,
        help="Порог min_p для серверного режима (если поддерживается бэкендом, например vLLM)"
    )
    parser.add_argument(
        "--max-attempts-per-token",
        type=int,
        default=int(os.getenv("LLM_MAX_ATTEMPTS_PER_TOKEN", "3")),
        help="Максимум попыток перегенерации для КАЖДОГО низкоуверенного токена"
    )
    
    args = parser.parse_args()
    
    # Загружаем конфигурацию
    load_dotenv()
    
    config = LLMConfig(
        endpoint=args.endpoint or os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_API_KEY") or os.getenv("LLM_TOKEN"),
        model=args.model or os.getenv("LLM_MODEL"),
        temperature=0.7,
        stream=True,  # Включаем стриминг по умолчанию
        # Настройки для получения logprobs в стриме
        logprobs=True,
        top_logprobs=5,
        # Настройки для стабильного стрима
        force_openai_streaming=True,
        suppress_stream_warnings=True,
        # Live-ремонт
        repair_mode=args.repair_mode,
        per_token_repair_threshold=args.per_token_threshold,
        server_min_p=args.server_min_p,
        max_attempts_per_token=args.max_attempts_per_token,
    )
    
    # Проверяем конфигурацию
    if not config.api_key:
        print("❌ Ошибка: не задан API ключ. Установите LLM_API_KEY или LLM_TOKEN")
        return
    
    # Создаем и запускаем стриминговый чат-бот
    streaming_mode = StreamingMode(args.mode)
    chatbot = StreamingConfidenceChatBot(
        config,
        streaming_mode,
        min_confidence=args.min_confidence,
        per_token_threshold=args.per_token_threshold,
        max_low_conf_fraction=args.max_low_conf_fraction,
        no_rollback_marker=bool(args.no_rollback_marker),
    )
    
    await chatbot.run()


if __name__ == "__main__":
    asyncio.run(main())