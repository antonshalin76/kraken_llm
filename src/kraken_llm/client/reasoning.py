"""
Клиент для рассуждающих моделей в Kraken LLM фреймворке.

Этот модуль предоставляет ReasoningLLMClient для работы с моделями,
поддерживающими режим рассуждений (reasoning), включая Chain of Thought,
пошаговое рассуждение и анализ логических цепочек.
"""

from typing import Dict, Any, List, Optional, Union, AsyncGenerator
from pydantic import BaseModel, Field
from enum import Enum
import logging

from .base import BaseLLMClient
from ..exceptions.validation import ValidationError
from ..exceptions.api import APIError

logger = logging.getLogger(__name__)

    
def _bool_from_env(val: Optional[str]) -> Optional[bool]:
    """Утилита для интерпретации булевых значений из окружения."""
    if val is None:
        return None
    sval = str(val).strip().lower()
    if sval in {"1", "true", "yes", "y", "on"}:
        return True
    if sval in {"0", "false", "no", "n", "off"}:
        return False
    return None


class ReasoningModelType(str, Enum):
    """Типы рассуждающих моделей"""
    PROMPT_BASED = "prompt_based"  # Обычная модель с CoT промптами
    NATIVE_THINKING = "native_thinking"  # Нативная рассуждающая модель с thinking токенами


class ThinkingBlock(BaseModel):
    """Модель для блока рассуждений (thinking) нативной модели"""
    content: str = Field(..., description="Содержимое блока рассуждений")
    token_count: Optional[int] = Field(None, description="Количество токенов в блоке")


class ReasoningStep(BaseModel):
    """Модель для одного шага рассуждения"""
    step_number: int = Field(..., description="Номер шага")
    thought: str = Field(..., description="Мысль или рассуждение на этом шаге")
    action: Optional[str] = Field(None, description="Действие, выполняемое на этом шаге")
    observation: Optional[str] = Field(None, description="Наблюдение или результат действия")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Уверенность в шаге")
    thinking_block: Optional[ThinkingBlock] = Field(None, description="Блок рассуждений для нативных моделей")


class ReasoningChain(BaseModel):
    """Модель для цепочки рассуждений"""
    steps: List[ReasoningStep] = Field(..., description="Шаги рассуждения")
    final_answer: str = Field(..., description="Финальный ответ")
    total_reasoning_tokens: Optional[int] = Field(None, description="Количество reasoning токенов")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Общая уверенность")
    reasoning_time: Optional[float] = Field(None, description="Время рассуждения в секундах")
    model_type: ReasoningModelType = Field(ReasoningModelType.PROMPT_BASED, description="Тип модели рассуждений")
    thinking_blocks: Optional[List[ThinkingBlock]] = Field(None, description="Блоки рассуждений для нативных моделей")


class ReasoningConfig(BaseModel):
    """Конфигурация для режима рассуждений"""
    # Общие настройки
    model_type: ReasoningModelType = Field(ReasoningModelType.PROMPT_BASED, description="Тип рассуждающей модели")
    max_reasoning_steps: int = Field(10, ge=1, le=50, description="Максимальное количество шагов")
    reasoning_temperature: float = Field(0.1, ge=0.0, le=2.0, description="Temperature для рассуждений")
    require_step_validation: bool = Field(False, description="Требовать валидацию каждого шага")
    extract_confidence: bool = Field(True, description="Извлекать показатели уверенности")
    
    # Настройки для prompt-based моделей
    enable_cot: bool = Field(True, description="Включить Chain of Thought (только для prompt-based)")
    reasoning_prompt_template: Optional[str] = Field(None, description="Шаблон промпта для рассуждений")
    
    # Настройки для нативных рассуждающих моделей
    enable_thinking: bool = Field(True, description="Включить thinking режим (только для native-thinking)")
    thinking_max_tokens: Optional[int] = Field(None, description="Максимум токенов для thinking блоков")
    expose_thinking: bool = Field(True, description="Показывать thinking блоки в ответе")
    thinking_temperature: Optional[float] = Field(None, description="Отдельная температура для thinking")


class ReasoningLLMClient(BaseLLMClient):
    """
    Клиент для работы с рассуждающими моделями.
    
    Поддерживает два типа рассуждающих моделей:
    1. Prompt-based: обычные модели с Chain of Thought промптами
    2. Native-thinking: модели с встроенными thinking токенами
    
    Возможности:
    - Chain of Thought (CoT) для prompt-based моделей
    - Native thinking режим для специальных моделей
    - Step-by-step reasoning
    - Reasoning token counting
    - Confidence estimation
    """
    
    def __init__(self, config, reasoning_config: Optional[ReasoningConfig] = None):
        """
        Инициализация клиента рассуждающих моделей.
        
        Args:
            config: Базовая конфигурация LLM
            reasoning_config: Конфигурация режима рассуждений
        """
        super().__init__(config)
        # Берём либо переданную конфигурацию, либо создаём по умолчанию
        self.reasoning_config = reasoning_config or ReasoningConfig()

        # Применяем переопределения из LLMConfig/.env
        self._apply_reasoning_overrides_from_llm_config()

        # Автодетекция нативного thinking по имени модели/переменным окружения
        try:
            if (
                self.reasoning_config.model_type == ReasoningModelType.PROMPT_BASED
                and self._model_supports_thinking()
                and (self.config.enable_thinking is not False)
                and (self.reasoning_config.enable_thinking is not False)
            ):
                # Включаем native thinking режим автоматически
                self.reasoning_config.model_type = ReasoningModelType.NATIVE_THINKING
                logger.info("Авто-включен режим NATIVE_THINKING на основании модели/окружения")
        except Exception as _:
            # Не мешаем инициализации при ошибке детекции
            pass

        logger.info(
            f"Инициализирован ReasoningLLMClient (тип: {self.reasoning_config.model_type})"
        )
    
    def _apply_reasoning_overrides_from_llm_config(self) -> None:
        """Применяет переопределения ReasoningConfig из LLMConfig/.env при наличии."""
        try:
            # Тип рассуждений
            if self.config.reasoning_type:
                rt = str(self.config.reasoning_type).lower().replace("-", "_")
                if rt in {"native_thinking", "native"}:
                    self.reasoning_config.model_type = ReasoningModelType.NATIVE_THINKING
                elif rt in {"prompt_based", "prompt"}:
                    self.reasoning_config.model_type = ReasoningModelType.PROMPT_BASED

            # Простые флаги/параметры
            if self.config.enable_cot is not None:
                self.reasoning_config.enable_cot = bool(self.config.enable_cot)
            if self.config.max_reasoning_steps is not None:
                self.reasoning_config.max_reasoning_steps = int(self.config.max_reasoning_steps)
            if self.config.expose_thinking is not None:
                self.reasoning_config.expose_thinking = bool(self.config.expose_thinking)
            if self.config.enable_thinking is not None:
                self.reasoning_config.enable_thinking = bool(self.config.enable_thinking)
            if self.config.thinking_max_tokens is not None:
                self.reasoning_config.thinking_max_tokens = int(self.config.thinking_max_tokens)
            if self.config.thinking_temperature is not None:
                self.reasoning_config.thinking_temperature = float(self.config.thinking_temperature)
        except Exception as e:
            logger.debug(f"Не удалось применить переопределения ReasoningConfig: {e}")


    def reasoning_completion(
        self,
        messages: List[Dict[str, str]],
        problem_type: str = "general",
        enable_streaming: bool = False,
        **kwargs
    ) -> Union[ReasoningChain, AsyncGenerator[ReasoningStep, None]]:
        """
        Выполняет completion с режимом рассуждений.
        
        Автоматически выбирает подходящий метод в зависимости от типа модели:
        - Prompt-based: использует CoT промпты
        - Native-thinking: использует thinking параметры
        
        Args:
            messages: Сообщения для обработки
            problem_type: Тип задачи (math, logic, coding, general)
            enable_streaming: Включить потоковый режим рассуждений
            **kwargs: Дополнительные параметры
            
        Returns:
            ReasoningChain или AsyncGenerator для streaming режима
        """
        logger.info(
            f"Начинаем reasoning completion для типа задачи: {problem_type} (модель: {self.reasoning_config.model_type})"
        )

        # Автопереключение на native thinking, если поддерживается
        if (
            self.reasoning_config.model_type == ReasoningModelType.PROMPT_BASED
            and self._model_supports_thinking()
            and (self.config.enable_thinking is not False)
            and (self.reasoning_config.enable_thinking is not False)
        ):
            self.reasoning_config.model_type = ReasoningModelType.NATIVE_THINKING
            logger.info("Переключение на режим NATIVE_THINKING (обнаружена поддержка модели)")
        
        if self.reasoning_config.model_type == ReasoningModelType.NATIVE_THINKING:
            # Используем нативный thinking режим
            if enable_streaming:
                return self._native_thinking_completion_stream(messages, problem_type, **kwargs)
            else:
                return self._native_thinking_completion_sync(messages, problem_type, **kwargs)
        else:
            # Используем prompt-based подход
            reasoning_messages = self._prepare_reasoning_prompt(messages, problem_type)
            
            if enable_streaming:
                return self._reasoning_completion_stream(reasoning_messages, **kwargs)
            else:
                return self._reasoning_completion_sync(reasoning_messages, **kwargs)
    
    def _prepare_reasoning_prompt(
        self, 
        messages: List[Dict[str, str]], 
        problem_type: str
    ) -> List[Dict[str, str]]:
        """
        Подготавливает промпт для режима рассуждений.
        
        Args:
            messages: Исходные сообщения
            problem_type: Тип задачи
            
        Returns:
            Модифицированные сообщения с инструкциями для рассуждений
        """
        # Базовые шаблоны для разных типов задач
        reasoning_templates = {
            "math": """
Решай эту математическую задачу пошагово. Для каждого шага:
1. Объясни, что ты делаешь
2. Покажи вычисления
3. Проверь результат
4. Оцени уверенность в шаге (0-1)

Формат ответа:
Шаг 1: [объяснение]
Вычисление: [формула и расчет]
Результат: [результат шага]
Уверенность: [0.0-1.0]

[повтори для каждого шага]

Финальный ответ: [окончательный результат]
""",
            "logic": """
Реши эту логическую задачу, рассуждая пошагово:
1. Определи известные факты
2. Найди логические связи
3. Сделай выводы на основе фактов
4. Проверь логическую последовательность

Формат ответа:
Шаг 1: [анализ фактов]
Логика: [логическое рассуждение]
Вывод: [промежуточный вывод]
Уверенность: [0.0-1.0]

[повтори для каждого шага]

Финальный ответ: [окончательный вывод]
""",
            "coding": """
Реши эту задачу программирования пошагово:
1. Проанализируй требования
2. Спланируй алгоритм
3. Напиши код по частям
4. Проверь корректность

Формат ответа:
Шаг 1: [анализ требований]
Планирование: [план решения]
Код: [фрагмент кода]
Проверка: [тестирование]
Уверенность: [0.0-1.0]

[повтори для каждого шага]

Финальный ответ: [полное решение]
""",
            "general": """
Реши эту задачу, рассуждая пошагово:
1. Проанализируй проблему
2. Определи подходы к решению
3. Выполни решение по шагам
4. Проверь результат

Формат ответа:
Шаг 1: [анализ проблемы]
Рассуждение: [логика решения]
Действие: [что делаем]
Результат: [что получили]
Уверенность: [0.0-1.0]

[повтори для каждого шага]

Финальный ответ: [окончательное решение]
"""
        }
        
        # Выбираем шаблон или используем кастомный
        template = (self.reasoning_config.reasoning_prompt_template or 
                   reasoning_templates.get(problem_type, reasoning_templates["general"]))
        
        # Модифицируем сообщения
        reasoning_messages = messages.copy()
        
        # Добавляем системный промпт для рассуждений
        system_message = {
            "role": "system",
            "content": f"""Ты эксперт в пошаговом рассуждении. {template}

Важно:
- Каждый шаг должен быть логически обоснован
- Показывай промежуточные результаты
- Оценивай уверенность в каждом шаге
- Максимум {self.reasoning_config.max_reasoning_steps} шагов
- Будь точным и методичным"""
        }
        
        # Вставляем системное сообщение в начало или обновляем существующее
        if reasoning_messages and reasoning_messages[0]["role"] == "system":
            reasoning_messages[0]["content"] += "\n\n" + system_message["content"]
        else:
            reasoning_messages.insert(0, system_message)
        
        return reasoning_messages
    
    async def _reasoning_completion_sync(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> ReasoningChain:
        """
        Синхронное выполнение reasoning completion.
        
        Args:
            messages: Подготовленные сообщения
            **kwargs: Дополнительные параметры
            
        Returns:
            Полная цепочка рассуждений
        """
        import time
        start_time = time.time()
        
        # Параметры для рассуждающей модели
        reasoning_params = {
            "temperature": self.reasoning_config.reasoning_temperature,
            "max_tokens": kwargs.get("max_tokens", 2000),
            **kwargs
        }
        
        try:
            # Выполняем запрос через базовый клиент
            response = await self.chat_completion(messages, **reasoning_params)
            
            # Парсим ответ на шаги рассуждения
            reasoning_chain = self._parse_reasoning_response(response)
            
            # Добавляем метрики
            reasoning_chain.reasoning_time = time.time() - start_time
            
            # Валидируем цепочку рассуждений если требуется
            if self.reasoning_config.require_step_validation:
                await self._validate_reasoning_chain(reasoning_chain)
            
            logger.info(f"Завершено reasoning completion за {reasoning_chain.reasoning_time:.2f}s")
            return reasoning_chain
            
        except Exception as e:
            logger.error(f"Ошибка в reasoning completion: {e}")
            raise APIError(f"Ошибка выполнения рассуждений: {e}")
    
    async def _reasoning_completion_stream(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> AsyncGenerator[ReasoningStep, None]:
        """
        Потоковое выполнение reasoning completion.
        
        Args:
            messages: Подготовленные сообщения
            **kwargs: Дополнительные параметры
            
        Yields:
            Шаги рассуждения по мере их генерации
        """
        # Убираем stream из kwargs чтобы избежать конфликта
        reasoning_params = {
            "temperature": self.reasoning_config.reasoning_temperature,
            **{k: v for k, v in kwargs.items() if k != 'stream'}
        }
        
        current_step = ""
        step_number = 0
        
        try:
            async for chunk in self.chat_completion_stream(messages, **reasoning_params):
                current_step += chunk
                
                # Проверяем, завершился ли шаг
                if self._is_step_complete(current_step):
                    step_number += 1
                    reasoning_step = self._parse_single_step(current_step, step_number)
                    
                    if reasoning_step:
                        yield reasoning_step
                    
                    current_step = ""
                    
                    # Проверяем лимит шагов
                    if step_number >= self.reasoning_config.max_reasoning_steps:
                        break
        
        except Exception as e:
            logger.error(f"Ошибка в streaming reasoning: {e}")
            raise APIError(f"Ошибка потокового рассуждения: {e}")
    
    def _parse_reasoning_response(self, response: str) -> ReasoningChain:
        """
        Парсит ответ модели в структурированную цепочку рассуждений.
        
        Args:
            response: Ответ модели
            
        Returns:
            Структурированная цепочка рассуждений
        """
        steps = []
        lines = response.split('\n')
        current_step = {}
        step_number = 0
        final_answer = ""
        
        # Улучшенный парсинг с поддержкой различных форматов
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Определяем тип строки с более гибким подходом
            if any(line.startswith(prefix) for prefix in ['Шаг ', '**Шаг ', 'Step ']):
                # Сохраняем предыдущий шаг
                if current_step:
                    steps.append(ReasoningStep(**current_step))
                
                # Начинаем новый шаг
                step_number += 1
                if ':' in line:
                    thought = line.split(':', 1)[1].strip()
                else:
                    thought = line
                
                current_step = {
                    "step_number": step_number,
                    "thought": thought
                }
            
            elif any(keyword in line.lower() for keyword in ['рассуждение:', 'логика:', 'планирование:', 'анализ:']):
                if current_step and ':' in line:
                    additional_thought = line.split(':', 1)[1].strip()
                    current_step["thought"] = current_step.get("thought", "") + " " + additional_thought
            
            elif any(keyword in line.lower() for keyword in ['действие:', 'вычисление:', 'код:', 'формула:']):
                if current_step and ':' in line:
                    current_step["action"] = line.split(':', 1)[1].strip()
            
            elif any(keyword in line.lower() for keyword in ['результат:', 'вывод:', 'проверка:', 'ответ:']):
                if current_step and ':' in line:
                    current_step["observation"] = line.split(':', 1)[1].strip()
            
            elif 'уверенность:' in line.lower():
                if current_step and ':' in line:
                    try:
                        confidence_str = line.split(':', 1)[1].strip()
                        # Извлекаем числовое значение
                        import re
                        confidence_match = re.search(r'(\d+\.?\d*)', confidence_str)
                        if confidence_match:
                            confidence_val = float(confidence_match.group(1))
                            # Нормализуем к диапазону 0-1
                            if confidence_val > 1.0:
                                confidence_val = confidence_val / 100.0
                            current_step["confidence"] = min(confidence_val, 1.0)
                    except (ValueError, AttributeError):
                        current_step["confidence"] = None
            
            elif any(keyword in line.lower() for keyword in ['финальный ответ:', 'итоговый ответ:', 'окончательный ответ:']):
                if ':' in line:
                    final_answer = line.split(':', 1)[1].strip()
        
        # Добавляем последний шаг
        if current_step:
            steps.append(ReasoningStep(**current_step))
        
        # Если шагов нет, создаем один шаг из всего ответа
        if not steps:
            steps.append(ReasoningStep(
                step_number=1,
                thought=response[:500] + "..." if len(response) > 500 else response
            ))
        
        # Если финальный ответ не найден, извлекаем из конца ответа
        if not final_answer:
            # Ищем финальный ответ в конце текста
            final_answer_patterns = [
                'финальный ответ:',
                'итоговый ответ:',
                'ответ:',
                'результат:'
            ]
            
            response_lower = response.lower()
            for pattern in final_answer_patterns:
                if pattern in response_lower:
                    final_answer = response.split(pattern, 1)[-1].strip()
                    break
            
            # Если все еще не найден, берем последние 200 символов
            if not final_answer:
                final_answer = response[-200:].strip()
        
        # Вычисляем общую уверенность
        confidences = [step.confidence for step in steps if step.confidence is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else None
        
        return ReasoningChain(
            steps=steps,
            final_answer=final_answer,
            confidence_score=avg_confidence,
            model_type=ReasoningModelType.PROMPT_BASED
        )
    
    def _is_step_complete(self, text: str) -> bool:
        """
        Проверяет, завершен ли текущий шаг рассуждения.
        
        Args:
            text: Текст для проверки
            
        Returns:
            True если шаг завершен
        """
        # Ищем маркеры завершения шага
        completion_markers = [
            "Уверенность:",
            "Шаг ",
            "**Шаг ",
            "Step ",
            "Финальный ответ:",
            "Итоговый ответ:",
            "\n\n"  # Двойной перенос строки как маркер завершения
        ]
        
        text_lower = text.lower()
        
        # Проверяем наличие маркеров
        for marker in completion_markers:
            if marker.lower() in text_lower:
                return True
        
        # Проверяем длину текста (если шаг слишком длинный, считаем завершенным)
        if len(text) > 300:
            return True
        
        # Проверяем наличие точки в конце предложения
        if text.strip().endswith('.') and len(text.strip()) > 50:
            return True
            
        return False
    
    def _parse_single_step(self, text: str, step_number: int) -> Optional[ReasoningStep]:
        """
        Парсит один шаг рассуждения из текста.
        
        Args:
            text: Текст шага
            step_number: Номер шага
            
        Returns:
            Объект ReasoningStep или None
        """
        try:
            # Очищаем текст
            text = text.strip()
            if not text:
                return None
            
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            step_data = {"step_number": step_number, "thought": ""}
            
            # Если есть структурированные строки с двоеточием
            structured_content = False
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    if any(keyword in key for keyword in ['шаг', 'рассуждение', 'логика', 'анализ', 'мысль']):
                        step_data["thought"] = value
                        structured_content = True
                    elif any(keyword in key for keyword in ['действие', 'вычисление', 'код', 'формула']):
                        step_data["action"] = value
                        structured_content = True
                    elif any(keyword in key for keyword in ['результат', 'вывод', 'наблюдение', 'проверка']):
                        step_data["observation"] = value
                        structured_content = True
                    elif 'уверенность' in key:
                        try:
                            import re
                            confidence_match = re.search(r'(\d+\.?\d*)', value)
                            if confidence_match:
                                confidence_val = float(confidence_match.group(1))
                                if confidence_val > 1.0:
                                    confidence_val = confidence_val / 100.0
                                step_data["confidence"] = min(confidence_val, 1.0)
                        except (ValueError, AttributeError):
                            pass
                        structured_content = True
            
            # Если нет структурированного контента, используем весь текст как мысль
            if not structured_content or not step_data["thought"]:
                # Убираем маркеры шагов из начала
                clean_text = text
                for prefix in ['Шаг ', '**Шаг ', 'Step ']:
                    if clean_text.startswith(prefix):
                        if ':' in clean_text:
                            clean_text = clean_text.split(':', 1)[1].strip()
                        break
                
                step_data["thought"] = clean_text[:300] + "..." if len(clean_text) > 300 else clean_text
            
            # Создаем шаг только если есть содержательная мысль
            if step_data["thought"] and len(step_data["thought"].strip()) > 5:
                return ReasoningStep(**step_data)
            
        except Exception as e:
            logger.warning(f"Ошибка парсинга шага: {e}")
        
        return None
    
    async def _validate_reasoning_chain(self, chain: ReasoningChain) -> None:
        """
        Валидирует логическую последовательность цепочки рассуждений.
        
        Args:
            chain: Цепочка рассуждений для валидации
            
        Raises:
            ValidationError: Если цепочка содержит логические ошибки
        """
        if not chain.steps:
            raise ValidationError("Цепочка рассуждений пуста")
        
        # Проверяем последовательность шагов
        for i, step in enumerate(chain.steps):
            if step.step_number != i + 1:
                raise ValidationError(f"Нарушена последовательность шагов: ожидался {i+1}, получен {step.step_number}")
            
            if not step.thought:
                raise ValidationError(f"Шаг {step.step_number} не содержит рассуждения")
        
        # Проверяем наличие финального ответа
        if not chain.final_answer:
            raise ValidationError("Отсутствует финальный ответ")
        
        logger.info("Валидация цепочки рассуждений прошла успешно")
    
    async def analyze_reasoning_quality(self, chain: ReasoningChain) -> Dict[str, Any]:
        """
        Анализирует качество цепочки рассуждений.
        
        Args:
            chain: Цепочка рассуждений для анализа
            
        Returns:
            Словарь с метриками качества
        """
        analysis = {
            "total_steps": len(chain.steps),
            "avg_confidence": chain.confidence_score,
            "has_final_answer": bool(chain.final_answer),
            "reasoning_completeness": 0.0,
            "logical_consistency": 0.0,
            "step_quality_scores": []
        }
        
        # Анализируем каждый шаг
        for step in chain.steps:
            step_quality = {
                "step_number": step.step_number,
                "has_thought": bool(step.thought),
                "has_action": bool(step.action),
                "has_observation": bool(step.observation),
                "has_confidence": step.confidence is not None,
                "thought_length": len(step.thought) if step.thought else 0
            }
            
            # Вычисляем оценку качества шага
            quality_score = sum([
                step_quality["has_thought"] * 0.4,
                step_quality["has_action"] * 0.2,
                step_quality["has_observation"] * 0.2,
                step_quality["has_confidence"] * 0.1,
                min(step_quality["thought_length"] / 50, 1.0) * 0.1
            ])
            
            step_quality["quality_score"] = quality_score
            analysis["step_quality_scores"].append(step_quality)
        
        # Вычисляем общие метрики
        if analysis["step_quality_scores"]:
            analysis["reasoning_completeness"] = sum(
                sq["quality_score"] for sq in analysis["step_quality_scores"]
            ) / len(analysis["step_quality_scores"])
        
        # Простая оценка логической последовательности
        analysis["logical_consistency"] = min(1.0, analysis["reasoning_completeness"] * 1.2)
        
        return analysis
    
    async def chat_completion_with_reasoning(
        self,
        messages: List[Dict[str, str]],
        reasoning_config: Optional[ReasoningConfig] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Выполняет chat completion с пошаговым рассуждением.
        
        Args:
            messages: Сообщения для обработки
            reasoning_config: Конфигурация рассуждений
            **kwargs: Дополнительные параметры
            
        Returns:
            Словарь с результатами рассуждения
        """
        if reasoning_config is None:
            reasoning_config = ReasoningConfig()
        
        # Выполняем рассуждение
        chain = await self.reasoning_completion(
            messages=messages,
            problem_type="general",
            enable_streaming=False,
            **kwargs
        )
        
        # Возвращаем результат в удобном формате
        return {
            "reasoning_steps": [step.thought for step in chain.steps],
            "final_answer": chain.final_answer,
            "confidence_score": chain.confidence_score,
            "total_steps": len(chain.steps),
            "reasoning_tokens": chain.total_reasoning_tokens
        }

    # Методы для нативных рассуждающих моделей
    async def _native_thinking_completion_sync(
        self,
        messages: List[Dict[str, str]],
        problem_type: str = "general",
        **kwargs
    ) -> ReasoningChain:
        """
        Синхронное выполнение с нативной рассуждающей моделью.
        
        Если модель не поддерживает thinking параметры, использует fallback
        к prompt-based подходу с специальным промптом для имитации thinking.
        
        Args:
            messages: Сообщения для обработки
            problem_type: Тип задачи
            **kwargs: Дополнительные параметры
            
        Returns:
            Цепочка рассуждений с thinking блоками
        """
        import time
        start_time = time.time()
        
        # Проверяем поддержку thinking
        if self._model_supports_thinking():
            # Пытаемся использовать нативные thinking параметры
            thinking_params = self._prepare_thinking_params(messages, **kwargs)
            
            try:
                response = await self.openai_client.chat.completions.create(**thinking_params)
                reasoning_chain = self._parse_thinking_response(response)
                
            except Exception as e:
                # Если API не поддерживает thinking параметры, используем fallback
                if "unexpected keyword argument" in str(e) and "thinking" in str(e):
                    logger.info("API не поддерживает thinking параметры, используем fallback")
                    thinking_messages = self._prepare_thinking_style_prompt(messages, problem_type)
                    response_text = await self.chat_completion(thinking_messages, **kwargs)
                    reasoning_chain = self._parse_thinking_style_response(response_text)
                else:
                    logger.error(f"Ошибка в native thinking completion: {e}")
                    raise APIError(f"Ошибка выполнения нативного рассуждения: {e}")
        else:
            # Fallback к prompt-based подходу с thinking имитацией
            logger.info("Модель не поддерживает thinking параметры, используем thinking-style промпт")
            thinking_messages = self._prepare_thinking_style_prompt(messages, problem_type)
            
            try:
                # Используем обычный chat completion
                response_text = await self.chat_completion(thinking_messages, **kwargs)
                reasoning_chain = self._parse_thinking_style_response(response_text)
                
            except Exception as e:
                logger.error(f"Ошибка в thinking-style completion: {e}")
                raise APIError(f"Ошибка выполнения thinking-style рассуждения: {e}")
        
        # Добавляем метрики
        reasoning_chain.reasoning_time = time.time() - start_time
        reasoning_chain.model_type = ReasoningModelType.NATIVE_THINKING
        
        # Валидируем если требуется
        if self.reasoning_config.require_step_validation:
            await self._validate_reasoning_chain(reasoning_chain)
        
        logger.info(f"Завершено native thinking completion за {reasoning_chain.reasoning_time:.2f}s")
        return reasoning_chain
    
    async def _native_thinking_completion_stream(
        self,
        messages: List[Dict[str, str]],
        problem_type: str = "general",
        **kwargs
    ) -> AsyncGenerator[ReasoningStep, None]:
        """
        Потоковое выполнение с нативной рассуждающей моделью.
        
        Если модель не поддерживает thinking параметры, использует fallback
        к обычному потоковому режиму с thinking-style промптом.
        
        Args:
            messages: Сообщения для обработки
            problem_type: Тип задачи
            **kwargs: Дополнительные параметры
            
        Yields:
            Шаги рассуждения по мере их генерации
        """
        if self._model_supports_thinking():
            # Используем нативный thinking режим
            async for step in self._native_thinking_stream_native(messages, problem_type, **kwargs):
                yield step
        else:
            # Fallback к thinking-style потоку
            async for step in self._native_thinking_stream_fallback(messages, problem_type, **kwargs):
                yield step
    
    async def _native_thinking_stream_native(
        self,
        messages: List[Dict[str, str]],
        problem_type: str = "general",
        **kwargs
    ) -> AsyncGenerator[ReasoningStep, None]:
        """Нативный thinking поток для поддерживающих моделей"""
        thinking_params = self._prepare_thinking_params(messages, stream=True, **kwargs)
        
        current_thinking = ""
        current_content = ""
        step_number = 0
        
        try:
            stream = await self.openai_client.chat.completions.create(**thinking_params)
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta
                    
                    # Обрабатываем thinking блоки
                    if hasattr(delta, 'thinking') and delta.thinking:
                        current_thinking += delta.thinking
                        
                        if self._is_thinking_block_complete(current_thinking):
                            step_number += 1
                            thinking_step = self._create_thinking_step(
                                current_thinking, 
                                step_number
                            )
                            
                            if thinking_step:
                                yield thinking_step
                            
                            current_thinking = ""
                    
                    # Обрабатываем обычный контент
                    if delta.content:
                        current_content += delta.content
            
            # Финальный шаг
            if current_content:
                step_number += 1
                final_step = ReasoningStep(
                    step_number=step_number,
                    thought="Финальный ответ",
                    observation=current_content,
                    confidence=1.0
                )
                yield final_step
        
        except Exception as e:
            logger.error(f"Ошибка в native thinking stream: {e}")
            raise APIError(f"Ошибка потокового нативного рассуждения: {e}")
    
    async def _native_thinking_stream_fallback(
        self,
        messages: List[Dict[str, str]],
        problem_type: str = "general",
        **kwargs
    ) -> AsyncGenerator[ReasoningStep, None]:
        """Fallback thinking поток для обычных моделей"""
        thinking_messages = self._prepare_thinking_style_prompt(messages, problem_type)
        
        current_text = ""
        step_number = 0
        in_thinking = False
        thinking_content = ""
        
        try:
            # Убираем stream из kwargs чтобы избежать конфликта
            stream_params = {k: v for k, v in kwargs.items() if k != 'stream'}
            
            async for chunk in self.chat_completion_stream(thinking_messages, **stream_params):
                current_text += chunk
                
                # Проверяем начало thinking блока
                if '<thinking>' in current_text and not in_thinking:
                    in_thinking = True
                    thinking_start = current_text.find('<thinking>') + len('<thinking>')
                    thinking_content = current_text[thinking_start:]
                
                # Если мы в thinking блоке
                if in_thinking:
                    if '</thinking>' in current_text:
                        # Завершение thinking блока
                        thinking_end = current_text.find('</thinking>')
                        thinking_content = current_text[current_text.find('<thinking>') + len('<thinking>'):thinking_end]
                        
                        # Создаем шаг из thinking блока
                        step_number += 1
                        thinking_step = ReasoningStep(
                            step_number=step_number,
                            thought=thinking_content.strip(),
                            confidence=0.8,
                            thinking_block=ThinkingBlock(
                                content=thinking_content.strip(),
                                token_count=self._count_tokens(thinking_content)
                            )
                        )
                        yield thinking_step
                        
                        in_thinking = False
                    else:
                        # Продолжаем накапливать thinking контент
                        continue
                
                # Обрабатываем обычный контент после thinking
                if not in_thinking and '</thinking>' in current_text:
                    final_content = current_text.split('</thinking>')[-1].strip()
                    if final_content and len(final_content) > 20:  # Достаточно контента для шага
                        step_number += 1
                        final_step = ReasoningStep(
                            step_number=step_number,
                            thought="Финальный ответ",
                            observation=final_content,
                            confidence=0.9
                        )
                        yield final_step
                        break
        
        except Exception as e:
            logger.error(f"Ошибка в fallback thinking stream: {e}")
            raise APIError(f"Ошибка fallback потокового рассуждения: {e}")
    
    def _prepare_thinking_params(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Подготавливает параметры для нативной рассуждающей модели.
        
        Для некоторых провайдеров (например, Qwen Thinking) достаточно обычного
        chat.completions вызова — модель вернёт внутренние размышления в виде токенов.
        Если провайдер требует спец.параметров (enable_thinking и т.п.), их нужно
        передавать через extra_body, поскольку AsyncOpenAI имеет типизированную
        сигнатуру и игнорирует неизвестные именованные аргументы.
        """
        # Параметры, поддерживаемые SDK непосредственно
        supported_keys = {
            "model", "messages", "temperature", "top_p", "n", "stream", "stop",
            "max_tokens", "presence_penalty", "frequency_penalty", "logit_bias",
            "user", "response_format", "tools", "tool_choice", "functions",
            "function_call", "seed"
        }

        # Базовые параметры
        params: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.reasoning_config.reasoning_temperature,
            "stream": stream,
        }

        # Собираем provider-specific thinking флаги в extra_body
        extra_body: Dict[str, Any] = {}
        if self.reasoning_config.enable_thinking or (self.config.enable_thinking is True):
            for key, val in {
                "enable_thinking": True,
                "expose_thinking": self.reasoning_config.expose_thinking,
                "thinking_max_tokens": self.reasoning_config.thinking_max_tokens,
                "thinking_temperature": self.reasoning_config.thinking_temperature,
            }.items():
                if val is not None:
                    extra_body[key] = val

        # Разносим дополнительные kwargs: известные в params, остальные в extra_body
        for k, v in kwargs.items():
            if k in supported_keys:
                params[k] = v
            else:
                extra_body[k] = v

        # Прикрепляем extra_body, если там есть поля
        if extra_body:
            params["extra_body"] = extra_body

        # Удаляем None-значения из верхнего уровня
        params = {k: v for k, v in params.items() if v is not None}
        
        return params
    
    def _model_supports_thinking(self) -> bool:
        """
        Проверяет, поддерживает ли модель нативные thinking параметры.
        
        Returns:
            True если модель поддерживает thinking режим
        """
        import os

        model_name = (self.config.model or "").lower()

        # Явное указание через окружение имеет приоритет
        if (self.config.reasoning_type and str(self.config.reasoning_type).lower() in {"native_thinking", "native-thinking", "native"}) or \
           (os.getenv("LLM_REASONING_TYPE", "").lower() in {"native_thinking", "native-thinking", "native"}):
            return True

        # Эвристика по названию модели
        if ("thinking" in model_name or "reasoning" in model_name or "o1" in model_name or "r1" in model_name) and \
           ("qwen" in model_name or "deepseek" in model_name or "openai" in model_name or "anthropic" in model_name or "gemini" in model_name or True):
            return True

        # Поддержка конкретных часто встречающихся вариантов
        thinking_models = [
            # OpenAI reasoning models
            "o1-preview",
            "o1-mini",
            "o1-pro",

            # Qwen (включая вариации с суффиксами версий)
            "qwen3-4b-thinking",
            "qwen3-7b-thinking",
            "qwen2.5-thinking",
            "qwen2-thinking",
            "qwen-thinking",

            # DeepSeek (R1 series)
            "deepseek-r1",
            "deepseek-reasoner",

            # Generic
            "thinking-",
            "-thinking",
            "reasoner",
        ]
        return any(pat in model_name for pat in thinking_models)
    
    def _get_thinking_tokens(self) -> tuple[str, str]:
        """
        Возвращает наиболее вероятные токены для данной модели.
        
        Примечание: Это используется только для генерации промптов.
        Для парсинга ответов используется универсальный поиск всех токенов.
        
        Returns:
            Кортеж (start_token, end_token)
        """
        model_name = self.config.model.lower()
        
        # Известные модели с конкретными токенами
        if "qwen" in model_name and "thinking" in model_name:
            return ("<think>", "</think>")
        elif "o1" in model_name:
            return ("<thinking>", "</thinking>")
        else:
            # По умолчанию используем стандартные токены
            return ("<thinking>", "</thinking>")
    
    def _get_all_possible_thinking_tokens(self) -> List[tuple[str, str]]:
        """
        Возвращает все возможные варианты thinking токенов для fallback парсинга.
        
        Returns:
            Список кортежей (start_token, end_token)
        """
        return [
            # Наиболее распространенные
            ("<thinking>", "</thinking>"),
            ("<think>", "</think>"),
            ("<reasoning>", "</reasoning>"),
            ("<reason>", "</reason>"),
            
            # Альтернативные варианты
            ("<thought>", "</thought>"),
            ("<analysis>", "</analysis>"),
            ("<reflection>", "</reflection>"),
            ("<internal>", "</internal>"),
            
            # Специфичные для провайдеров
            ("<cot>", "</cot>"),  # Chain of Thought
            ("<scratchpad>", "</scratchpad>"),
            ("<workingmemory>", "</workingmemory>"),
            ("<process>", "</process>"),
            
            # Возможные будущие варианты
            ("<deliberation>", "</deliberation>"),
            ("<contemplation>", "</contemplation>"),
            ("<brainstorm>", "</brainstorm>"),
            ("<mindmap>", "</mindmap>")
        ]
    
    def _parse_thinking_response(self, response) -> ReasoningChain:
        """
        Парсит ответ нативной рассуждающей модели.
        
        Поддерживает как API с отдельным thinking полем, так и модели
        с thinking токенами в основном контенте (как Qwen3-Thinking).
        
        Args:
            response: Ответ от OpenAI API
            
        Returns:
            Цепочка рассуждений с thinking блоками
        """
        choice = response.choices[0]
        message = choice.message
        
        steps = []
        thinking_blocks = []
        final_answer = ""
        
        # Получаем токены thinking для данной модели
        start_token, end_token = self._get_thinking_tokens()
        
        # Проверяем наличие отдельного thinking поля (OpenAI style)
        if hasattr(message, 'thinking') and message.thinking:
            thinking_content = message.thinking
            thinking_blocks.append(ThinkingBlock(
                content=thinking_content,
                token_count=self._count_tokens(thinking_content)
            ))
            
            thinking_steps = self._extract_steps_from_thinking(thinking_content)
            steps.extend(thinking_steps)
            
            final_answer = message.content or ""
        
        # Проверяем наличие thinking токенов в основном контенте
        elif message.content:
            full_content = message.content
            thinking_content, final_answer = self._extract_thinking_from_content(full_content)
            
            if thinking_content:
                thinking_blocks.append(ThinkingBlock(
                    content=thinking_content,
                    token_count=self._count_tokens(thinking_content)
                ))
                
                thinking_steps = self._extract_steps_from_thinking(thinking_content)
                steps.extend(thinking_steps)
            else:
                final_answer = full_content
        
        else:
            # Fallback: весь контент как финальный ответ
            final_answer = message.content or ""
        
        # Если нет шагов из thinking, создаем один базовый шаг
        if not steps:
            steps.append(ReasoningStep(
                step_number=1,
                thought="Рассуждение модели",
                observation=final_answer[:200] + "..." if len(final_answer) > 200 else final_answer,
                confidence=0.9
            ))
        
        # Подсчитываем reasoning токены
        total_reasoning_tokens = sum(
            block.token_count for block in thinking_blocks 
            if block.token_count
        ) or None
        
        return ReasoningChain(
            steps=steps,
            final_answer=final_answer,
            total_reasoning_tokens=total_reasoning_tokens,
            confidence_score=self._calculate_average_confidence(steps),
            model_type=ReasoningModelType.NATIVE_THINKING,
            thinking_blocks=thinking_blocks
        )
    
    def _extract_steps_from_thinking(self, thinking_content: str) -> List[ReasoningStep]:
        """
        Извлекает шаги рассуждения из thinking блока.
        
        Args:
            thinking_content: Содержимое thinking блока
            
        Returns:
            Список шагов рассуждения
        """
        steps = []
        
        # Разбиваем thinking на логические части
        parts = thinking_content.split('\n\n')
        
        for i, part in enumerate(parts, 1):
            if part.strip():
                step = ReasoningStep(
                    step_number=i,
                    thought=part.strip(),
                    confidence=0.8,  # Базовая уверенность для thinking блоков
                    thinking_block=ThinkingBlock(
                        content=part.strip(),
                        token_count=self._count_tokens(part)
                    )
                )
                steps.append(step)
        
        return steps
    
    def _is_thinking_block_complete(self, thinking_text: str) -> bool:
        """
        Проверяет завершенность thinking блока.
        
        Args:
            thinking_text: Текст thinking блока
            
        Returns:
            True если блок завершен
        """
        # Простая эвристика - блок завершен если есть заключение
        completion_indicators = [
            "поэтому",
            "следовательно",
            "итак",
            "в результате",
            "таким образом"
        ]
        
        text_lower = thinking_text.lower()
        return any(indicator in text_lower for indicator in completion_indicators)
    
    def _create_thinking_step(self, thinking_text: str, step_number: int) -> Optional[ReasoningStep]:
        """
        Создает шаг рассуждения из thinking блока.
        
        Args:
            thinking_text: Текст thinking блока
            step_number: Номер шага
            
        Returns:
            Шаг рассуждения или None
        """
        if not thinking_text.strip():
            return None
        
        return ReasoningStep(
            step_number=step_number,
            thought=thinking_text.strip(),
            confidence=0.8,
            thinking_block=ThinkingBlock(
                content=thinking_text.strip(),
                token_count=self._count_tokens(thinking_text)
            )
        )
    
    def _count_tokens(self, text: str) -> int:
        """
        Приблизительный подсчет токенов в тексте.
        
        Args:
            text: Текст для подсчета
            
        Returns:
            Приблизительное количество токенов
        """
        # Простая эвристика: ~4 символа на токен для русского текста
        return len(text) // 4
    
    def _calculate_average_confidence(self, steps: List[ReasoningStep]) -> Optional[float]:
        """
        Вычисляет среднюю уверенность по шагам.
        
        Args:
            steps: Список шагов рассуждения
            
        Returns:
            Средняя уверенность или None
        """
        confidences = [step.confidence for step in steps if step.confidence is not None]
        return sum(confidences) / len(confidences) if confidences else None
    
    def _extract_thinking_from_content(self, content: str) -> tuple[str, str]:
        """
        Универсальное извлечение thinking контента из текста.
        
        Ищет все возможные варианты thinking токенов и возвращает первый найденный.
        
        Args:
            content: Текст для анализа
            
        Returns:
            Кортеж (thinking_content, final_answer)
        """
        import re
        
        # Получаем все возможные токены
        all_tokens = self._get_all_possible_thinking_tokens()
        
        for start_token, end_token in all_tokens:
            # Экранируем токены для regex
            start_escaped = re.escape(start_token)
            end_escaped = re.escape(end_token)
            
            # Ищем полный блок с начальным и конечным токеном
            full_pattern = f'{start_escaped}(.*?){end_escaped}'
            full_match = re.search(full_pattern, content, re.DOTALL)
            
            if full_match:
                thinking_content = full_match.group(1).strip()
                final_answer = re.sub(full_pattern, '', content, flags=re.DOTALL).strip()
                logger.info(f"Найден thinking блок с токенами: {start_token} ... {end_token}")
                return thinking_content, final_answer
            
            # Ищем только конечный токен (как в Qwen3-Thinking)
            if end_token in content and start_token not in content:
                end_idx = content.find(end_token)
                thinking_content = content[:end_idx].strip()
                final_answer = content[end_idx + len(end_token):].strip()
                
                if thinking_content:  # Проверяем, что есть контент до конечного токена
                    logger.info(f"Найден thinking блок только с конечным токеном: {end_token}")
                    return thinking_content, final_answer
        
        # Если ничего не найдено
        return "", content
    
    def _prepare_thinking_style_prompt(
        self,
        messages: List[Dict[str, str]],
        problem_type: str = "general"
    ) -> List[Dict[str, str]]:
        """
        Подготавливает промпт в стиле thinking для обычных моделей.
        
        Использует подходящие thinking токены в зависимости от модели.
        
        Args:
            messages: Исходные сообщения
            problem_type: Тип задачи
            
        Returns:
            Сообщения с thinking-style промптом
        """
        # Получаем подходящие токены для модели
        start_token, end_token = self._get_thinking_tokens()
        
        thinking_template = f"""
Ты эксперт в пошаговом рассуждении. Реши задачу, показав свой внутренний процесс мышления.

Формат ответа:
{start_token}
[Здесь покажи свои внутренние рассуждения, анализ проблемы, рассмотрение различных подходов]
{end_token}

[Здесь дай финальный ответ пользователю]

Важно:
- В блоке {start_token}...{end_token} покажи весь процесс рассуждения
- Анализируй проблему с разных сторон
- Рассматривай альтернативные решения
- Проверяй свои выводы
- Финальный ответ должен быть четким и полным
- Не повторяй токены {start_token} и {end_token} в финальном ответе
"""
        
        # Создаем новые сообщения с thinking промптом
        thinking_messages = messages.copy()
        
        # Добавляем системный промпт для thinking стиля
        system_message = {
            "role": "system",
            "content": thinking_template
        }
        
        # Вставляем системное сообщение
        if thinking_messages and thinking_messages[0]["role"] == "system":
            thinking_messages[0]["content"] += "\n\n" + system_message["content"]
        else:
            thinking_messages.insert(0, system_message)
        
        return thinking_messages
    
    def _parse_thinking_style_response(self, response: str) -> ReasoningChain:
        """
        Парсит ответ в thinking стиле от обычной модели.
        
        Поддерживает различные thinking токены в зависимости от модели.
        
        Args:
            response: Ответ модели с thinking блоками
            
        Returns:
            Цепочка рассуждений с имитированными thinking блоками
        """
        import re
        
        # Используем универсальный метод извлечения thinking контента
        thinking_content, final_answer = self._extract_thinking_from_content(response)
        
        steps = []
        thinking_blocks = []
        
        if thinking_content:
            # Создаем thinking блок
            thinking_block = ThinkingBlock(
                content=thinking_content,
                token_count=self._count_tokens(thinking_content)
            )
            thinking_blocks.append(thinking_block)
            
            # Разбиваем thinking на шаги
            thinking_parts = [part.strip() for part in thinking_content.split('\n\n') if part.strip()]
            
            for i, part in enumerate(thinking_parts, 1):
                step = ReasoningStep(
                    step_number=i,
                    thought=part,
                    confidence=0.8,
                    thinking_block=ThinkingBlock(
                        content=part,
                        token_count=self._count_tokens(part)
                    )
                )
                steps.append(step)
        
        # Если нет шагов, создаем один базовый
        if not steps:
            steps.append(ReasoningStep(
                step_number=1,
                thought="Рассуждение модели",
                observation=final_answer[:200] + "..." if len(final_answer) > 200 else final_answer,
                confidence=0.7
            ))
        
        # Подсчитываем reasoning токены
        total_reasoning_tokens = sum(
            block.token_count for block in thinking_blocks 
            if block.token_count
        ) or None
        
        return ReasoningChain(
            steps=steps,
            final_answer=final_answer,
            total_reasoning_tokens=total_reasoning_tokens,
            confidence_score=self._calculate_average_confidence(steps),
            model_type=ReasoningModelType.NATIVE_THINKING,
            thinking_blocks=thinking_blocks
        )

    # Методы базового класса (заглушки для совместимости)
    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Базовый chat completion через стандартный клиент"""
        # Используем стандартный клиент для базовых операций
        from .standard import StandardLLMClient
        standard_client = StandardLLMClient(self.config)
        return await standard_client.chat_completion(messages, **kwargs)
    
    async def chat_completion_stream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """Базовый streaming через стандартный клиент"""
        from .streaming import StreamingLLMClient
        streaming_client = StreamingLLMClient(self.config)
        async for chunk in streaming_client.chat_completion_stream(messages, **kwargs):
            yield chunk
    
    async def chat_completion_structured(self, messages: List[Dict[str, str]], response_model, **kwargs):
        """Structured output не поддерживается в reasoning режиме"""
        raise NotImplementedError(
            "ReasoningLLMClient не поддерживает structured output. "
            "Используйте StructuredLLMClient для структурированных ответов."
        )