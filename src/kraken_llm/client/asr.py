"""
ASR (Automatic Speech Recognition) клиент для Kraken LLM фреймворка.

Этот модуль предоставляет ASRClient для работы с речевыми технологиями:
- Распознавание речи (Speech-to-Text)
- Генерация речи (Text-to-Speech)
- Детекция речевой активности (Voice Activity Detection)
- Диаризация спикеров (Speaker Diarization)
- Анализ эмоций в речи
"""

import asyncio
import base64
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, AsyncGenerator, Tuple
from pydantic import BaseModel, Field
import logging

from .base import BaseLLMClient
from ..exceptions.validation import ValidationError
from ..exceptions.api import APIError
from ..utils.media import MediaUtils

logger = logging.getLogger(__name__)


class ASRSegment(BaseModel):
    """Сегмент распознанной речи"""
    start_time: float = Field(..., description="Время начала сегмента в секундах")
    end_time: float = Field(..., description="Время окончания сегмента в секундах")
    text: str = Field(..., description="Распознанный текст")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Уверенность распознавания")
    speaker_id: Optional[str] = Field(None, description="Идентификатор спикера")
    language: Optional[str] = Field(None, description="Язык сегмента")


class SpeakerInfo(BaseModel):
    """Информация о спикере"""
    speaker_id: str = Field(..., description="Идентификатор спикера")
    total_duration: float = Field(..., description="Общая длительность речи спикера")
    segments_count: int = Field(..., description="Количество сегментов")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Средняя уверенность")
    gender: Optional[str] = Field(None, description="Пол спикера (если определен)")
    age_group: Optional[str] = Field(None, description="Возрастная группа")


class VADResult(BaseModel):
    """Результат детекции речевой активности"""
    speech_segments: List[Tuple[float, float]] = Field(..., description="Сегменты с речью (начало, конец)")
    total_speech_duration: float = Field(..., description="Общая длительность речи")
    total_silence_duration: float = Field(..., description="Общая длительность тишины")
    speech_ratio: float = Field(..., ge=0.0, le=1.0, description="Доля речи в аудио")


class EmotionAnalysis(BaseModel):
    """Анализ эмоций в речи"""
    dominant_emotion: str = Field(..., description="Доминирующая эмоция")
    emotions: Dict[str, float] = Field(..., description="Распределение эмоций")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность анализа")
    valence: float = Field(..., ge=-1.0, le=1.0, description="Валентность (-1 негативная, +1 позитивная)")
    arousal: float = Field(..., ge=0.0, le=1.0, description="Возбуждение (0 спокойствие, 1 активность)")


class ASRConfig(BaseModel):
    """Конфигурация ASR клиента"""
    # Общие настройки
    default_language: str = Field("ru", description="Язык по умолчанию")
    enable_timestamps: bool = Field(True, description="Включить временные метки")
    enable_word_timestamps: bool = Field(False, description="Временные метки для слов")
    
    # Настройки качества
    quality_level: str = Field("high", description="Уровень качества (low, medium, high)")
    enable_noise_reduction: bool = Field(True, description="Подавление шума")
    enable_echo_cancellation: bool = Field(True, description="Подавление эха")
    
    # Диаризация
    enable_speaker_diarization: bool = Field(False, description="Включить диаризацию спикеров")
    max_speakers: int = Field(10, ge=1, le=50, description="Максимальное количество спикеров")
    min_speaker_duration: float = Field(1.0, ge=0.1, description="Минимальная длительность речи спикера")
    
    # VAD настройки
    vad_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Порог детекции речи")
    vad_min_speech_duration: float = Field(0.25, description="Минимальная длительность речи")
    vad_min_silence_duration: float = Field(0.1, description="Минимальная длительность тишины")
    
    # TTS настройки
    tts_voice: str = Field("default", description="Голос для синтеза речи")
    tts_speed: float = Field(1.0, ge=0.1, le=3.0, description="Скорость речи")
    tts_pitch: float = Field(1.0, ge=0.1, le=2.0, description="Высота тона")
    tts_volume: float = Field(1.0, ge=0.1, le=2.0, description="Громкость")
    
    # Форматы
    supported_audio_formats: List[str] = Field(
        default=["wav", "mp3", "flac", "ogg", "m4a", "webm"],
        description="Поддерживаемые аудио форматы"
    )
    output_audio_format: str = Field("wav", description="Формат выходного аудио")
    sample_rate: int = Field(16000, description="Частота дискретизации")
    
    # Лимиты
    max_audio_duration: int = Field(3600, description="Максимальная длительность аудио в секундах")
    max_file_size: int = Field(100 * 1024 * 1024, description="Максимальный размер файла в байтах")


class ASRClient(BaseLLMClient):
    """
    Клиент для работы с ASR (Automatic Speech Recognition) технологиями.
    
    Поддерживает:
    - Распознавание речи (Speech-to-Text)
    - Генерация речи (Text-to-Speech)  
    - Детекция речевой активности (VAD)
    - Диаризация спикеров
    - Анализ эмоций в речи
    """
    
    def __init__(self, config, asr_config: Optional[ASRConfig] = None):
        """
        Инициализация ASR клиента.
        
        Args:
            config: Базовая конфигурация LLM
            asr_config: Конфигурация ASR функций
        """
        super().__init__(config)
        self.asr_config = asr_config or ASRConfig()
        logger.info("Инициализирован ASRClient")
    
    async def speech_to_text(
        self,
        audio_file: Union[str, Path],
        language: Optional[str] = None,
        enable_diarization: Optional[bool] = None,
        enable_emotions: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Распознавание речи из аудио файла.
        
        Args:
            audio_file: Путь к аудио файлу
            language: Язык для распознавания
            enable_diarization: Включить диаризацию спикеров
            enable_emotions: Включить анализ эмоций
            **kwargs: Дополнительные параметры
            
        Returns:
            Результат распознавания речи
        """
        logger.info(f"Начинаем распознавание речи: {audio_file}")
        
        # Валидация аудио файла
        await self._validate_audio_file(audio_file)
        
        # Подготовка параметров
        params = {
            "language": language or self.asr_config.default_language,
            "enable_timestamps": self.asr_config.enable_timestamps,
            "enable_word_timestamps": self.asr_config.enable_word_timestamps,
            "quality": self.asr_config.quality_level,
            "noise_reduction": self.asr_config.enable_noise_reduction,
            "echo_cancellation": self.asr_config.enable_echo_cancellation,
            **kwargs
        }
        
        # Диаризация спикеров
        if enable_diarization or self.asr_config.enable_speaker_diarization:
            params.update({
                "enable_diarization": True,
                "max_speakers": self.asr_config.max_speakers,
                "min_speaker_duration": self.asr_config.min_speaker_duration
            })
        
        # Загружаем аудио файл
        audio_data = await self._load_audio_data(audio_file)
        
        try:
            # Выполняем распознавание через LLM API
            result = await self._call_asr_api("speech-to-text", audio_data, params)
            
            # Обрабатываем результат
            processed_result = await self._process_stt_result(result, enable_emotions)
            
            logger.info(f"Распознавание завершено: {len(processed_result.get('segments', []))} сегментов")
            return processed_result
            
        except Exception as e:
            logger.error(f"Ошибка распознавания речи: {e}")
            raise APIError(f"Ошибка распознавания речи: {e}")
    
    async def text_to_speech(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        output_file: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Union[bytes, Path]:
        """
        Синтез речи из текста.
        
        Args:
            text: Текст для синтеза
            voice: Голос для синтеза
            language: Язык синтеза
            output_file: Путь для сохранения аудио файла
            **kwargs: Дополнительные параметры
            
        Returns:
            Аудио данные или путь к файлу
        """
        logger.info(f"Начинаем синтез речи: {len(text)} символов")
        
        if not text.strip():
            raise ValidationError("Текст для синтеза не может быть пустым")
        
        # Подготовка параметров
        params = {
            "text": text,
            "voice": voice or self.asr_config.tts_voice,
            "language": language or self.asr_config.default_language,
            "speed": self.asr_config.tts_speed,
            "pitch": self.asr_config.tts_pitch,
            "volume": self.asr_config.tts_volume,
            "format": self.asr_config.output_audio_format,
            "sample_rate": self.asr_config.sample_rate,
            **kwargs
        }
        
        try:
            # Выполняем синтез через LLM API
            result = await self._call_asr_api("text-to-speech", None, params)
            
            # Получаем аудио данные
            audio_data = await self._extract_audio_data(result)
            
            # Сохраняем в файл если указан путь
            if output_file:
                output_path = Path(output_file)
                with open(output_path, 'wb') as f:
                    f.write(audio_data)
                logger.info(f"Аудио сохранено: {output_path}")
                return output_path
            
            logger.info(f"Синтез завершен: {len(audio_data)} байт")
            return audio_data
            
        except Exception as e:
            logger.error(f"Ошибка синтеза речи: {e}")
            raise APIError(f"Ошибка синтеза речи: {e}")
    
    async def voice_activity_detection(
        self,
        audio_file: Union[str, Path],
        **kwargs
    ) -> VADResult:
        """
        Детекция речевой активности в аудио.
        
        Args:
            audio_file: Путь к аудио файлу
            **kwargs: Дополнительные параметры
            
        Returns:
            Результат детекции речевой активности
        """
        logger.info(f"Начинаем VAD анализ: {audio_file}")
        
        # Валидация аудио файла
        await self._validate_audio_file(audio_file)
        
        # Подготовка параметров
        params = {
            "threshold": self.asr_config.vad_threshold,
            "min_speech_duration": self.asr_config.vad_min_speech_duration,
            "min_silence_duration": self.asr_config.vad_min_silence_duration,
            **kwargs
        }
        
        # Загружаем аудио файл
        audio_data = await self._load_audio_data(audio_file)
        
        try:
            # Выполняем VAD через LLM API
            result = await self._call_asr_api("voice-activity-detection", audio_data, params)
            
            # Обрабатываем результат
            vad_result = VADResult(
                speech_segments=result.get("speech_segments", []),
                total_speech_duration=result.get("total_speech_duration", 0.0),
                total_silence_duration=result.get("total_silence_duration", 0.0),
                speech_ratio=result.get("speech_ratio", 0.0)
            )
            
            logger.info(f"VAD завершен: {len(vad_result.speech_segments)} сегментов речи")
            return vad_result
            
        except Exception as e:
            logger.error(f"Ошибка VAD анализа: {e}")
            raise APIError(f"Ошибка VAD анализа: {e}")
    
    async def speaker_diarization(
        self,
        audio_file: Union[str, Path],
        num_speakers: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Диаризация спикеров в аудио.
        
        Args:
            audio_file: Путь к аудио файлу
            num_speakers: Ожидаемое количество спикеров
            **kwargs: Дополнительные параметры
            
        Returns:
            Результат диаризации спикеров
        """
        logger.info(f"Начинаем диаризацию спикеров: {audio_file}")
        
        # Валидация аудио файла
        await self._validate_audio_file(audio_file)
        
        # Подготовка параметров
        params = {
            "max_speakers": num_speakers or self.asr_config.max_speakers,
            "min_speaker_duration": self.asr_config.min_speaker_duration,
            **kwargs
        }
        
        # Загружаем аудио файл
        audio_data = await self._load_audio_data(audio_file)
        
        try:
            # Выполняем диаризацию через LLM API
            result = await self._call_asr_api("speaker-diarization", audio_data, params)
            
            # Обрабатываем результат
            speakers = []
            for speaker_data in result.get("speakers", []):
                speaker = SpeakerInfo(
                    speaker_id=speaker_data["speaker_id"],
                    total_duration=speaker_data["total_duration"],
                    segments_count=speaker_data["segments_count"],
                    confidence=speaker_data.get("confidence", 0.0),
                    gender=speaker_data.get("gender"),
                    age_group=speaker_data.get("age_group")
                )
                speakers.append(speaker)
            
            processed_result = {
                "speakers": speakers,
                "segments": result.get("segments", []),
                "total_speakers": len(speakers),
                "total_duration": result.get("total_duration", 0.0)
            }
            
            logger.info(f"Диаризация завершена: {len(speakers)} спикеров")
            return processed_result
            
        except Exception as e:
            logger.error(f"Ошибка диаризации спикеров: {e}")
            raise APIError(f"Ошибка диаризации спикеров: {e}")
    
    async def emotion_analysis(
        self,
        audio_file: Union[str, Path],
        **kwargs
    ) -> EmotionAnalysis:
        """
        Анализ эмоций в речи.
        
        Args:
            audio_file: Путь к аудио файлу
            **kwargs: Дополнительные параметры
            
        Returns:
            Результат анализа эмоций
        """
        logger.info(f"Начинаем анализ эмоций: {audio_file}")
        
        # Валидация аудио файла
        await self._validate_audio_file(audio_file)
        
        # Загружаем аудио файл
        audio_data = await self._load_audio_data(audio_file)
        
        try:
            # Выполняем анализ эмоций через LLM API
            result = await self._call_asr_api("emotion-analysis", audio_data, kwargs)
            
            # Обрабатываем результат
            emotion_result = EmotionAnalysis(
                dominant_emotion=result.get("dominant_emotion", "neutral"),
                emotions=result.get("emotions", {}),
                confidence=result.get("confidence", 0.0),
                valence=result.get("valence", 0.0),
                arousal=result.get("arousal", 0.0)
            )
            
            logger.info(f"Анализ эмоций завершен: {emotion_result.dominant_emotion}")
            return emotion_result
            
        except Exception as e:
            logger.error(f"Ошибка анализа эмоций: {e}")
            raise APIError(f"Ошибка анализа эмоций: {e}")
    
    async def transcribe_with_analysis(
        self,
        audio_file: Union[str, Path],
        include_diarization: bool = True,
        include_emotions: bool = True,
        include_vad: bool = True,
        language: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Комплексная обработка аудио с полным анализом.
        
        Args:
            audio_file: Путь к аудио файлу
            include_diarization: Включить диаризацию спикеров
            include_emotions: Включить анализ эмоций
            include_vad: Включить детекцию речевой активности
            language: Язык для распознавания
            **kwargs: Дополнительные параметры
            
        Returns:
            Комплексный результат анализа
        """
        logger.info(f"Начинаем комплексный анализ аудио: {audio_file}")
        
        results = {}
        
        # Основное распознавание речи
        stt_result = await self.speech_to_text(
            audio_file,
            language=language,
            enable_diarization=include_diarization,
            enable_emotions=include_emotions,
            **kwargs
        )
        results["transcription"] = stt_result
        
        # VAD анализ
        if include_vad:
            try:
                vad_result = await self.voice_activity_detection(audio_file)
                results["voice_activity"] = vad_result
            except Exception as e:
                logger.warning(f"VAD анализ не удался: {e}")
                results["voice_activity"] = None
        
        # Диаризация спикеров (если не включена в основном распознавании)
        if include_diarization and not stt_result.get("speakers"):
            try:
                diarization_result = await self.speaker_diarization(audio_file)
                results["diarization"] = diarization_result
            except Exception as e:
                logger.warning(f"Диаризация не удалась: {e}")
                results["diarization"] = None
        
        # Анализ эмоций (если не включен в основном распознавании)
        if include_emotions and not stt_result.get("emotions"):
            try:
                emotion_result = await self.emotion_analysis(audio_file)
                results["emotions"] = emotion_result
            except Exception as e:
                logger.warning(f"Анализ эмоций не удался: {e}")
                results["emotions"] = None
        
        # Сводная статистика
        results["summary"] = self._generate_analysis_summary(results)
        
        logger.info("Комплексный анализ завершен")
        return results
    
    async def streaming_speech_to_text(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        language: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Потоковое распознавание речи.
        
        Args:
            audio_stream: Поток аудио данных
            language: Язык для распознавания
            **kwargs: Дополнительные параметры
            
        Yields:
            Промежуточные результаты распознавания
        """
        logger.info("Начинаем потоковое распознавание речи")
        
        # Подготовка параметров
        params = {
            "language": language or self.asr_config.default_language,
            "enable_timestamps": True,
            "streaming": True,
            **kwargs
        }
        
        try:
            # Инициализируем потоковое соединение
            session_id = await self._init_streaming_session("streaming-stt", params)
            
            # Обрабатываем поток аудио
            async for audio_chunk in audio_stream:
                # Отправляем chunk на сервер
                result = await self._send_audio_chunk(session_id, audio_chunk)
                
                if result and result.get("text"):
                    yield {
                        "text": result["text"],
                        "is_final": result.get("is_final", False),
                        "confidence": result.get("confidence", 0.0),
                        "timestamp": result.get("timestamp", 0.0)
                    }
            
            # Завершаем сессию
            final_result = await self._finalize_streaming_session(session_id)
            if final_result:
                yield final_result
                
        except Exception as e:
            logger.error(f"Ошибка потокового распознавания: {e}")
            raise APIError(f"Ошибка потокового распознавания: {e}")
    
    # Вспомогательные методы
    
    async def _validate_audio_file(self, audio_file: Union[str, Path]) -> None:
        """Валидация аудио файла"""
        path = Path(audio_file)
        
        if not path.exists():
            raise ValidationError(f"Аудио файл не найден: {audio_file}")
        
        # Проверяем размер файла
        file_size = path.stat().st_size
        if file_size > self.asr_config.max_file_size:
            raise ValidationError(
                f"Размер файла ({file_size} байт) превышает максимальный "
                f"({self.asr_config.max_file_size} байт)"
            )
        
        # Проверяем формат
        extension = path.suffix[1:].lower()
        if extension not in self.asr_config.supported_audio_formats:
            raise ValidationError(f"Неподдерживаемый формат аудио: {extension}")
        
        # Дополнительная валидация через MediaUtils
        validation_result = MediaUtils.validate_media_file(
            path, 'audio', max_size=self.asr_config.max_file_size
        )
        
        if not validation_result['valid']:
            errors = "; ".join(validation_result['errors'])
            raise ValidationError(f"Валидация аудио файла не прошла: {errors}")
    
    async def _load_audio_data(self, audio_file: Union[str, Path]) -> str:
        """Загрузка аудио данных в base64"""
        try:
            return MediaUtils.encode_file_to_base64(audio_file)
        except Exception as e:
            raise ValidationError(f"Ошибка загрузки аудио файла: {e}")
    
    async def _call_asr_api(
        self, 
        endpoint: str, 
        audio_data: Optional[str], 
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Вызов ASR API через базовый LLM клиент"""
        
        # Формируем запрос для LLM API
        messages = [
            {
                "role": "system",
                "content": f"Выполни ASR операцию: {endpoint}"
            },
            {
                "role": "user", 
                "content": json.dumps({
                    "operation": endpoint,
                    "audio_data": audio_data,
                    "parameters": params
                })
            }
        ]
        
        try:
            # Используем базовый chat completion
            response = await self.chat_completion(messages)
            
            # Парсим JSON ответ
            if isinstance(response, str):
                try:
                    return json.loads(response)
                except json.JSONDecodeError:
                    # Если не JSON, возвращаем как текст
                    return {"result": response}
            
            return response
            
        except Exception as e:
            raise APIError(f"Ошибка вызова ASR API: {e}")
    
    async def _process_stt_result(
        self, 
        result: Dict[str, Any], 
        include_emotions: bool = False
    ) -> Dict[str, Any]:
        """Обработка результата распознавания речи"""
        
        processed = {
            "text": result.get("text", ""),
            "language": result.get("language", self.asr_config.default_language),
            "confidence": result.get("confidence", 0.0),
            "duration": result.get("duration", 0.0)
        }
        
        # Обрабатываем сегменты
        if "segments" in result:
            segments = []
            for seg_data in result["segments"]:
                segment = ASRSegment(
                    start_time=seg_data["start_time"],
                    end_time=seg_data["end_time"],
                    text=seg_data["text"],
                    confidence=seg_data.get("confidence", 0.0),
                    speaker_id=seg_data.get("speaker_id"),
                    language=seg_data.get("language")
                )
                segments.append(segment)
            processed["segments"] = segments
        
        # Обрабатываем информацию о спикерах
        if "speakers" in result:
            speakers = []
            for speaker_data in result["speakers"]:
                speaker = SpeakerInfo(
                    speaker_id=speaker_data["speaker_id"],
                    total_duration=speaker_data["total_duration"],
                    segments_count=speaker_data["segments_count"],
                    confidence=speaker_data.get("confidence", 0.0),
                    gender=speaker_data.get("gender"),
                    age_group=speaker_data.get("age_group")
                )
                speakers.append(speaker)
            processed["speakers"] = speakers
        
        # Обрабатываем эмоции
        if include_emotions and "emotions" in result:
            emotion_data = result["emotions"]
            emotions = EmotionAnalysis(
                dominant_emotion=emotion_data.get("dominant_emotion", "neutral"),
                emotions=emotion_data.get("emotions", {}),
                confidence=emotion_data.get("confidence", 0.0),
                valence=emotion_data.get("valence", 0.0),
                arousal=emotion_data.get("arousal", 0.0)
            )
            processed["emotions"] = emotions
        
        return processed
    
    async def _extract_audio_data(self, result: Dict[str, Any]) -> bytes:
        """Извлечение аудио данных из результата"""
        if "audio_data" in result:
            try:
                return base64.b64decode(result["audio_data"])
            except Exception as e:
                raise APIError(f"Ошибка декодирования аудио данных: {e}")
        
        raise APIError("Аудио данные не найдены в ответе")
    
    def _generate_analysis_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Генерация сводки анализа"""
        summary = {
            "total_duration": 0.0,
            "total_speakers": 0,
            "speech_ratio": 0.0,
            "dominant_emotion": "neutral",
            "confidence_avg": 0.0
        }
        
        # Из транскрипции
        if "transcription" in results and results["transcription"]:
            trans = results["transcription"]
            summary["total_duration"] = trans.get("duration", 0.0)
            summary["confidence_avg"] = trans.get("confidence", 0.0)
            
            if "speakers" in trans:
                summary["total_speakers"] = len(trans["speakers"])
        
        # Из VAD
        if "voice_activity" in results and results["voice_activity"]:
            vad = results["voice_activity"]
            summary["speech_ratio"] = vad.speech_ratio
        
        # Из анализа эмоций
        if "emotions" in results and results["emotions"]:
            emotions = results["emotions"]
            summary["dominant_emotion"] = emotions.dominant_emotion
        
        return summary
    
    # Потоковые методы (заглушки для будущей реализации)
    
    async def _init_streaming_session(self, operation: str, params: Dict[str, Any]) -> str:
        """Инициализация потоковой сессии"""
        # TODO: Реализовать инициализацию потоковой сессии
        return f"session_{operation}_{id(params)}"
    
    async def _send_audio_chunk(self, session_id: str, audio_chunk: bytes) -> Optional[Dict[str, Any]]:
        """Отправка chunk аудио данных"""
        # TODO: Реализовать отправку chunk
        return None
    
    async def _finalize_streaming_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Завершение потоковой сессии"""
        # TODO: Реализовать завершение сессии
        return None
    
    # Методы базового класса (делегирование к стандартному клиенту)
    
    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Базовый chat completion через стандартный клиент"""
        from .standard import StandardLLMClient
        standard_client = StandardLLMClient(self.config)
        return await standard_client.chat_completion(messages, **kwargs)
    
    async def chat_completion_stream(self, messages: List[Dict[str, str]], **kwargs):
        """Streaming не поддерживается в ASR режиме"""
        raise NotImplementedError(
            "ASRClient не поддерживает обычный streaming режим. "
            "Используйте streaming_speech_to_text для потокового распознавания речи."
        )
    
    async def chat_completion_structured(self, messages: List[Dict[str, str]], response_model, **kwargs):
        """Structured output не поддерживается в ASR режиме"""
        raise NotImplementedError(
            "ASRClient не поддерживает structured output. "
            "Используйте StructuredLLMClient для структурированных ответов."
        )
    
    # Утилитарные методы
    
    @staticmethod
    def get_supported_languages() -> List[str]:
        """Возвращает список поддерживаемых языков"""
        return [
            "ru", "en", "de", "fr", "es", "it", "pt", "nl", "pl", "tr",
            "zh", "ja", "ko", "ar", "hi", "th", "vi", "uk", "cs", "sk"
        ]
    
    @staticmethod
    def get_supported_voices() -> Dict[str, List[str]]:
        """Возвращает список поддерживаемых голосов по языкам"""
        return {
            "ru": ["elena", "pavel", "irina", "maxim"],
            "en": ["alice", "brian", "emma", "ryan"],
            "de": ["marlene", "hans", "vicki", "daniel"],
            "fr": ["celine", "mathieu", "lea", "remy"],
            "es": ["conchita", "enrique", "lucia", "miguel"]
        }
    
    @staticmethod
    def get_supported_emotions() -> List[str]:
        """Возвращает список распознаваемых эмоций"""
        return [
            "neutral", "happy", "sad", "angry", "fear", "surprise", 
            "disgust", "contempt", "excitement", "calm", "stress"
        ]
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Возвращает сводку текущей конфигурации"""
        return {
            "default_language": self.asr_config.default_language,
            "quality_level": self.asr_config.quality_level,
            "max_speakers": self.asr_config.max_speakers,
            "supported_formats": self.asr_config.supported_audio_formats,
            "max_file_size_mb": self.asr_config.max_file_size / (1024 * 1024),
            "max_duration_hours": self.asr_config.max_audio_duration / 3600,
            "features": {
                "speaker_diarization": self.asr_config.enable_speaker_diarization,
                "noise_reduction": self.asr_config.enable_noise_reduction,
                "echo_cancellation": self.asr_config.enable_echo_cancellation,
                "timestamps": self.asr_config.enable_timestamps
            }
        }