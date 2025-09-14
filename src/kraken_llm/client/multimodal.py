"""
Мультимодальный клиент для Kraken LLM фреймворка.

Этот модуль предоставляет MultimodalLLMClient для работы с моделями,
поддерживающими различные типы медиа: изображения, аудио, видео.
"""

import base64
import mimetypes
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, BinaryIO
from pydantic import BaseModel, Field, field_validator
import logging

from .base import BaseLLMClient
from ..exceptions.validation import ValidationError
from ..exceptions.api import APIError
from ..utils.media import MediaUtils

logger = logging.getLogger(__name__)


class MediaFile(BaseModel):
    """Модель для медиа файла"""
    content: str = Field(..., description="Содержимое файла в base64")
    mime_type: str = Field(..., description="MIME тип файла")
    filename: Optional[str] = Field(None, description="Имя файла")
    size: Optional[int] = Field(None, description="Размер файла в байтах")
    
    @field_validator('mime_type')
    @classmethod
    def validate_mime_type(cls, v):
        """Валидация MIME типа"""
        allowed_types = [
            # Изображения
            'image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/bmp',
            # Аудио
            'audio/mpeg', 'audio/wav', 'audio/x-wav', 'audio/ogg', 'audio/mp4', 'audio/webm',
            # Видео
            'video/mp4', 'video/webm', 'video/ogg', 'video/avi', 'video/mov'
        ]
        
        if v not in allowed_types:
            raise ValueError(f"Неподдерживаемый MIME тип: {v}")
        return v


class MultimodalMessage(BaseModel):
    """Модель для мультимодального сообщения"""
    role: str = Field(..., description="Роль отправителя")
    content: Union[str, List[Dict[str, Any]]] = Field(..., description="Содержимое сообщения")
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v):
        """Валидация содержимого сообщения"""
        if isinstance(v, str):
            return v
        
        if isinstance(v, list):
            for item in v:
                if not isinstance(item, dict):
                    raise ValueError("Элементы content должны быть словарями")
                
                if 'type' not in item:
                    raise ValueError("Каждый элемент должен содержать поле 'type'")
                
                allowed_types = ['text', 'image_url', 'audio_url', 'video_url']
                if item['type'] not in allowed_types:
                    raise ValueError(f"Неподдерживаемый тип контента: {item['type']}")
        
        return v


class MultimodalConfig(BaseModel):
    """Конфигурация для мультимодального режима"""
    max_image_size: int = Field(20 * 1024 * 1024, description="Максимальный размер изображения в байтах")
    max_audio_duration: int = Field(300, description="Максимальная длительность аудио в секундах")
    max_video_duration: int = Field(60, description="Максимальная длительность видео в секундах")
    supported_image_formats: List[str] = Field(
        default=['jpeg', 'png', 'gif', 'webp', 'bmp'],
        description="Поддерживаемые форматы изображений"
    )
    supported_audio_formats: List[str] = Field(
        default=['mp3', 'wav', 'ogg', 'm4a', 'webm'],
        description="Поддерживаемые форматы аудио"
    )
    supported_video_formats: List[str] = Field(
        default=['mp4', 'webm', 'ogg', 'avi', 'mov'],
        description="Поддерживаемые форматы видео"
    )
    auto_resize_images: bool = Field(True, description="Автоматически изменять размер больших изображений")
    extract_video_frames: bool = Field(False, description="Извлекать кадры из видео для анализа")


class MultimodalLLMClient(BaseLLMClient):
    """
    Клиент для работы с мультимодальными моделями.
    
    Поддерживает:
    - Обработку изображений (vision models)
    - Анализ аудио файлов
    - Обработку видео контента
    - Комбинированные мультимодальные запросы
    """
    
    def __init__(self, config, multimodal_config: Optional[MultimodalConfig] = None):
        """
        Инициализация мультимодального клиента.
        
        Args:
            config: Базовая конфигурация LLM
            multimodal_config: Конфигурация мультимодального режима
        """
        super().__init__(config)
        self.multimodal_config = multimodal_config or MultimodalConfig()
        self._asr_client = None  # Ленивая инициализация ASR клиента
        logger.info("Инициализирован MultimodalLLMClient")
    
    async def vision_completion(
        self,
        text_prompt: str,
        images: Union[str, Path, List[Union[str, Path]]],
        detail_level: str = "auto",
        **kwargs
    ) -> str:
        """
        Выполняет анализ изображений с текстовым промптом.
        
        Args:
            text_prompt: Текстовый промпт для анализа
            images: Путь к изображению или список путей
            detail_level: Уровень детализации (low, high, auto)
            **kwargs: Дополнительные параметры
            
        Returns:
            Ответ модели с анализом изображений
        """
        logger.info(f"Начинаем vision completion с {len(images) if isinstance(images, list) else 1} изображениями")
        
        # Нормализуем входные данные
        if not isinstance(images, list):
            images = [images]
        
        # Загружаем и валидируем изображения
        media_files = []
        for image_path in images:
            media_file = await self._load_image_file(image_path)
            media_files.append(media_file)
        
        # Создаем мультимодальное сообщение
        content = [{"type": "text", "text": text_prompt}]
        
        for media_file in media_files:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_file.mime_type};base64,{media_file.content}",
                    "detail": detail_level
                }
            })
        
        messages = [MultimodalMessage(role="user", content=content)]
        
        return await self._multimodal_completion(messages, **kwargs)
    
    async def audio_completion(
        self,
        text_prompt: str,
        audio_files: Union[str, Path, List[Union[str, Path]]],
        task_type: str = "transcription",
        **kwargs
    ) -> str:
        """
        Выполняет обработку аудио файлов.
        
        Args:
            text_prompt: Текстовый промпт для обработки
            audio_files: Путь к аудио файлу или список путей
            task_type: Тип задачи (transcription, analysis, translation)
            **kwargs: Дополнительные параметры
            
        Returns:
            Ответ модели с обработкой аудио
        """
        logger.info(f"Начинаем audio completion с задачей: {task_type}")
        
        # Нормализуем входные данные
        if not isinstance(audio_files, list):
            audio_files = [audio_files]
        
        # Загружаем и валидируем аудио файлы
        media_files = []
        for audio_path in audio_files:
            media_file = await self._load_audio_file(audio_path)
            media_files.append(media_file)
        
        # Создаем мультимодальное сообщение
        content = [{"type": "text", "text": f"{text_prompt}\n\nЗадача: {task_type}"}]
        
        for media_file in media_files:
            content.append({
                "type": "audio_url",
                "audio_url": {
                    "url": f"data:{media_file.mime_type};base64,{media_file.content}"
                }
            })
        
        messages = [MultimodalMessage(role="user", content=content)]
        
        return await self._multimodal_completion(messages, **kwargs)
    
    async def video_completion(
        self,
        text_prompt: str,
        video_files: Union[str, Path, List[Union[str, Path]]],
        analysis_type: str = "description",
        **kwargs
    ) -> str:
        """
        Выполняет анализ видео файлов.
        
        Args:
            text_prompt: Текстовый промпт для анализа
            video_files: Путь к видео файлу или список путей
            analysis_type: Тип анализа (description, action_recognition, summary)
            **kwargs: Дополнительные параметры
            
        Returns:
            Ответ модели с анализом видео
        """
        logger.info(f"Начинаем video completion с типом анализа: {analysis_type}")
        
        # Нормализуем входные данные
        if not isinstance(video_files, list):
            video_files = [video_files]
        
        # Загружаем и валидируем видео файлы
        media_files = []
        for video_path in video_files:
            media_file = await self._load_video_file(video_path)
            media_files.append(media_file)
        
        # Создаем мультимодальное сообщение
        content = [{"type": "text", "text": f"{text_prompt}\n\nТип анализа: {analysis_type}"}]
        
        for media_file in media_files:
            content.append({
                "type": "video_url",
                "video_url": {
                    "url": f"data:{media_file.mime_type};base64,{media_file.content}"
                }
            })
        
        messages = [MultimodalMessage(role="user", content=content)]
        
        return await self._multimodal_completion(messages, **kwargs)
    
    async def mixed_media_completion(
        self,
        text_prompt: str,
        media_files: Dict[str, List[Union[str, Path]]],
        **kwargs
    ) -> str:
        """
        Выполняет анализ смешанных медиа файлов.
        
        Args:
            text_prompt: Текстовый промпт
            media_files: Словарь с типами медиа и путями к файлам
                        {"images": [...], "audio": [...], "video": [...]}
            **kwargs: Дополнительные параметры
            
        Returns:
            Ответ модели с анализом всех медиа
        """
        logger.info("Начинаем mixed media completion")
        
        content = [{"type": "text", "text": text_prompt}]
        
        # Обрабатываем изображения
        if "images" in media_files:
            for image_path in media_files["images"]:
                media_file = await self._load_image_file(image_path)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_file.mime_type};base64,{media_file.content}"
                    }
                })
        
        # Обрабатываем аудио
        if "audio" in media_files:
            for audio_path in media_files["audio"]:
                media_file = await self._load_audio_file(audio_path)
                content.append({
                    "type": "audio_url",
                    "audio_url": {
                        "url": f"data:{media_file.mime_type};base64,{media_file.content}"
                    }
                })
        
        # Обрабатываем видео
        if "video" in media_files:
            for video_path in media_files["video"]:
                media_file = await self._load_video_file(video_path)
                content.append({
                    "type": "video_url",
                    "video_url": {
                        "url": f"data:{media_file.mime_type};base64,{media_file.content}"
                    }
                })
        
        messages = [MultimodalMessage(role="user", content=content)]
        
        return await self._multimodal_completion(messages, **kwargs)
    
    async def _load_image_file(self, file_path: Union[str, Path]) -> MediaFile:
        """
        Загружает и валидирует изображение.
        
        Args:
            file_path: Путь к файлу изображения
            
        Returns:
            Объект MediaFile с загруженным изображением
        """
        path = Path(file_path)
        
        if not path.exists():
            raise ValidationError(f"Файл изображения не найден: {file_path}")
        
        # Проверяем размер файла
        file_size = path.stat().st_size
        if file_size > self.multimodal_config.max_image_size:
            raise ValidationError(
                f"Размер изображения ({file_size} байт) превышает максимальный "
                f"({self.multimodal_config.max_image_size} байт)"
            )
        
        # Определяем MIME тип
        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type or not mime_type.startswith('image/'):
            raise ValidationError(f"Неподдерживаемый формат изображения: {path.suffix}")
        
        # Проверяем поддерживаемые форматы
        format_name = path.suffix[1:].lower()
        if format_name not in self.multimodal_config.supported_image_formats:
            raise ValidationError(f"Неподдерживаемый формат изображения: {format_name}")
        
        # Загружаем файл
        try:
            with open(path, 'rb') as f:
                content = base64.b64encode(f.read()).decode('utf-8')
            
            return MediaFile(
                content=content,
                mime_type=mime_type,
                filename=path.name,
                size=file_size
            )
            
        except Exception as e:
            raise ValidationError(f"Ошибка загрузки изображения: {e}")
    
    async def _load_audio_file(self, file_path: Union[str, Path]) -> MediaFile:
        """
        Загружает и валидирует аудио файл.
        
        Args:
            file_path: Путь к аудио файлу
            
        Returns:
            Объект MediaFile с загруженным аудио
        """
        path = Path(file_path)
        
        if not path.exists():
            raise ValidationError(f"Аудио файл не найден: {file_path}")
        
        # Определяем MIME тип
        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type or not mime_type.startswith('audio/'):
            raise ValidationError(f"Неподдерживаемый формат аудио: {path.suffix}")
        
        # Проверяем поддерживаемые форматы
        format_name = path.suffix[1:].lower()
        if format_name not in self.multimodal_config.supported_audio_formats:
            raise ValidationError(f"Неподдерживаемый формат аудио: {format_name}")
        
        # Загружаем файл
        try:
            file_size = path.stat().st_size
            
            with open(path, 'rb') as f:
                content = base64.b64encode(f.read()).decode('utf-8')
            
            return MediaFile(
                content=content,
                mime_type=mime_type,
                filename=path.name,
                size=file_size
            )
            
        except Exception as e:
            raise ValidationError(f"Ошибка загрузки аудио: {e}")
    
    async def _load_video_file(self, file_path: Union[str, Path]) -> MediaFile:
        """
        Загружает и валидирует видео файл.
        
        Args:
            file_path: Путь к видео файлу
            
        Returns:
            Объект MediaFile с загруженным видео
        """
        path = Path(file_path)
        
        if not path.exists():
            raise ValidationError(f"Видео файл не найден: {file_path}")
        
        # Определяем MIME тип
        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type or not mime_type.startswith('video/'):
            raise ValidationError(f"Неподдерживаемый формат видео: {path.suffix}")
        
        # Проверяем поддерживаемые форматы
        format_name = path.suffix[1:].lower()
        if format_name not in self.multimodal_config.supported_video_formats:
            raise ValidationError(f"Неподдерживаемый формат видео: {format_name}")
        
        # Загружаем файл
        try:
            file_size = path.stat().st_size
            
            with open(path, 'rb') as f:
                content = base64.b64encode(f.read()).decode('utf-8')
            
            return MediaFile(
                content=content,
                mime_type=mime_type,
                filename=path.name,
                size=file_size
            )
            
        except Exception as e:
            raise ValidationError(f"Ошибка загрузки видео: {e}")
    
    async def _multimodal_completion(
        self, 
        messages: List[MultimodalMessage], 
        **kwargs
    ) -> str:
        """
        Выполняет мультимодальный запрос к модели.
        
        Args:
            messages: Список мультимодальных сообщений
            **kwargs: Дополнительные параметры
            
        Returns:
            Ответ модели
        """
        try:
            # Конвертируем в формат OpenAI
            openai_messages = []
            for msg in messages:
                openai_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            # Выполняем запрос через базовый клиент
            return await self.chat_completion(openai_messages, **kwargs)
            
        except Exception as e:
            logger.error(f"Ошибка в мультимодальном запросе: {e}")
            raise APIError(f"Ошибка мультимодального запроса: {e}")
    
    # Утилитарные методы
    @staticmethod
    def create_image_url_content(image_path: Union[str, Path], detail: str = "auto") -> Dict[str, Any]:
        """
        Создает контент для изображения в формате OpenAI.
        
        Args:
            image_path: Путь к изображению
            detail: Уровень детализации
            
        Returns:
            Словарь с контентом изображения
        """
        data_url = MediaUtils.create_data_url(image_path)
        
        return {
            "type": "image_url",
            "image_url": {
                "url": data_url,
                "detail": detail
            }
        }
    
    @staticmethod
    def validate_and_process_image(
        image_path: Union[str, Path],
        max_size: int = 20 * 1024 * 1024,
        max_width: int = 2048,
        max_height: int = 2048,
        auto_resize: bool = True
    ) -> Dict[str, Any]:
        """
        Валидирует и при необходимости обрабатывает изображение.
        
        Args:
            image_path: Путь к изображению
            max_size: Максимальный размер файла в байтах
            max_width: Максимальная ширина
            max_height: Максимальная высота
            auto_resize: Автоматически изменять размер если нужно
            
        Returns:
            Результат валидации и обработки
        """
        path = Path(image_path)
        
        # Валидация
        validation_result = MediaUtils.validate_media_file(
            path, 'image', max_size=max_size
        )
        
        if not validation_result['valid']:
            return validation_result
        
        # Проверяем размеры изображения
        try:
            img_info = MediaUtils.get_image_info(path)
            
            needs_resize = (
                img_info['width'] > max_width or 
                img_info['height'] > max_height
            )
            
            if needs_resize and auto_resize:
                # Создаем временный файл для измененного изображения
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=path.suffix, delete=False) as f:
                    temp_path = Path(f.name)
                
                try:
                    resize_result = MediaUtils.resize_image(
                        path, temp_path, max_width, max_height
                    )
                    
                    validation_result['resized'] = True
                    validation_result['resize_info'] = resize_result
                    validation_result['processed_path'] = temp_path
                    
                except Exception as e:
                    validation_result['warnings'].append(f"Не удалось изменить размер: {e}")
                    validation_result['processed_path'] = path
            else:
                validation_result['resized'] = False
                validation_result['processed_path'] = path
                
        except Exception as e:
            validation_result['warnings'].append(f"Не удалось проанализировать изображение: {e}")
            validation_result['processed_path'] = path
        
        return validation_result
    
    @staticmethod
    def batch_validate_media_files(
        file_paths: List[Union[str, Path]],
        media_type: str = 'image',
        max_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Пакетная валидация медиа файлов.
        
        Args:
            file_paths: Список путей к файлам
            media_type: Тип медиа для валидации
            max_size: Максимальный размер файла
            
        Returns:
            Результаты валидации всех файлов
        """
        results = {
            'valid_files': [],
            'invalid_files': [],
            'total_size': 0,
            'summary': {}
        }
        
        for file_path in file_paths:
            try:
                validation = MediaUtils.validate_media_file(
                    file_path, media_type, max_size
                )
                
                file_info = {
                    'path': str(file_path),
                    'validation': validation
                }
                
                if validation['valid']:
                    results['valid_files'].append(file_info)
                    results['total_size'] += validation['file_info']['size']
                else:
                    results['invalid_files'].append(file_info)
                    
            except Exception as e:
                results['invalid_files'].append({
                    'path': str(file_path),
                    'validation': {
                        'valid': False,
                        'errors': [f"Ошибка валидации: {e}"]
                    }
                })
        
        # Сводка
        results['summary'] = {
            'total_files': len(file_paths),
            'valid_count': len(results['valid_files']),
            'invalid_count': len(results['invalid_files']),
            'total_size_mb': results['total_size'] / (1024 * 1024)
        }
        
        return results
    
    @staticmethod
    def get_supported_formats() -> Dict[str, List[str]]:
        """
        Возвращает список поддерживаемых форматов для каждого типа медиа.
        
        Returns:
            Словарь с поддерживаемыми форматами
        """
        return MediaUtils.get_supported_formats()
    
    async def chat_completion_multimodal(
        self,
        messages: List[Dict[str, str]],
        images: Optional[List[Union[str, Path]]] = None,
        audio_files: Optional[List[Union[str, Path]]] = None,
        video_files: Optional[List[Union[str, Path]]] = None,
        **kwargs
    ) -> str:
        """
        Выполняет мультимодальный chat completion.
        
        Args:
            messages: Текстовые сообщения
            images: Список путей к изображениям
            audio_files: Список путей к аудио файлам
            video_files: Список путей к видео файлам
            **kwargs: Дополнительные параметры
            
        Returns:
            Ответ модели
        """
        # Если нет медиа файлов, используем обычный режим
        if not any([images, audio_files, video_files]):
            return await self.chat_completion(messages, **kwargs)
        
        # Подготавливаем медиа файлы
        media_files = []
        
        if images:
            for image_path in images:
                media_file = await self._load_image_file(image_path)
                media_files.append(media_file)
        
        if audio_files:
            for audio_path in audio_files:
                media_file = await self._load_audio_file(audio_path)
                media_files.append(media_file)
        
        if video_files:
            for video_path in video_files:
                media_file = await self._load_video_file(video_path)
                media_files.append(media_file)
        
        # Используем mixed_media_completion для обработки
        text_prompt = messages[-1].get("content", "") if messages else ""
        
        return await self.mixed_media_completion(
            text_prompt=text_prompt,
            media_files=media_files,
            **kwargs
        )
    
    # ASR интеграция
    
    def _get_asr_client(self):
        """Получение ASR клиента с ленивой инициализацией"""
        if self._asr_client is None:
            from .asr import ASRClient, ASRConfig
            asr_config = ASRConfig()
            self._asr_client = ASRClient(self.config, asr_config)
        return self._asr_client
    
    async def speech_to_text_advanced(
        self,
        audio_files: Union[str, Path, List[Union[str, Path]]],
        language: Optional[str] = None,
        include_diarization: bool = True,
        include_emotions: bool = True,
        include_vad: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Продвинутое распознавание речи с полным анализом.
        
        Args:
            audio_files: Путь к аудио файлу или список путей
            language: Язык для распознавания
            include_diarization: Включить диаризацию спикеров
            include_emotions: Включить анализ эмоций
            include_vad: Включить детекцию речевой активности
            **kwargs: Дополнительные параметры
            
        Returns:
            Результат продвинутого распознавания речи
        """
        logger.info("Начинаем продвинутое распознавание речи")
        
        # Нормализуем входные данные
        if not isinstance(audio_files, list):
            audio_files = [audio_files]
        
        asr_client = self._get_asr_client()
        results = []
        
        for audio_file in audio_files:
            try:
                result = await asr_client.transcribe_with_analysis(
                    audio_file,
                    include_diarization=include_diarization,
                    include_emotions=include_emotions,
                    include_vad=include_vad,
                    language=language,
                    **kwargs
                )
                
                result['file_path'] = str(audio_file)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Ошибка обработки {audio_file}: {e}")
                results.append({
                    'file_path': str(audio_file),
                    'error': str(e),
                    'success': False
                })
        
        # Если один файл, возвращаем результат напрямую
        if len(results) == 1:
            return results[0]
        
        # Для нескольких файлов возвращаем сводку
        return {
            'files_processed': len(results),
            'successful': len([r for r in results if 'error' not in r]),
            'failed': len([r for r in results if 'error' in r]),
            'results': results
        }
    
    async def text_to_speech_multimodal(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        output_file: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Union[bytes, Path]:
        """
        Синтез речи из текста через мультимодальный клиент.
        
        Args:
            text: Текст для синтеза
            voice: Голос для синтеза
            language: Язык синтеза
            output_file: Путь для сохранения аудио файла
            **kwargs: Дополнительные параметры
            
        Returns:
            Аудио данные или путь к файлу
        """
        logger.info("Начинаем синтез речи через мультимодальный клиент")
        
        asr_client = self._get_asr_client()
        return await asr_client.text_to_speech(
            text=text,
            voice=voice,
            language=language,
            output_file=output_file,
            **kwargs
        )
    
    async def voice_activity_detection_multimodal(
        self,
        audio_files: Union[str, Path, List[Union[str, Path]]],
        **kwargs
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Детекция речевой активности через мультимодальный клиент.
        
        Args:
            audio_files: Путь к аудио файлу или список путей
            **kwargs: Дополнительные параметры
            
        Returns:
            Результат VAD анализа
        """
        logger.info("Начинаем VAD анализ через мультимодальный клиент")
        
        # Нормализуем входные данные
        if not isinstance(audio_files, list):
            audio_files = [audio_files]
        
        asr_client = self._get_asr_client()
        results = []
        
        for audio_file in audio_files:
            try:
                vad_result = await asr_client.voice_activity_detection(audio_file, **kwargs)
                results.append({
                    'file_path': str(audio_file),
                    'vad_result': vad_result,
                    'success': True
                })
            except Exception as e:
                logger.error(f"Ошибка VAD анализа {audio_file}: {e}")
                results.append({
                    'file_path': str(audio_file),
                    'error': str(e),
                    'success': False
                })
        
        # Если один файл, возвращаем результат напрямую
        if len(results) == 1:
            return results[0]
        
        return results
    
    async def speaker_diarization_multimodal(
        self,
        audio_files: Union[str, Path, List[Union[str, Path]]],
        num_speakers: Optional[int] = None,
        **kwargs
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Диаризация спикеров через мультимодальный клиент.
        
        Args:
            audio_files: Путь к аудио файлу или список путей
            num_speakers: Ожидаемое количество спикеров
            **kwargs: Дополнительные параметры
            
        Returns:
            Результат диаризации спикеров
        """
        logger.info("Начинаем диаризацию спикеров через мультимодальный клиент")
        
        # Нормализуем входные данные
        if not isinstance(audio_files, list):
            audio_files = [audio_files]
        
        asr_client = self._get_asr_client()
        results = []
        
        for audio_file in audio_files:
            try:
                diarization_result = await asr_client.speaker_diarization(
                    audio_file, 
                    num_speakers=num_speakers, 
                    **kwargs
                )
                results.append({
                    'file_path': str(audio_file),
                    'diarization_result': diarization_result,
                    'success': True
                })
            except Exception as e:
                logger.error(f"Ошибка диаризации {audio_file}: {e}")
                results.append({
                    'file_path': str(audio_file),
                    'error': str(e),
                    'success': False
                })
        
        # Если один файл, возвращаем результат напрямую
        if len(results) == 1:
            return results[0]
        
        return results
    
    async def multimodal_with_speech_analysis(
        self,
        text_prompt: str,
        media_files: Dict[str, List[Union[str, Path]]],
        speech_analysis_options: Optional[Dict[str, bool]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Комплексный мультимодальный анализ с продвинутой обработкой речи.
        
        Args:
            text_prompt: Текстовый промпт
            media_files: Словарь с типами медиа и путями к файлам
            speech_analysis_options: Опции анализа речи
            **kwargs: Дополнительные параметры
            
        Returns:
            Комплексный результат анализа
        """
        logger.info("Начинаем комплексный мультимодальный анализ с речью")
        
        # Опции анализа речи по умолчанию
        speech_options = speech_analysis_options or {
            'include_diarization': True,
            'include_emotions': True,
            'include_vad': True
        }
        
        results = {
            'text_prompt': text_prompt,
            'media_analysis': {},
            'speech_analysis': {},
            'combined_insights': {}
        }
        
        # Обрабатываем изображения и видео как обычно
        visual_media = {}
        if 'images' in media_files:
            visual_media['images'] = media_files['images']
        if 'video' in media_files:
            visual_media['video'] = media_files['video']
        
        if visual_media:
            try:
                visual_result = await self.mixed_media_completion(
                    text_prompt, visual_media, **kwargs
                )
                results['media_analysis']['visual'] = visual_result
            except Exception as e:
                logger.error(f"Ошибка анализа визуальных медиа: {e}")
                results['media_analysis']['visual'] = {'error': str(e)}
        
        # Продвинутая обработка аудио
        if 'audio' in media_files:
            try:
                speech_results = []
                for audio_file in media_files['audio']:
                    speech_result = await self.speech_to_text_advanced(
                        audio_file,
                        **speech_options,
                        **kwargs
                    )
                    speech_results.append(speech_result)
                
                results['speech_analysis'] = {
                    'files_count': len(speech_results),
                    'results': speech_results
                }
                
                # Генерируем комбинированные инсайты
                results['combined_insights'] = self._generate_combined_insights(
                    results['media_analysis'],
                    results['speech_analysis'],
                    text_prompt
                )
                
            except Exception as e:
                logger.error(f"Ошибка анализа речи: {e}")
                results['speech_analysis'] = {'error': str(e)}
        
        return results
    
    def _generate_combined_insights(
        self,
        media_analysis: Dict[str, Any],
        speech_analysis: Dict[str, Any],
        text_prompt: str
    ) -> Dict[str, Any]:
        """Генерация комбинированных инсайтов из всех типов анализа"""
        
        insights = {
            'summary': '',
            'key_findings': [],
            'recommendations': [],
            'confidence_scores': {}
        }
        
        # Анализируем речевые данные
        if 'results' in speech_analysis:
            speech_results = speech_analysis['results']
            
            # Собираем статистику по спикерам
            total_speakers = 0
            dominant_emotions = []
            avg_confidence = 0.0
            
            for result in speech_results:
                if 'transcription' in result and 'speakers' in result['transcription']:
                    total_speakers += len(result['transcription']['speakers'])
                
                if 'emotions' in result and result['emotions']:
                    dominant_emotions.append(result['emotions'].dominant_emotion)
                
                if 'transcription' in result:
                    avg_confidence += result['transcription'].get('confidence', 0.0)
            
            if speech_results:
                avg_confidence /= len(speech_results)
            
            insights['key_findings'].extend([
                f"Обнаружено {total_speakers} уникальных спикеров",
                f"Средняя уверенность распознавания: {avg_confidence:.2f}",
                f"Доминирующие эмоции: {', '.join(set(dominant_emotions))}"
            ])
            
            insights['confidence_scores']['speech_recognition'] = avg_confidence
        
        # Анализируем визуальные данные
        if 'visual' in media_analysis and 'error' not in media_analysis['visual']:
            insights['key_findings'].append("Визуальный контент успешно проанализирован")
            insights['confidence_scores']['visual_analysis'] = 0.8  # Примерная оценка
        
        # Генерируем сводку
        findings_count = len(insights['key_findings'])
        insights['summary'] = f"Проанализировано {findings_count} ключевых аспектов мультимодального контента"
        
        # Рекомендации
        if avg_confidence < 0.7:
            insights['recommendations'].append("Рекомендуется улучшить качество аудио для лучшего распознавания")
        
        if total_speakers > 5:
            insights['recommendations'].append("Большое количество спикеров может усложнить анализ")
        
        return insights

    # Методы базового класса (делегирование к стандартному клиенту)
    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Базовый chat completion через стандартный клиент"""
        from .standard import StandardLLMClient
        standard_client = StandardLLMClient(self.config)
        return await standard_client.chat_completion(messages, **kwargs)
    
    async def chat_completion_stream(self, messages: List[Dict[str, str]], **kwargs):
        """Streaming не поддерживается в мультимодальном режиме"""
        raise NotImplementedError(
            "MultimodalLLMClient не поддерживает streaming режим. "
            "Используйте обычные методы completion."
        )
        # Нужно добавить yield для правильной сигнатуры async generator
        yield  # pragma: no cover
    
    async def chat_completion_structured(self, messages: List[Dict[str, str]], response_model, **kwargs):
        """Structured output не поддерживается в мультимодальном режиме"""
        raise NotImplementedError(
            "MultimodalLLMClient не поддерживает structured output. "
            "Используйте StructuredLLMClient для структурированных ответов."
        )