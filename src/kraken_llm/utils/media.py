"""
Утилиты для работы с медиа файлами в Kraken LLM фреймворке.

Этот модуль предоставляет функции для:
- Кодирования медиа файлов в base64
- Валидации форматов и размеров файлов
- Обработки и оптимизации медиа контента
- Извлечения метаданных из файлов
"""

import base64
import mimetypes
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
from PIL import Image, ImageOps
import logging

logger = logging.getLogger(__name__)


class MediaUtils:
    """Утилиты для работы с медиа файлами"""
    
    # Поддерживаемые форматы
    SUPPORTED_IMAGE_FORMATS = {
        'jpeg', 'jpg', 'png', 'gif', 'webp', 'bmp', 'tiff', 'tif'
    }
    
    SUPPORTED_AUDIO_FORMATS = {
        'mp3', 'wav', 'ogg', 'm4a', 'webm', 'flac', 'aac'
    }
    
    SUPPORTED_VIDEO_FORMATS = {
        'mp4', 'webm', 'ogg', 'avi', 'mov', 'mkv', 'wmv', 'flv'
    }
    
    # MIME типы
    MIME_TYPE_MAPPING = {
        # Изображения
        'jpeg': 'image/jpeg',
        'jpg': 'image/jpeg',
        'png': 'image/png',
        'gif': 'image/gif',
        'webp': 'image/webp',
        'bmp': 'image/bmp',
        'tiff': 'image/tiff',
        'tif': 'image/tiff',
        
        # Аудио
        'mp3': 'audio/mpeg',
        'wav': 'audio/wav',
        'ogg': 'audio/ogg',
        'm4a': 'audio/mp4',
        'webm': 'audio/webm',
        'flac': 'audio/flac',
        'aac': 'audio/aac',
        
        # Видео
        'mp4': 'video/mp4',
        'webm': 'video/webm',
        'avi': 'video/avi',
        'mov': 'video/quicktime',
        'mkv': 'video/x-matroska',
        'wmv': 'video/x-ms-wmv',
        'flv': 'video/x-flv'
    }
    
    @staticmethod
    def encode_file_to_base64(file_path: Union[str, Path]) -> str:
        """
        Кодирует файл в base64.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            Строка base64
            
        Raises:
            FileNotFoundError: Если файл не найден
            IOError: Если ошибка чтения файла
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        
        try:
            with open(path, 'rb') as f:
                content = f.read()
            return base64.b64encode(content).decode('utf-8')
        except Exception as e:
            raise IOError(f"Ошибка чтения файла {file_path}: {e}")
    
    @staticmethod
    def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Получает информацию о файле.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            Словарь с информацией о файле
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        
        # Базовая информация
        stat = path.stat()
        file_info = {
            'filename': path.name,
            'size': stat.st_size,
            'extension': path.suffix[1:].lower() if path.suffix else '',
            'path': str(path.absolute())
        }
        
        # MIME тип
        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type:
            # Попробуем определить по расширению
            ext = file_info['extension']
            mime_type = MediaUtils.MIME_TYPE_MAPPING.get(ext)
        
        file_info['mime_type'] = mime_type
        
        # Определяем тип медиа
        if mime_type:
            if mime_type.startswith('image/'):
                file_info['media_type'] = 'image'
            elif mime_type.startswith('audio/'):
                file_info['media_type'] = 'audio'
            elif mime_type.startswith('video/'):
                file_info['media_type'] = 'video'
            else:
                file_info['media_type'] = 'unknown'
        else:
            file_info['media_type'] = 'unknown'
        
        return file_info
    
    @staticmethod
    def validate_file_format(file_path: Union[str, Path], allowed_formats: Optional[List[str]] = None) -> bool:
        """
        Валидирует формат файла.
        
        Args:
            file_path: Путь к файлу
            allowed_formats: Список разрешенных форматов (расширений)
            
        Returns:
            True если формат поддерживается
        """
        path = Path(file_path)
        extension = path.suffix[1:].lower() if path.suffix else ''
        
        if allowed_formats:
            return extension in [fmt.lower() for fmt in allowed_formats]
        
        # Проверяем против всех поддерживаемых форматов
        all_formats = (
            MediaUtils.SUPPORTED_IMAGE_FORMATS |
            MediaUtils.SUPPORTED_AUDIO_FORMATS |
            MediaUtils.SUPPORTED_VIDEO_FORMATS
        )
        
        return extension in all_formats
    
    @staticmethod
    def validate_file_size(file_path: Union[str, Path], max_size: int) -> bool:
        """
        Валидирует размер файла.
        
        Args:
            file_path: Путь к файлу
            max_size: Максимальный размер в байтах
            
        Returns:
            True если размер допустимый
        """
        path = Path(file_path)
        
        if not path.exists():
            return False
        
        return path.stat().st_size <= max_size
    
    @staticmethod
    def get_image_info(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Получает детальную информацию об изображении.
        
        Args:
            file_path: Путь к изображению
            
        Returns:
            Словарь с информацией об изображении
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Изображение не найдено: {file_path}")
        
        try:
            with Image.open(path) as img:
                info = {
                    'width': img.width,
                    'height': img.height,
                    'format': img.format,
                    'mode': img.mode,
                    'has_transparency': img.mode in ('RGBA', 'LA') or 'transparency' in img.info,
                    'size_bytes': path.stat().st_size
                }
                
                # Дополнительная информация если доступна
                if hasattr(img, 'info') and img.info:
                    info['dpi'] = img.info.get('dpi', (72, 72))
                    info['exif'] = bool(img.info.get('exif'))
                
                return info
                
        except Exception as e:
            raise ValueError(f"Ошибка анализа изображения {file_path}: {e}")
    
    @staticmethod
    def resize_image(
        file_path: Union[str, Path],
        output_path: Union[str, Path],
        max_width: int = 1024,
        max_height: int = 1024,
        quality: int = 85,
        preserve_aspect_ratio: bool = True
    ) -> Dict[str, Any]:
        """
        Изменяет размер изображения.
        
        Args:
            file_path: Путь к исходному изображению
            output_path: Путь для сохранения
            max_width: Максимальная ширина
            max_height: Максимальная высота
            quality: Качество JPEG (1-100)
            preserve_aspect_ratio: Сохранять пропорции
            
        Returns:
            Информация о результате
        """
        input_path = Path(file_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Изображение не найдено: {file_path}")
        
        try:
            with Image.open(input_path) as img:
                original_size = img.size
                
                # Вычисляем новый размер
                if preserve_aspect_ratio:
                    img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                    new_size = img.size
                else:
                    img = img.resize((max_width, max_height), Image.Resampling.LANCZOS)
                    new_size = (max_width, max_height)
                
                # Сохраняем
                save_kwargs = {}
                if output_path.suffix.lower() in ['.jpg', '.jpeg']:
                    save_kwargs['quality'] = quality
                    save_kwargs['optimize'] = True
                    # Конвертируем в RGB если есть прозрачность
                    if img.mode in ('RGBA', 'LA'):
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                        img = background
                
                img.save(output_path, **save_kwargs)
                
                # Информация о результате
                result = {
                    'original_size': original_size,
                    'new_size': new_size,
                    'original_file_size': input_path.stat().st_size,
                    'new_file_size': output_path.stat().st_size,
                    'compression_ratio': input_path.stat().st_size / output_path.stat().st_size
                }
                
                logger.info(f"Изображение изменено: {original_size} -> {new_size}")
                return result
                
        except Exception as e:
            raise ValueError(f"Ошибка изменения размера изображения: {e}")
    
    @staticmethod
    def create_data_url(file_path: Union[str, Path], mime_type: Optional[str] = None) -> str:
        """
        Создает data URL для файла.
        
        Args:
            file_path: Путь к файлу
            mime_type: MIME тип (определяется автоматически если не указан)
            
        Returns:
            Data URL строка
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        
        # Определяем MIME тип
        if not mime_type:
            mime_type, _ = mimetypes.guess_type(str(path))
            if not mime_type:
                extension = path.suffix[1:].lower()
                mime_type = MediaUtils.MIME_TYPE_MAPPING.get(extension, 'application/octet-stream')
        
        # Кодируем в base64
        base64_content = MediaUtils.encode_file_to_base64(path)
        
        return f"data:{mime_type};base64,{base64_content}"
    
    @staticmethod
    def batch_process_images(
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        max_width: int = 1024,
        max_height: int = 1024,
        quality: int = 85,
        formats: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Пакетная обработка изображений.
        
        Args:
            input_dir: Директория с исходными изображениями
            output_dir: Директория для сохранения
            max_width: Максимальная ширина
            max_height: Максимальная высота
            quality: Качество JPEG
            formats: Список форматов для обработки
            
        Returns:
            Список результатов обработки
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Входная директория не найдена: {input_dir}")
        
        # Создаем выходную директорию
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Форматы по умолчанию
        if not formats:
            formats = ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp']
        
        results = []
        
        # Обрабатываем все изображения
        for file_path in input_path.iterdir():
            if file_path.is_file() and file_path.suffix[1:].lower() in formats:
                try:
                    output_file = output_path / file_path.name
                    
                    result = MediaUtils.resize_image(
                        file_path,
                        output_file,
                        max_width,
                        max_height,
                        quality
                    )
                    
                    result['input_file'] = str(file_path)
                    result['output_file'] = str(output_file)
                    result['status'] = 'success'
                    
                    results.append(result)
                    
                except Exception as e:
                    results.append({
                        'input_file': str(file_path),
                        'status': 'error',
                        'error': str(e)
                    })
                    logger.error(f"Ошибка обработки {file_path}: {e}")
        
        logger.info(f"Обработано {len(results)} файлов")
        return results
    
    @staticmethod
    def get_supported_formats() -> Dict[str, List[str]]:
        """
        Возвращает список поддерживаемых форматов.
        
        Returns:
            Словарь с форматами по типам медиа
        """
        return {
            'images': sorted(list(MediaUtils.SUPPORTED_IMAGE_FORMATS)),
            'audio': sorted(list(MediaUtils.SUPPORTED_AUDIO_FORMATS)),
            'video': sorted(list(MediaUtils.SUPPORTED_VIDEO_FORMATS))
        }
    
    @staticmethod
    def validate_media_file(
        file_path: Union[str, Path],
        media_type: str,
        max_size: Optional[int] = None,
        allowed_formats: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Комплексная валидация медиа файла.
        
        Args:
            file_path: Путь к файлу
            media_type: Тип медиа ('image', 'audio', 'video')
            max_size: Максимальный размер в байтах
            allowed_formats: Разрешенные форматы
            
        Returns:
            Результат валидации
        """
        path = Path(file_path)
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'file_info': {}
        }
        
        try:
            # Проверяем существование файла
            if not path.exists():
                result['valid'] = False
                result['errors'].append(f"Файл не найден: {file_path}")
                return result
            
            # Получаем информацию о файле
            file_info = MediaUtils.get_file_info(path)
            result['file_info'] = file_info
            
            # Проверяем тип медиа
            if file_info['media_type'] != media_type:
                result['valid'] = False
                result['errors'].append(
                    f"Неверный тип медиа: ожидался {media_type}, получен {file_info['media_type']}"
                )
            
            # Проверяем формат
            if allowed_formats and not MediaUtils.validate_file_format(path, allowed_formats):
                result['valid'] = False
                result['errors'].append(
                    f"Неподдерживаемый формат: {file_info['extension']}"
                )
            
            # Проверяем размер
            if max_size and not MediaUtils.validate_file_size(path, max_size):
                result['valid'] = False
                result['errors'].append(
                    f"Файл слишком большой: {file_info['size']} байт (максимум {max_size})"
                )
            
            # Дополнительные проверки для изображений
            if media_type == 'image' and result['valid']:
                try:
                    img_info = MediaUtils.get_image_info(path)
                    result['file_info'].update(img_info)
                    
                    # Предупреждения
                    if img_info['width'] > 4096 or img_info['height'] > 4096:
                        result['warnings'].append("Очень большое разрешение изображения")
                    
                    if img_info['size_bytes'] > 10 * 1024 * 1024:  # 10MB
                        result['warnings'].append("Большой размер файла изображения")
                        
                except Exception as e:
                    result['warnings'].append(f"Не удалось проанализировать изображение: {e}")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Ошибка валидации: {e}")
        
        return result


# Удобные функции для быстрого использования

def encode_image_to_base64(image_path: Union[str, Path]) -> str:
    """Быстрое кодирование изображения в base64"""
    return MediaUtils.encode_file_to_base64(image_path)


def create_image_data_url(image_path: Union[str, Path]) -> str:
    """Быстрое создание data URL для изображения"""
    return MediaUtils.create_data_url(image_path)


def validate_image(image_path: Union[str, Path], max_size: int = 20 * 1024 * 1024) -> bool:
    """Быстрая валидация изображения"""
    result = MediaUtils.validate_media_file(image_path, 'image', max_size)
    return result['valid']


def get_image_dimensions(image_path: Union[str, Path]) -> Tuple[int, int]:
    """Получение размеров изображения"""
    info = MediaUtils.get_image_info(image_path)
    return info['width'], info['height']