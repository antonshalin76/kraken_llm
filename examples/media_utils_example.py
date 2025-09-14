#!/usr/bin/env python3
"""
Пример использования утилит для работы с медиа файлами.

Демонстрирует различные возможности MediaUtils:
- Кодирование файлов в base64
- Валидация форматов и размеров
- Изменение размера изображений
- Пакетная обработка
- Создание data URLs
"""

import tempfile
from pathlib import Path
from PIL import Image, ImageDraw
import logging

from kraken_llm.utils.media import (
    MediaUtils,
    encode_image_to_base64,
    create_image_data_url,
    validate_image,
    get_image_dimensions
)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_images(output_dir: Path) -> list[Path]:
    """Создает набор тестовых изображений"""
    output_dir.mkdir(exist_ok=True)
    image_paths = []
    
    # Создаем изображения разных размеров и форматов
    configs = [
        ("small_red.png", (100, 100), "red", "PNG"),
        ("medium_green.jpg", (500, 300), "green", "JPEG"),
        ("large_blue.png", (1200, 800), "blue", "PNG"),
        ("tiny_yellow.gif", (50, 50), "yellow", "GIF"),
    ]
    
    for filename, size, color, format_name in configs:
        img = Image.new('RGB', size, color=color)
        
        # Добавляем простые фигуры для интереса
        draw = ImageDraw.Draw(img)
        draw.rectangle([10, 10, size[0]-10, size[1]-10], outline='black', width=2)
        draw.ellipse([size[0]//4, size[1]//4, 3*size[0]//4, 3*size[1]//4], 
                    outline='white', width=3)
        
        file_path = output_dir / filename
        
        save_kwargs = {}
        if format_name == "JPEG":
            save_kwargs['quality'] = 85
        
        img.save(file_path, format=format_name, **save_kwargs)
        image_paths.append(file_path)
        
        logger.info(f"Создано изображение: {filename} ({size[0]}x{size[1]})")
    
    return image_paths


def demo_file_info():
    """Демонстрация получения информации о файлах"""
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦИЯ ПОЛУЧЕНИЯ ИНФОРМАЦИИ О ФАЙЛАХ")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        image_paths = create_sample_images(temp_path)
        
        for image_path in image_paths:
            print(f"\nАнализ файла: {image_path.name}")
            print("-" * 40)
            
            # Базовая информация о файле
            file_info = MediaUtils.get_file_info(image_path)
            print(f"Размер файла: {file_info['size']:,} байт")
            print(f"MIME тип: {file_info['mime_type']}")
            print(f"Тип медиа: {file_info['media_type']}")
            print(f"Расширение: {file_info['extension']}")
            
            # Детальная информация об изображении
            if file_info['media_type'] == 'image':
                img_info = MediaUtils.get_image_info(image_path)
                print(f"Разрешение: {img_info['width']}x{img_info['height']}")
                print(f"Формат: {img_info['format']}")
                print(f"Цветовой режим: {img_info['mode']}")
                print(f"Прозрачность: {'Да' if img_info['has_transparency'] else 'Нет'}")


def demo_validation():
    """Демонстрация валидации файлов"""
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦИЯ ВАЛИДАЦИИ ФАЙЛОВ")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        image_paths = create_sample_images(temp_path)
        
        # Различные сценарии валидации
        validation_scenarios = [
            {
                'name': 'Валидация размера файла (лимит 1KB)',
                'max_size': 1024,
                'allowed_formats': None
            },
            {
                'name': 'Валидация формата (только PNG)',
                'max_size': None,
                'allowed_formats': ['png']
            },
            {
                'name': 'Строгая валидация (PNG, до 100KB)',
                'max_size': 100 * 1024,
                'allowed_formats': ['png']
            }
        ]
        
        for scenario in validation_scenarios:
            print(f"\n{scenario['name']}:")
            print("-" * 50)
            
            for image_path in image_paths:
                result = MediaUtils.validate_media_file(
                    image_path,
                    'image',
                    max_size=scenario['max_size'],
                    allowed_formats=scenario['allowed_formats']
                )
                
                status = "✓ ВАЛИДЕН" if result['valid'] else "✗ НЕВАЛИДЕН"
                print(f"{image_path.name:20} {status}")
                
                if result['errors']:
                    for error in result['errors']:
                        print(f"    Ошибка: {error}")
                
                if result['warnings']:
                    for warning in result['warnings']:
                        print(f"    Предупреждение: {warning}")


def demo_image_processing():
    """Демонстрация обработки изображений"""
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦИЯ ОБРАБОТКИ ИЗОБРАЖЕНИЙ")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir = temp_path / "input"
        output_dir = temp_path / "output"
        
        # Создаем исходные изображения
        image_paths = create_sample_images(input_dir)
        
        print(f"\nСоздано {len(image_paths)} исходных изображений")
        
        # Индивидуальное изменение размера
        print("\n1. Индивидуальное изменение размера:")
        print("-" * 40)
        
        for image_path in image_paths[:2]:  # Обрабатываем первые 2
            output_path = output_dir / f"resized_{image_path.name}"
            output_dir.mkdir(exist_ok=True)
            
            try:
                result = MediaUtils.resize_image(
                    image_path,
                    output_path,
                    max_width=300,
                    max_height=300,
                    quality=80
                )
                
                print(f"{image_path.name}:")
                print(f"  Исходный размер: {result['original_size']}")
                print(f"  Новый размер: {result['new_size']}")
                print(f"  Сжатие: {result['compression_ratio']:.2f}x")
                print(f"  Размер файла: {result['original_file_size']} -> {result['new_file_size']} байт")
                
            except Exception as e:
                print(f"Ошибка обработки {image_path.name}: {e}")
        
        # Пакетная обработка
        print("\n2. Пакетная обработка:")
        print("-" * 40)
        
        batch_output_dir = temp_path / "batch_output"
        
        results = MediaUtils.batch_process_images(
            input_dir,
            batch_output_dir,
            max_width=200,
            max_height=200,
            quality=75,
            formats=['png', 'jpg', 'gif']
        )
        
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'error']
        
        print(f"Успешно обработано: {len(successful)} файлов")
        print(f"Ошибок: {len(failed)} файлов")
        
        for result in successful:
            input_file = Path(result['input_file']).name
            compression = result.get('compression_ratio', 1)
            print(f"  {input_file}: сжатие {compression:.2f}x")


def demo_encoding_and_urls():
    """Демонстрация кодирования и создания URLs"""
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦИЯ КОДИРОВАНИЯ И DATA URLs")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Создаем небольшое изображение для демонстрации
        img = Image.new('RGB', (64, 64), color='purple')
        draw = ImageDraw.Draw(img)
        draw.text((10, 25), "TEST", fill='white')
        
        image_path = temp_path / "test_encoding.png"
        img.save(image_path, format='PNG')
        
        print(f"\nРабота с файлом: {image_path.name}")
        print(f"Размер файла: {image_path.stat().st_size} байт")
        
        # Base64 кодирование
        print("\n1. Base64 кодирование:")
        print("-" * 30)
        
        base64_content = MediaUtils.encode_file_to_base64(image_path)
        print(f"Длина base64: {len(base64_content)} символов")
        print(f"Начало: {base64_content[:50]}...")
        print(f"Конец: ...{base64_content[-20:]}")
        
        # Data URL
        print("\n2. Data URL:")
        print("-" * 30)
        
        data_url = MediaUtils.create_data_url(image_path)
        print(f"Длина data URL: {len(data_url)} символов")
        print(f"Начало: {data_url[:80]}...")
        
        # Удобные функции
        print("\n3. Удобные функции:")
        print("-" * 30)
        
        # Быстрое кодирование
        quick_base64 = encode_image_to_base64(image_path)
        print(f"Быстрое base64 кодирование: {len(quick_base64)} символов")
        
        # Быстрый data URL
        quick_data_url = create_image_data_url(image_path)
        print(f"Быстрый data URL: {len(quick_data_url)} символов")
        
        # Быстрая валидация
        is_valid = validate_image(image_path)
        print(f"Быстрая валидация: {'✓ Валиден' if is_valid else '✗ Невалиден'}")
        
        # Размеры изображения
        width, height = get_image_dimensions(image_path)
        print(f"Размеры изображения: {width}x{height}")


def demo_supported_formats():
    """Демонстрация поддерживаемых форматов"""
    print("\n" + "="*60)
    print("ПОДДЕРЖИВАЕМЫЕ ФОРМАТЫ МЕДИА")
    print("="*60)
    
    formats = MediaUtils.get_supported_formats()
    
    for media_type, format_list in formats.items():
        print(f"\n{media_type.upper()}:")
        print("-" * 20)
        
        # Группируем по 8 форматов в строке
        for i in range(0, len(format_list), 8):
            group = format_list[i:i+8]
            print("  " + ", ".join(f"{fmt:>4}" for fmt in group))
    
    print(f"\nВсего поддерживается:")
    for media_type, format_list in formats.items():
        print(f"  {media_type}: {len(format_list)} форматов")


def main():
    """Главная функция демонстрации"""
    print("ДЕМОНСТРАЦИЯ УТИЛИТ ДЛЯ РАБОТЫ С МЕДИА")
    print("=" * 80)
    print("Этот пример показывает возможности MediaUtils для обработки")
    print("изображений, аудио и видео файлов в Kraken LLM фреймворке.")
    
    try:
        demo_supported_formats()
        demo_file_info()
        demo_validation()
        demo_image_processing()
        demo_encoding_and_urls()
        
        print("\n" + "="*80)
        print("ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА УСПЕШНО!")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\nДемонстрация прервана пользователем")
    except Exception as e:
        print(f"\nОшибка во время демонстрации: {e}")
        logger.exception("Подробности ошибки:")


if __name__ == "__main__":
    main()