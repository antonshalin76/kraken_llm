#!/usr/bin/env python3
"""
Пример использования MultimodalLLMClient для работы с изображениями, аудио и видео.

Демонстрирует различные возможности мультимодального клиента:
- Анализ изображений (vision models)
- Обработка аудио файлов
- Анализ видео контента
- Смешанные мультимодальные запросы
"""

import asyncio
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw
import logging
import os
from dotenv import load_dotenv

from kraken_llm.client.multimodal import MultimodalLLMClient, MultimodalConfig
from kraken_llm.config.settings import LLMConfig

load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_image(filename: str, size: tuple = (300, 200), color: str = 'lightblue') -> Path:
    """
    Создает тестовое изображение с простыми фигурами и текстом.
    
    Args:
        filename: Имя файла
        size: Размер изображения (ширина, высота)
        color: Цвет фона
        
    Returns:
        Путь к созданному файлу
    """
    img = Image.new('RGB', size, color=color)
    draw = ImageDraw.Draw(img)
    
    # Рисуем простые фигуры
    draw.rectangle([50, 50, 150, 100], fill='red', outline='black', width=2)
    draw.ellipse([200, 50, 280, 130], fill='green', outline='black', width=2)
    draw.polygon([(100, 150), (150, 120), (200, 150), (175, 180), (125, 180)], 
                fill='yellow', outline='black')
    
    # Добавляем текст
    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
        draw.text((50, 160), "Тестовое изображение", fill='black', font=font)
        draw.text((50, 175), "Прямоугольник, круг, звезда", fill='darkblue', font=font)
    except:
        draw.text((50, 160), "Test Image", fill='black')
    
    # Сохраняем файл
    temp_path = Path(tempfile.gettempdir()) / filename
    img.save(temp_path, format='PNG')
    
    return temp_path


def create_test_audio(filename: str) -> Path:
    """
    Создает минимальный тестовый WAV файл.
    
    Args:
        filename: Имя файла
        
    Returns:
        Путь к созданному файлу
    """
    # Создаем простой WAV файл с тишиной
    wav_header = (
        b'RIFF'
        b'\x28\x00\x00\x00'  # Размер файла - 8
        b'WAVE'
        b'fmt '
        b'\x10\x00\x00\x00'  # Размер fmt chunk
        b'\x01\x00'          # Аудио формат (PCM)
        b'\x01\x00'          # Количество каналов (моно)
        b'\x44\xac\x00\x00'  # Частота дискретизации (44100 Hz)
        b'\x88\x58\x01\x00'  # Байт в секунду
        b'\x02\x00'          # Выравнивание блока
        b'\x10\x00'          # Бит на семпл (16 бит)
        b'data'
        b'\x04\x00\x00\x00'  # Размер данных
        b'\x00\x00\x00\x00'  # Данные (тишина)
    )
    
    temp_path = Path(tempfile.gettempdir()) / filename
    with open(temp_path, 'wb') as f:
        f.write(wav_header)
    
    return temp_path


async def demo_vision_analysis():
    """Демонстрация анализа изображений"""
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦИЯ АНАЛИЗА ИЗОБРАЖЕНИЙ")
    print("="*60)
    
    # Создаем конфигурацию
    config = LLMConfig(
        endpoint=os.getenv("MULTIMODAL_ENDPOINT"),
        api_key=os.getenv("MULTIMODAL_TOKEN"),
        model=os.getenv("MULTIMODAL_MODEL"),
        temperature=0.7,
        max_tokens=500
    )
    
    multimodal_config = MultimodalConfig(
        max_image_size=5 * 1024 * 1024,  # 5MB
        auto_resize_images=True
    )
    
    # Создаем клиент
    client = MultimodalLLMClient(config, multimodal_config)
    
    # Создаем тестовые изображения
    image1 = create_test_image("test_shapes.png", color='lightblue')
    image2 = create_test_image("test_shapes2.png", size=(400, 300), color='lightgreen')
    
    try:
        # 1. Простой анализ одного изображения
        print("\n1. Анализ одного изображения:")
        print("-" * 40)
        
        result = await client.vision_completion(
            text_prompt="Опиши это изображение детально на русском языке. Какие фигуры и цвета ты видишь?",
            images=image1,
            detail_level="high"
        )
        
        print(f"Результат: {result}")
        
        # 2. Сравнение двух изображений
        print("\n2. Сравнение двух изображений:")
        print("-" * 40)
        
        result = await client.vision_completion(
            text_prompt="Сравни эти два изображения. В чем их сходства и различия?",
            images=[image1, image2]
        )
        
        print(f"Результат: {result}")
        
        # 3. Специфический анализ
        print("\n3. Подсчет объектов на изображении:")
        print("-" * 40)
        
        result = await client.vision_completion(
            text_prompt="Сколько геометрических фигур ты видишь на этом изображении? Перечисли их по типам.",
            images=image1,
            detail_level="high"
        )
        
        print(f"Результат: {result}")
        
    except Exception as e:
        print(f"Ошибка при анализе изображений: {e}")
        print("Возможно, LLM сервер не поддерживает vision модели или недоступен")
    
    finally:
        # Очистка временных файлов
        for img_path in [image1, image2]:
            if img_path.exists():
                img_path.unlink()


async def demo_audio_processing():
    """Демонстрация обработки аудио"""
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦИЯ ОБРАБОТКИ АУДИО")
    print("="*60)
    
    config = LLMConfig(
        endpoint="http://10.129.0.37:8082",
        api_key="auth_7313ff09b5b24e529786c48f1bfc068c",
        model="chat"
    )
    
    client = MultimodalLLMClient(config)
    
    # Создаем тестовый аудио файл
    audio_file = create_test_audio("test_audio.wav")
    
    try:
        # 1. Транскрипция аудио
        print("\n1. Транскрипция аудио файла:")
        print("-" * 40)
        
        result = await client.audio_completion(
            text_prompt="Транскрибируй содержимое этого аудио файла",
            audio_files=audio_file,
            task_type="transcription"
        )
        
        print(f"Транскрипция: {result}")
        
        # 2. Анализ аудио
        print("\n2. Анализ аудио контента:")
        print("-" * 40)
        
        result = await client.audio_completion(
            text_prompt="Проанализируй это аудио: определи тип звука, качество, длительность",
            audio_files=audio_file,
            task_type="analysis"
        )
        
        print(f"Анализ: {result}")
        
    except Exception as e:
        print(f"Ошибка при обработке аудио: {e}")
        print("Возможно, LLM сервер не поддерживает аудио обработку или недоступен")
    
    finally:
        # Очистка
        if audio_file.exists():
            audio_file.unlink()


async def demo_mixed_media():
    """Демонстрация работы со смешанными медиа"""
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦИЯ СМЕШАННЫХ МЕДИА")
    print("="*60)
    
    config = LLMConfig(
        endpoint="http://10.129.0.37:8082",
        api_key="auth_7313ff09b5b24e529786c48f1bfc068c",
        model="chat",
        max_tokens=800
    )
    
    client = MultimodalLLMClient(config)
    
    # Создаем тестовые файлы
    image_file = create_test_image("mixed_test.png")
    audio_file = create_test_audio("mixed_audio.wav")
    
    try:
        # Анализ смешанных медиа
        print("\n1. Комплексный анализ изображения и аудио:")
        print("-" * 50)
        
        media_files = {
            "images": [image_file],
            "audio": [audio_file]
        }
        
        result = await client.mixed_media_completion(
            text_prompt="Проанализируй предоставленные медиа файлы. Опиши изображение и аудио контент.",
            media_files=media_files
        )
        
        print(f"Комплексный анализ: {result}")
        
        # 2. Мультимодальный чат
        print("\n2. Мультимодальный чат с контекстом:")
        print("-" * 50)
        
        messages = [
            {"role": "user", "content": "Привет! Я отправлю тебе изображение для анализа."}
        ]
        
        result = await client.chat_completion_multimodal(
            messages=messages,
            images=[image_file]
        )
        
        print(f"Ответ чата: {result}")
        
    except Exception as e:
        print(f"Ошибка при работе со смешанными медиа: {e}")
        print("Возможно, LLM сервер не поддерживает мультимодальные запросы")
    
    finally:
        # Очистка
        for file_path in [image_file, audio_file]:
            if file_path.exists():
                file_path.unlink()


async def demo_utility_functions():
    """Демонстрация утилитарных функций"""
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦИЯ УТИЛИТАРНЫХ ФУНКЦИЙ")
    print("="*60)
    
    # 1. Поддерживаемые форматы
    print("\n1. Поддерживаемые форматы медиа:")
    print("-" * 40)
    
    formats = MultimodalLLMClient.get_supported_formats()
    
    for media_type, format_list in formats.items():
        print(f"{media_type.capitalize()}: {', '.join(format_list)}")
    
    # 2. Создание image URL контента
    print("\n2. Создание image URL контента:")
    print("-" * 40)
    
    test_image = create_test_image("utility_test.png", size=(100, 100))
    
    try:
        content = MultimodalLLMClient.create_image_url_content(test_image, "auto")
        
        print(f"Тип контента: {content['type']}")
        print(f"URL начинается с: {content['image_url']['url'][:50]}...")
        print(f"Уровень детализации: {content['image_url']['detail']}")
        
    finally:
        if test_image.exists():
            test_image.unlink()
    
    # 3. Конфигурация мультимодального режима
    print("\n3. Конфигурация мультимодального режима:")
    print("-" * 40)
    
    config = MultimodalConfig(
        max_image_size=10 * 1024 * 1024,  # 10MB
        max_audio_duration=600,  # 10 минут
        supported_image_formats=["png", "jpg", "webp"],
        auto_resize_images=True
    )
    
    print(f"Максимальный размер изображения: {config.max_image_size / (1024*1024):.1f} MB")
    print(f"Максимальная длительность аудио: {config.max_audio_duration} сек")
    print(f"Поддерживаемые форматы изображений: {config.supported_image_formats}")
    print(f"Автоматическое изменение размера: {config.auto_resize_images}")


async def main():
    """Главная функция демонстрации"""
    print("ДЕМОНСТРАЦИЯ MULTIMODAL LLM CLIENT")
    print("=" * 80)
    print("Этот пример показывает возможности мультимодального клиента Kraken")
    print("для работы с изображениями, аудио и видео файлами.")
    print()
    
    try:
        # Демонстрация различных возможностей
        await demo_vision_analysis()
        await demo_audio_processing()
        await demo_mixed_media()
        await demo_utility_functions()
        
        print("\n" + "="*80)
        print("ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА УСПЕШНО!")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\nДемонстрация прервана пользователем")
    except Exception as e:
        print(f"\nОшибка во время демонстрации: {e}")
        logger.exception("Подробности ошибки:")


if __name__ == "__main__":
    # Запуск демонстрации
    asyncio.run(main())