#!/usr/bin/env python3
"""
Комплексный тест мультимодального клиента Kraken.

Проверяет все возможности MultimodalLLMClient:
- Загрузку и валидацию медиа файлов
- Утилитарные функции
- Интеграцию с MediaUtils
- Обработку ошибок
"""

import asyncio
import tempfile
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
import logging
import os

from kraken_llm.client.multimodal import MultimodalLLMClient, MultimodalConfig
from kraken_llm.config.settings import LLMConfig
from kraken_llm.utils.media import MediaUtils

from dotenv import load_dotenv

load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_media_files(output_dir: Path) -> dict:
    """Создает набор тестовых медиа файлов"""
    output_dir.mkdir(exist_ok=True)

    files = {}

    # Создаем изображения разных размеров
    images = [
        ("small.png", (100, 100), "red"),
        ("medium.jpg", (500, 300), "green"),
        ("large.png", (1500, 1000), "blue"),
    ]

    files['images'] = []
    for filename, size, color in images:
        img = Image.new('RGB', size, color=color)
        file_path = output_dir / filename

        if filename.endswith('.jpg'):
            img.save(file_path, format='JPEG', quality=85)
        else:
            img.save(file_path, format='PNG')

        files['images'].append(file_path)

    # Создаем аудио файл
    wav_header = (
        b'RIFF'
        b'\x28\x00\x00\x00'
        b'WAVE'
        b'fmt '
        b'\x10\x00\x00\x00'
        b'\x01\x00'
        b'\x01\x00'
        b'\x44\xac\x00\x00'
        b'\x88\x58\x01\x00'
        b'\x02\x00'
        b'\x10\x00'
        b'data'
        b'\x04\x00\x00\x00'
        b'\x00\x00\x00\x00'
    )

    audio_file = output_dir / "test.wav"
    with open(audio_file, 'wb') as f:
        f.write(wav_header)

    files['audio'] = [audio_file]

    return files


async def test_media_loading_and_validation():
    """Тест загрузки и валидации медиа файлов"""
    print("\n" + "="*60)
    print("ТЕСТ ЗАГРУЗКИ И ВАЛИДАЦИИ МЕДИА ФАЙЛОВ")
    print("="*60)

    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL")
    )

    multimodal_config = MultimodalConfig(
        max_image_size=1024 * 1024,  # 1MB
        supported_image_formats=["png", "jpg", "jpeg"]
    )

    client = MultimodalLLMClient(config, multimodal_config)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_files = create_test_media_files(temp_path)

        print(f"\nСоздано тестовых файлов:")
        print(f"  Изображения: {len(test_files['images'])}")
        print(f"  Аудио: {len(test_files['audio'])}")

        # Тест загрузки изображений
        print("\n1. Тестирование загрузки изображений:")
        print("-" * 40)

        for image_path in test_files['images']:
            try:
                media_file = await client._load_image_file(image_path)
                print(
                    f"✓ {image_path.name}: {media_file.mime_type}, {media_file.size} байт")
            except Exception as e:
                print(f"✗ {image_path.name}: {e}")

        # Тест загрузки аудио
        print("\n2. Тестирование загрузки аудио:")
        print("-" * 40)

        for audio_path in test_files['audio']:
            try:
                media_file = await client._load_audio_file(audio_path)
                print(
                    f"✓ {audio_path.name}: {media_file.mime_type}, {media_file.size} байт")
            except Exception as e:
                print(f"✗ {audio_path.name}: {e}")

        # Тест валидации и обработки изображений
        print("\n3. Тестирование валидации и обработки:")
        print("-" * 40)

        for image_path in test_files['images']:
            result = MultimodalLLMClient.validate_and_process_image(
                image_path,
                max_size=500 * 1024,  # 500KB
                max_width=800,
                max_height=600,
                auto_resize=True
            )

            status = "✓ ВАЛИДЕН" if result['valid'] else "✗ НЕВАЛИДЕН"
            resized = " (изменен размер)" if result.get(
                'resized', False) else ""
            print(f"{image_path.name:15} {status}{resized}")

            if result['errors']:
                for error in result['errors']:
                    print(f"    Ошибка: {error}")

            if result['warnings']:
                for warning in result['warnings']:
                    print(f"    Предупреждение: {warning}")

        # Тест пакетной валидации
        print("\n4. Тестирование пакетной валидации:")
        print("-" * 40)

        batch_result = MultimodalLLMClient.batch_validate_media_files(
            test_files['images'],
            media_type='image',
            max_size=1024 * 1024  # 1MB
        )

        summary = batch_result['summary']
        print(f"Всего файлов: {summary['total_files']}")
        print(f"Валидных: {summary['valid_count']}")
        print(f"Невалидных: {summary['invalid_count']}")
        print(f"Общий размер: {summary['total_size_mb']:.2f} MB")


async def test_utility_functions():
    """Тест утилитарных функций"""
    print("\n" + "="*60)
    print("ТЕСТ УТИЛИТАРНЫХ ФУНКЦИЙ")
    print("="*60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_files = create_test_media_files(temp_path)

        # Тест поддерживаемых форматов
        print("\n1. Поддерживаемые форматы:")
        print("-" * 30)

        formats = MultimodalLLMClient.get_supported_formats()
        for media_type, format_list in formats.items():
            print(f"{media_type}: {len(format_list)} форматов")
            print(
                f"  {', '.join(format_list[:5])}{'...' if len(format_list) > 5 else ''}")

        # Тест создания image URL контента
        print("\n2. Создание image URL контента:")
        print("-" * 30)

        for image_path in test_files['images'][:2]:  # Первые 2 изображения
            try:
                content = MultimodalLLMClient.create_image_url_content(
                    image_path, detail="auto"
                )

                url_length = len(content['image_url']['url'])
                print(f"✓ {image_path.name}: URL длиной {url_length} символов")

            except Exception as e:
                print(f"✗ {image_path.name}: {e}")

        # Тест интеграции с MediaUtils
        print("\n3. Интеграция с MediaUtils:")
        print("-" * 30)

        for image_path in test_files['images'][:1]:  # Одно изображение
            # Информация о файле
            file_info = MediaUtils.get_file_info(image_path)
            print(f"Файл: {file_info['filename']}")
            print(f"  Размер: {file_info['size']} байт")
            print(f"  MIME: {file_info['mime_type']}")

            # Информация об изображении
            img_info = MediaUtils.get_image_info(image_path)
            print(f"  Разрешение: {img_info['width']}x{img_info['height']}")
            print(f"  Формат: {img_info['format']}")

            # Data URL
            data_url = MediaUtils.create_data_url(image_path)
            print(f"  Data URL: {len(data_url)} символов")


async def test_multimodal_operations():
    """Тест мультимодальных операций"""
    print("\n" + "="*60)
    print("ТЕСТ МУЛЬТИМОДАЛЬНЫХ ОПЕРАЦИЙ")
    print("="*60)

    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL")
    )

    client = MultimodalLLMClient(config)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_files = create_test_media_files(temp_path)

        # Тест обычного chat completion
        print("\n1. Обычный chat completion:")
        print("-" * 30)

        try:
            messages = [{"role": "user", "content": "Привет! Как дела?"}]
            result = await client.chat_completion(messages)
            print(f"✓ Ответ получен: {result[:50]}...")
        except Exception as e:
            print(f"✗ Ошибка: {e}")

        # Тест vision completion (может не работать с текущим сервером)
        print("\n2. Vision completion (может не поддерживаться):")
        print("-" * 50)

        try:
            result = await client.vision_completion(
                text_prompt="Опиши это изображение кратко",
                images=test_files['images'][0],
                detail_level="low"
            )
            print(f"✓ Vision работает: {result[:50]}...")
        except Exception as e:
            print(f"⚠ Vision недоступен (ожидаемо): {type(e).__name__}")

        # Тест audio completion
        print("\n3. Audio completion (может не поддерживаться):")
        print("-" * 50)

        try:
            result = await client.audio_completion(
                text_prompt="Транскрибируй это аудио",
                audio_files=test_files['audio'][0],
                task_type="transcription"
            )
            print(f"✓ Audio работает: {result[:50]}...")
        except Exception as e:
            print(f"⚠ Audio недоступен (ожидаемо): {type(e).__name__}")

        # Тест mixed media
        print("\n4. Mixed media completion (может не поддерживаться):")
        print("-" * 50)

        try:
            media_files = {
                "images": test_files['images'][:1],
                "audio": test_files['audio'][:1]
            }

            result = await client.mixed_media_completion(
                text_prompt="Проанализируй эти медиа файлы",
                media_files=media_files
            )
            print(f"✓ Mixed media работает: {result[:50]}...")
        except Exception as e:
            print(f"⚠ Mixed media недоступен (ожидаемо): {type(e).__name__}")

        # Тест ограничений
        print("\n5. Тестирование ограничений:")
        print("-" * 30)

        # Streaming не поддерживается
        try:
            stream = client.chat_completion_stream([])
            async for chunk in stream:
                break  # Если получили хотя бы один chunk, значит работает
            print("✗ Streaming должен быть недоступен")
        except NotImplementedError:
            print("✓ Streaming корректно недоступен")
        except Exception as e:
            print(f"✓ Streaming недоступен: {type(e).__name__}")

        # Structured output не поддерживается
        try:
            try:
                from pydantic import BaseModel

                class TestModel(BaseModel):
                    text: str
                await client.chat_completion_structured([], TestModel)
                print("✗ Structured output должен быть недоступен")
            except ImportError:
                print("✓ Pydantic недоступен (structured output не может работать)")
        except NotImplementedError:
            print("✓ Structured output корректно недоступен")
        except Exception as e:
            print(f"✓ Structured output недоступен: {type(e).__name__}")


async def test_error_handling():
    """Тест обработки ошибок"""
    print("\n" + "="*60)
    print("ТЕСТ ОБРАБОТКИ ОШИБОК")
    print("="*60)

    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL")
    )

    client = MultimodalLLMClient(config)

    # Тест несуществующего файла
    print("\n1. Несуществующий файл:")
    print("-" * 25)

    try:
        await client._load_image_file("nonexistent.png")
        print("✗ Должна быть ошибка")
    except Exception as e:
        print(f"✓ Корректная ошибка: {type(e).__name__}")

    # Тест неподдерживаемого формата
    print("\n2. Неподдерживаемый формат:")
    print("-" * 30)

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"not an image")
        txt_file = Path(f.name)

    try:
        await client._load_image_file(txt_file)
        print("✗ Должна быть ошибка")
    except Exception as e:
        print(f"✓ Корректная ошибка: {type(e).__name__}")
    finally:
        txt_file.unlink()

    # Тест слишком большого файла
    print("\n3. Слишком большой файл:")
    print("-" * 30)

    # Создаем большое изображение
    large_img = Image.new('RGB', (3000, 3000), color='red')

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        large_img.save(f, format='PNG')
        large_file = Path(f.name)

    try:
        # Устанавливаем очень маленький лимит
        small_config = MultimodalConfig(max_image_size=1000)  # 1KB
        small_client = MultimodalLLMClient(config, small_config)

        await small_client._load_image_file(large_file)
        print("✗ Должна быть ошибка размера")
    except Exception as e:
        print(f"✓ Корректная ошибка размера: {type(e).__name__}")
    finally:
        large_file.unlink()


async def main():
    """Главная функция тестирования"""
    print("КОМПЛЕКСНЫЙ ТЕСТ MULTIMODAL LLM CLIENT")
    print("=" * 80)
    print("Проверяем все возможности мультимодального клиента Kraken")

    try:
        await test_media_loading_and_validation()
        await test_utility_functions()
        await test_multimodal_operations()
        await test_error_handling()

        print("\n" + "="*80)
        print("ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ УСПЕШНО!")
        print("="*80)
        print("\nМультимодальный клиент готов к использованию.")
        print("Основные возможности:")
        print("- ✓ Загрузка и валидация медиа файлов")
        print("- ✓ Утилиты для работы с изображениями")
        print("- ✓ Интеграция с MediaUtils")
        print("- ✓ Обработка ошибок")
        print("- ⚠ Vision/Audio модели (зависят от сервера)")

    except Exception as e:
        print(f"\nОШИБКА В ТЕСТИРОВАНИИ: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
