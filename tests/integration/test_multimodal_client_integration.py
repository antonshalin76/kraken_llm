"""
Интеграционные тесты для MultimodalLLMClient.

Тестирует реальное взаимодействие с LLM API с медиа файлами.
"""

import pytest
import asyncio
import base64
import tempfile
from pathlib import Path
from PIL import Image
import io
import os

from kraken_llm.client.multimodal import MultimodalLLMClient, MultimodalConfig
from kraken_llm.config.settings import LLMConfig
from kraken_llm.exceptions.validation import ValidationError

from dotenv import load_dotenv

load_dotenv()


class TestMultimodalClientIntegration:
    """Интеграционные тесты для мультимодального клиента"""

    @pytest.fixture
    def llm_config(self):
        """Конфигурация для реального LLM сервера"""
        return LLMConfig(
            endpoint=os.getenv("LLM_ENDPOINT"),
            api_key=os.getenv("LLM_TOKEN"),
            model=os.getenv("LLM_MODEL"),
            temperature=0.7,
            max_tokens=500
        )

    @pytest.fixture
    async def server_supports_vision(self, client, test_image_file):
        """Проверяет, поддерживает ли сервер vision capabilities"""
        try:
            # Пробуем простой vision запрос
            result = await client.vision_completion(
                text_prompt="Test",
                images=test_image_file,
                detail_level="low"
            )
            # Если результат не пустой, сервер поддерживает vision
            return len(result.strip()) > 0
        except Exception:
            return False

    @pytest.fixture
    def multimodal_config(self):
        """Конфигурация мультимодального режима"""
        return MultimodalConfig(
            max_image_size=5 * 1024 * 1024,  # 5MB
            auto_resize_images=True
        )

    @pytest.fixture
    def client(self, llm_config, multimodal_config):
        """Мультимодальный клиент для тестов"""
        return MultimodalLLMClient(llm_config, multimodal_config)

    @pytest.fixture
    def test_image_file(self):
        """Создает тестовое изображение"""
        # Создаем простое цветное изображение 100x100
        img = Image.new('RGB', (100, 100), color='red')

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f, format='PNG')
            temp_path = Path(f.name)

        yield temp_path

        # Очистка
        if temp_path.exists():
            temp_path.unlink()

    @pytest.fixture
    def test_complex_image_file(self):
        """Создает более сложное тестовое изображение с текстом"""
        from PIL import Image, ImageDraw, ImageFont

        # Создаем изображение с текстом
        img = Image.new('RGB', (300, 200), color='white')
        draw = ImageDraw.Draw(img)

        # Рисуем простые фигуры и текст
        draw.rectangle([50, 50, 150, 100], fill='blue', outline='black')
        draw.ellipse([200, 50, 250, 100], fill='green', outline='black')

        try:
            # Пытаемся использовать системный шрифт
            font = ImageFont.load_default()
            draw.text((50, 120), "Test Image", fill='black', font=font)
        except:
            # Если шрифт недоступен, рисуем без него
            draw.text((50, 120), "Test Image", fill='black')

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f, format='PNG')
            temp_path = Path(f.name)

        yield temp_path

        # Очистка
        if temp_path.exists():
            temp_path.unlink()

    @pytest.fixture
    def test_audio_file(self):
        """Создает минимальный тестовый WAV файл"""
        # Создаем минимальный WAV файл (44 байта заголовок + немного данных)
        wav_header = (
            b'RIFF'
            b'\x28\x00\x00\x00'  # Размер файла - 8
            b'WAVE'
            b'fmt '
            b'\x10\x00\x00\x00'  # Размер fmt chunk
            b'\x01\x00'          # Аудио формат (PCM)
            b'\x01\x00'          # Количество каналов
            b'\x44\xac\x00\x00'  # Частота дискретизации (44100)
            b'\x88\x58\x01\x00'  # Байт в секунду
            b'\x02\x00'          # Выравнивание блока
            b'\x10\x00'          # Бит на семпл
            b'data'
            b'\x04\x00\x00\x00'  # Размер данных
            b'\x00\x00\x00\x00'  # Данные (тишина)
        )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_header)
            temp_path = Path(f.name)

        yield temp_path

        # Очистка
        if temp_path.exists():
            temp_path.unlink()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_vision_completion_simple(self, client, test_image_file, server_supports_vision):
        """Тест простого анализа изображения"""
        if not server_supports_vision:
            pytest.skip("Сервер не поддерживает vision capabilities")

        result = await client.vision_completion(
            text_prompt="Опиши это изображение кратко на русском языке",
            images=test_image_file,
            detail_level="low"
        )

        assert isinstance(result, str)
        assert len(result) > 0
        print(f"Результат анализа изображения: {result}")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_vision_completion_detailed(self, client, test_complex_image_file, server_supports_vision):
        """Тест детального анализа сложного изображения"""
        if not server_supports_vision:
            pytest.skip("Сервер не поддерживает vision capabilities")

        result = await client.vision_completion(
            text_prompt="Детально опиши все элементы на этом изображении: фигуры, цвета, текст",
            images=test_complex_image_file,
            detail_level="high"
        )

        assert isinstance(result, str)
        assert len(result) > 50  # Ожидаем детальное описание
        print(f"Детальный анализ изображения: {result}")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_vision_completion_multiple_images(self, client, test_image_file, test_complex_image_file, server_supports_vision):
        """Тест анализа нескольких изображений"""
        if not server_supports_vision:
            pytest.skip("Сервер не поддерживает vision capabilities")

        result = await client.vision_completion(
            text_prompt="Сравни эти два изображения и опиши различия",
            images=[test_image_file, test_complex_image_file]
        )

        assert isinstance(result, str)
        assert len(result) > 0
        print(f"Сравнение изображений: {result}")

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.xfail(reason="Audio content type не поддерживается OpenAI API")
    async def test_audio_completion_transcription(self, client, test_audio_file):
        """Тест транскрипции аудио"""
        result = await client.audio_completion(
            text_prompt="Транскрибируй содержимое этого аудио файла",
            audio_files=test_audio_file,
            task_type="transcription"
        )

        assert isinstance(result, str)
        assert len(result) > 0
        print(f"Транскрипция аудио: {result}")

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.xfail(reason="Mixed media с audio не поддерживается OpenAI API")
    async def test_mixed_media_completion(self, client, test_image_file, test_audio_file):
        """Тест анализа смешанных медиа"""
        media_files = {
            "images": [test_image_file],
            "audio": [test_audio_file]
        }

        result = await client.mixed_media_completion(
            text_prompt="Проанализируй предоставленное изображение и аудио файл",
            media_files=media_files
        )

        assert isinstance(result, str)
        assert len(result) > 0
        print(f"Анализ смешанных медиа: {result}")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_chat_completion_multimodal(self, client, test_image_file, server_supports_vision):
        """Тест мультимодального chat completion"""
        if not server_supports_vision:
            pytest.skip("Сервер не поддерживает vision capabilities")

        messages = [
            {"role": "user", "content": "Привет! Можешь проанализировать изображение?"}
        ]

        result = await client.chat_completion_multimodal(
            messages=messages,
            images=[test_image_file]
        )

        assert isinstance(result, str)
        assert len(result) > 0
        print(f"Мультимодальный chat: {result}")

    @pytest.mark.asyncio
    async def test_file_validation_errors(self, client):
        """Тест валидации файлов"""
        # Тест несуществующего файла
        with pytest.raises(ValidationError, match="не найден"):
            await client._load_image_file("nonexistent.png")

        # Тест неподдерживаемого формата
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not an image")
            txt_file = Path(f.name)

        try:
            with pytest.raises(ValidationError, match="Неподдерживаемый формат"):
                await client._load_image_file(txt_file)
        finally:
            txt_file.unlink()

    @pytest.mark.asyncio
    async def test_large_file_handling(self, client):
        """Тест обработки больших файлов"""
        # Создаем большое изображение
        large_img = Image.new('RGB', (2000, 2000), color='blue')

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            large_img.save(f, format='PNG')
            large_file = Path(f.name)

        try:
            # Если файл больше лимита, должна быть ошибка
            file_size = large_file.stat().st_size
            if file_size > client.multimodal_config.max_image_size:
                with pytest.raises(ValidationError, match="превышает максимальный"):
                    await client._load_image_file(large_file)
            else:
                # Если файл в пределах лимита, должен загрузиться
                media_file = await client._load_image_file(large_file)
                assert media_file.size == file_size

        finally:
            large_file.unlink()

    def test_utility_methods(self, test_image_file):
        """Тест утилитарных методов"""
        # Тест создания image URL контента
        content = MultimodalLLMClient.create_image_url_content(
            test_image_file, "auto")

        assert content["type"] == "image_url"
        assert "data:image/png;base64," in content["image_url"]["url"]
        assert content["image_url"]["detail"] == "auto"

        # Тест получения поддерживаемых форматов
        formats = MultimodalLLMClient.get_supported_formats()

        assert isinstance(formats, dict)
        assert "images" in formats
        assert "audio" in formats
        assert "video" in formats

        for format_list in formats.values():
            assert isinstance(format_list, list)
            assert len(format_list) > 0


if __name__ == "__main__":
    # Запуск только интеграционных тестов
    pytest.main([__file__, "-m", "integration", "-v"])
