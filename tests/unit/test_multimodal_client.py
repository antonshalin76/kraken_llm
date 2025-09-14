"""
Модульные тесты для MultimodalLLMClient.

Тестирует функциональность мультимодального клиента с мокированными медиа файлами.
"""

import pytest
import asyncio
import base64
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from kraken_llm.client.multimodal import (
    MultimodalLLMClient,
    MediaFile,
    MultimodalMessage,
    MultimodalConfig
)
from kraken_llm.config.settings import LLMConfig
from kraken_llm.exceptions.validation import ValidationError
from kraken_llm.exceptions.api import APIError


class TestMediaFile:
    """Тесты для модели MediaFile"""

    def test_media_file_creation(self):
        """Тест создания MediaFile с валидными данными"""
        media_file = MediaFile(
            content="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU77zgAAAABJRU5ErkJggg==",
            mime_type="image/png",
            filename="test.png",
            size=1024
        )

        assert media_file.mime_type == "image/png"
        assert media_file.filename == "test.png"
        assert media_file.size == 1024

    def test_media_file_invalid_mime_type(self):
        """Тест валидации неподдерживаемого MIME типа"""
        with pytest.raises(ValueError, match="Неподдерживаемый MIME тип"):
            MediaFile(
                content="test_content",
                mime_type="application/pdf",
                filename="test.pdf"
            )

    def test_media_file_valid_mime_types(self):
        """Тест поддерживаемых MIME типов"""
        valid_types = [
            "image/jpeg", "image/png", "image/gif", "image/webp",
            "audio/mpeg", "audio/wav", "audio/ogg",
            "video/mp4", "video/webm", "video/avi"
        ]

        for mime_type in valid_types:
            media_file = MediaFile(
                content="test_content",
                mime_type=mime_type
            )
            assert media_file.mime_type == mime_type


class TestMultimodalMessage:
    """Тесты для модели MultimodalMessage"""

    def test_text_message(self):
        """Тест создания текстового сообщения"""
        message = MultimodalMessage(
            role="user",
            content="Привет, как дела?"
        )

        assert message.role == "user"
        assert message.content == "Привет, как дела?"

    def test_multimodal_message(self):
        """Тест создания мультимодального сообщения"""
        content = [
            {"type": "text", "text": "Опиши это изображение"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]

        message = MultimodalMessage(
            role="user",
            content=content
        )

        assert message.role == "user"
        assert isinstance(message.content, list)
        assert len(message.content) == 2

    def test_invalid_content_type(self):
        """Тест валидации неподдерживаемого типа контента"""
        with pytest.raises(ValueError, match="Неподдерживаемый тип контента"):
            MultimodalMessage(
                role="user",
                content=[{"type": "invalid_type", "data": "test"}]
            )


class TestMultimodalConfig:
    """Тесты для конфигурации мультимодального режима"""

    def test_default_config(self):
        """Тест конфигурации по умолчанию"""
        config = MultimodalConfig()

        assert config.max_image_size == 20 * 1024 * 1024
        assert config.max_audio_duration == 300
        assert config.max_video_duration == 60
        assert "jpeg" in config.supported_image_formats
        assert "mp3" in config.supported_audio_formats
        assert "mp4" in config.supported_video_formats

    def test_custom_config(self):
        """Тест кастомной конфигурации"""
        config = MultimodalConfig(
            max_image_size=10 * 1024 * 1024,
            supported_image_formats=["png", "jpg"]
        )

        assert config.max_image_size == 10 * 1024 * 1024
        assert config.supported_image_formats == ["png", "jpg"]


class TestMultimodalLLMClient:
    """Тесты для MultimodalLLMClient"""

    @pytest.fixture
    def llm_config(self):
        """Фикстура для базовой конфигурации LLM"""
        return LLMConfig(
            endpoint="http://test-endpoint:8080",
            api_key="test-key",
            model="test-model"
        )

    @pytest.fixture
    def multimodal_config(self):
        """Фикстура для конфигурации мультимодального режима"""
        return MultimodalConfig(
            max_image_size=1024 * 1024,  # 1MB для тестов
            supported_image_formats=["png", "jpg", "jpeg"]
        )

    @pytest.fixture
    def client(self, llm_config, multimodal_config):
        """Фикстура для мультимодального клиента"""
        with patch('kraken_llm.client.multimodal.BaseLLMClient.__init__'):
            client = MultimodalLLMClient(llm_config, multimodal_config)
            client.config = llm_config
            client.multimodal_config = multimodal_config
            return client

    @pytest.fixture
    def temp_image_file(self):
        """Фикстура для временного файла изображения"""
        # Создаем минимальное PNG изображение 1x1 пиксель
        png_data = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU77zgAAAABJRU5ErkJggg=="
        )

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(png_data)
            temp_path = Path(f.name)

        yield temp_path

        # Очистка
        if temp_path.exists():
            temp_path.unlink()

    @pytest.fixture
    def temp_audio_file(self):
        """Фикстура для временного аудио файла"""
        # Создаем минимальный WAV файл
        wav_header = b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00'

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_header)
            temp_path = Path(f.name)

        yield temp_path

        # Очистка
        if temp_path.exists():
            temp_path.unlink()

    def test_client_initialization(self, llm_config, multimodal_config):
        """Тест инициализации клиента"""
        with patch('kraken_llm.client.multimodal.BaseLLMClient.__init__'):
            client = MultimodalLLMClient(llm_config, multimodal_config)
            assert client.multimodal_config == multimodal_config

    def test_client_initialization_default_config(self, llm_config):
        """Тест инициализации клиента с конфигурацией по умолчанию"""
        with patch('kraken_llm.client.multimodal.BaseLLMClient.__init__'):
            client = MultimodalLLMClient(llm_config)
            assert isinstance(client.multimodal_config, MultimodalConfig)

    @pytest.mark.asyncio
    async def test_load_image_file_success(self, client, temp_image_file):
        """Тест успешной загрузки изображения"""
        media_file = await client._load_image_file(temp_image_file)

        assert isinstance(media_file, MediaFile)
        assert media_file.mime_type == "image/png"
        assert media_file.filename == temp_image_file.name
        assert media_file.size > 0
        assert len(media_file.content) > 0

    @pytest.mark.asyncio
    async def test_load_image_file_not_found(self, client):
        """Тест загрузки несуществующего файла"""
        with pytest.raises(ValidationError, match="Файл изображения не найден"):
            await client._load_image_file("nonexistent.png")

    @pytest.mark.asyncio
    async def test_load_image_file_too_large(self, client):
        """Тест загрузки слишком большого файла"""
        # Создаем большой файл
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            # Записываем данные больше максимального размера
            f.write(b"x" * (client.multimodal_config.max_image_size + 1))
            large_file_path = Path(f.name)

        try:
            with pytest.raises(ValidationError, match="превышает максимальный"):
                await client._load_image_file(large_file_path)
        finally:
            large_file_path.unlink()

    @pytest.mark.asyncio
    async def test_load_audio_file_success(self, client, temp_audio_file):
        """Тест успешной загрузки аудио файла"""
        media_file = await client._load_audio_file(temp_audio_file)

        assert isinstance(media_file, MediaFile)
        assert media_file.mime_type in ["audio/wav", "audio/x-wav"]
        assert media_file.filename == temp_audio_file.name
        assert media_file.size > 0

    @pytest.mark.asyncio
    async def test_load_audio_file_unsupported_format(self, client):
        """Тест загрузки неподдерживаемого формата аудио"""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"fake audio data")
            unsupported_file = Path(f.name)

        try:
            with pytest.raises(ValidationError, match="Неподдерживаемый формат аудио"):
                await client._load_audio_file(unsupported_file)
        finally:
            unsupported_file.unlink()

    @pytest.mark.asyncio
    async def test_vision_completion(self, client, temp_image_file):
        """Тест vision completion"""
        with patch.object(client, '_multimodal_completion', new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = "Это изображение содержит..."

            result = await client.vision_completion(
                text_prompt="Опиши это изображение",
                images=temp_image_file
            )

            assert result == "Это изображение содержит..."
            mock_completion.assert_called_once()

            # Проверяем структуру вызова
            call_args = mock_completion.call_args[0][0]  # messages
            assert len(call_args) == 1
            assert call_args[0].role == "user"
            assert isinstance(call_args[0].content, list)
            assert call_args[0].content[0]["type"] == "text"
            assert call_args[0].content[1]["type"] == "image_url"

    @pytest.mark.asyncio
    async def test_vision_completion_multiple_images(self, client, temp_image_file):
        """Тест vision completion с несколькими изображениями"""
        # Создаем второй временный файл
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            png_data = base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU77zgAAAABJRU5ErkJggg=="
            )
            f.write(png_data)
            temp_image_file2 = Path(f.name)

        try:
            with patch.object(client, '_multimodal_completion', new_callable=AsyncMock) as mock_completion:
                mock_completion.return_value = "Анализ двух изображений..."

                result = await client.vision_completion(
                    text_prompt="Сравни эти изображения",
                    images=[temp_image_file, temp_image_file2]
                )

                assert result == "Анализ двух изображений..."

                # Проверяем, что передано 3 элемента контента (1 текст + 2 изображения)
                call_args = mock_completion.call_args[0][0]
                assert len(call_args[0].content) == 3
                assert call_args[0].content[0]["type"] == "text"
                assert call_args[0].content[1]["type"] == "image_url"
                assert call_args[0].content[2]["type"] == "image_url"
        finally:
            temp_image_file2.unlink()

    @pytest.mark.asyncio
    async def test_audio_completion(self, client, temp_audio_file):
        """Тест audio completion"""
        with patch.object(client, '_multimodal_completion', new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = "Транскрипция аудио: Привет мир"

            result = await client.audio_completion(
                text_prompt="Транскрибируй это аудио",
                audio_files=temp_audio_file,
                task_type="transcription"
            )

            assert result == "Транскрипция аудио: Привет мир"
            mock_completion.assert_called_once()

            # Проверяем структуру вызова
            call_args = mock_completion.call_args[0][0]
            assert len(call_args) == 1
            assert "transcription" in call_args[0].content[0]["text"]
            assert call_args[0].content[1]["type"] == "audio_url"

    @pytest.mark.asyncio
    async def test_mixed_media_completion(self, client, temp_image_file, temp_audio_file):
        """Тест mixed media completion"""
        with patch.object(client, '_multimodal_completion', new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = "Анализ изображения и аудио..."

            media_files = {
                "images": [temp_image_file],
                "audio": [temp_audio_file]
            }

            result = await client.mixed_media_completion(
                text_prompt="Проанализируй изображение и аудио",
                media_files=media_files
            )

            assert result == "Анализ изображения и аудио..."
            mock_completion.assert_called_once()

            # Проверяем структуру вызова
            call_args = mock_completion.call_args[0][0]
            assert len(call_args[0].content) == 3  # text + image + audio
            assert call_args[0].content[0]["type"] == "text"
            assert call_args[0].content[1]["type"] == "image_url"
            assert call_args[0].content[2]["type"] == "audio_url"

    @pytest.mark.asyncio
    async def test_multimodal_completion_error_handling(self, client):
        """Тест обработки ошибок в мультимодальном запросе"""
        with patch.object(client, 'chat_completion', new_callable=AsyncMock) as mock_chat:
            mock_chat.side_effect = Exception("API Error")

            messages = [MultimodalMessage(role="user", content="test")]

            with pytest.raises(APIError, match="Ошибка мультимодального запроса"):
                await client._multimodal_completion(messages)

    def test_create_image_url_content(self, temp_image_file):
        """Тест создания контента для изображения"""
        content = MultimodalLLMClient.create_image_url_content(
            temp_image_file, "high")

        assert content["type"] == "image_url"
        assert "data:image/png;base64," in content["image_url"]["url"]
        assert content["image_url"]["detail"] == "high"

    def test_get_supported_formats(self):
        """Тест получения поддерживаемых форматов"""
        formats = MultimodalLLMClient.get_supported_formats()

        assert "images" in formats
        assert "audio" in formats
        assert "video" in formats
        assert "jpeg" in formats["images"]
        assert "mp3" in formats["audio"]
        assert "mp4" in formats["video"]

    @pytest.mark.asyncio
    async def test_streaming_not_supported(self, client):
        """Тест что streaming не поддерживается"""
        with pytest.raises(NotImplementedError, match="не поддерживает streaming"):
            async for _ in client.chat_completion_stream([]):
                break

    @pytest.mark.asyncio
    async def test_structured_output_not_supported(self, client):
        """Тест что structured output не поддерживается"""
        with pytest.raises(NotImplementedError, match="не поддерживает structured output"):
            await client.chat_completion_structured([], Mock())


if __name__ == "__main__":
    pytest.main([__file__])
