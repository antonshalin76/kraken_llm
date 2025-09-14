"""
Модульные тесты для ASRClient.

Тестирует функциональность ASR клиента с мокированными данными.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from kraken_llm.client.asr import (
    ASRClient,
    ASRConfig,
    ASRSegment,
    SpeakerInfo,
    VADResult,
    EmotionAnalysis
)
from kraken_llm.config.settings import LLMConfig
from kraken_llm.exceptions.validation import ValidationError
from kraken_llm.exceptions.api import APIError


class TestASRConfig:
    """Тесты для конфигурации ASR"""

    def test_default_config(self):
        """Тест конфигурации по умолчанию"""
        config = ASRConfig()

        assert config.default_language == "ru"
        assert config.enable_timestamps is True
        assert config.quality_level == "high"
        assert config.max_speakers == 10
        assert config.vad_threshold == 0.5
        assert config.tts_voice == "default"
        assert "wav" in config.supported_audio_formats
        assert config.sample_rate == 16000

    def test_custom_config(self):
        """Тест кастомной конфигурации"""
        config = ASRConfig(
            default_language="en",
            max_speakers=5,
            vad_threshold=0.7,
            tts_speed=1.5
        )

        assert config.default_language == "en"
        assert config.max_speakers == 5
        assert config.vad_threshold == 0.7
        assert config.tts_speed == 1.5


class TestASRModels:
    """Тесты для моделей данных ASR"""

    def test_asr_segment_creation(self):
        """Тест создания ASRSegment"""
        segment = ASRSegment(
            start_time=0.0,
            end_time=5.0,
            text="Привет мир",
            confidence=0.95,
            speaker_id="speaker_1",
            language="ru"
        )

        assert segment.start_time == 0.0
        assert segment.end_time == 5.0
        assert segment.text == "Привет мир"
        assert segment.confidence == 0.95
        assert segment.speaker_id == "speaker_1"
        assert segment.language == "ru"

    def test_speaker_info_creation(self):
        """Тест создания SpeakerInfo"""
        speaker = SpeakerInfo(
            speaker_id="speaker_1",
            total_duration=120.5,
            segments_count=15,
            confidence=0.88,
            gender="male",
            age_group="adult"
        )

        assert speaker.speaker_id == "speaker_1"
        assert speaker.total_duration == 120.5
        assert speaker.segments_count == 15
        assert speaker.confidence == 0.88
        assert speaker.gender == "male"
        assert speaker.age_group == "adult"

    def test_vad_result_creation(self):
        """Тест создания VADResult"""
        vad = VADResult(
            speech_segments=[(0.0, 5.0), (10.0, 15.0)],
            total_speech_duration=10.0,
            total_silence_duration=5.0,
            speech_ratio=0.67
        )

        assert len(vad.speech_segments) == 2
        assert vad.total_speech_duration == 10.0
        assert vad.total_silence_duration == 5.0
        assert vad.speech_ratio == 0.67

    def test_emotion_analysis_creation(self):
        """Тест создания EmotionAnalysis"""
        emotions = EmotionAnalysis(
            dominant_emotion="happy",
            emotions={"happy": 0.7, "neutral": 0.2, "sad": 0.1},
            confidence=0.85,
            valence=0.6,
            arousal=0.4
        )

        assert emotions.dominant_emotion == "happy"
        assert emotions.emotions["happy"] == 0.7
        assert emotions.confidence == 0.85
        assert emotions.valence == 0.6
        assert emotions.arousal == 0.4


class TestASRClient:
    """Тесты для ASRClient"""

    @pytest.fixture
    def llm_config(self):
        """Фикстура для базовой конфигурации LLM"""
        return LLMConfig(
            endpoint="http://test-endpoint:8080",
            api_key="test-key",
            model="test-model"
        )

    @pytest.fixture
    def asr_config(self):
        """Фикстура для конфигурации ASR"""
        return ASRConfig(
            default_language="ru",
            max_speakers=5,
            max_file_size=10 * 1024 * 1024  # 10MB для тестов
        )

    @pytest.fixture
    def client(self, llm_config, asr_config):
        """Фикстура для ASR клиента"""
        with patch('kraken_llm.client.base.BaseLLMClient.__init__'):
            client = ASRClient(llm_config, asr_config)
            client.config = llm_config
            client.asr_config = asr_config
            return client

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

    def test_client_initialization(self, llm_config, asr_config):
        """Тест инициализации клиента"""
        with patch('kraken_llm.client.base.BaseLLMClient.__init__'):
            client = ASRClient(llm_config, asr_config)
            assert client.asr_config == asr_config

    def test_client_initialization_default_config(self, llm_config):
        """Тест инициализации клиента с конфигурацией по умолчанию"""
        with patch('kraken_llm.client.base.BaseLLMClient.__init__'):
            client = ASRClient(llm_config)
            assert isinstance(client.asr_config, ASRConfig)

    @pytest.mark.asyncio
    async def test_validate_audio_file_success(self, client, temp_audio_file):
        """Тест успешной валидации аудио файла"""
        with patch('kraken_llm.utils.media.MediaUtils.validate_media_file') as mock_validate:
            mock_validate.return_value = {'valid': True, 'errors': []}

            # Не должно выбросить исключение
            await client._validate_audio_file(temp_audio_file)

            mock_validate.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_audio_file_not_found(self, client):
        """Тест валидации несуществующего файла"""
        with pytest.raises(ValidationError, match="Аудио файл не найден"):
            await client._validate_audio_file("nonexistent.wav")

    @pytest.mark.asyncio
    async def test_validate_audio_file_too_large(self, client):
        """Тест валидации слишком большого файла"""
        # Создаем большой файл
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            # Записываем данные больше максимального размера
            f.write(b"x" * (client.asr_config.max_file_size + 1))
            large_file_path = Path(f.name)

        try:
            with pytest.raises(ValidationError, match="превышает максимальный"):
                await client._validate_audio_file(large_file_path)
        finally:
            large_file_path.unlink()

    @pytest.mark.asyncio
    async def test_validate_audio_file_unsupported_format(self, client):
        """Тест валидации неподдерживаемого формата"""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"fake audio data")
            unsupported_file = Path(f.name)

        try:
            with pytest.raises(ValidationError, match="Неподдерживаемый формат аудио"):
                await client._validate_audio_file(unsupported_file)
        finally:
            unsupported_file.unlink()

    @pytest.mark.asyncio
    async def test_load_audio_data(self, client, temp_audio_file):
        """Тест загрузки аудио данных"""
        with patch('kraken_llm.utils.media.MediaUtils.encode_file_to_base64') as mock_encode:
            mock_encode.return_value = "base64_encoded_data"

            result = await client._load_audio_data(temp_audio_file)

            assert result == "base64_encoded_data"
            mock_encode.assert_called_once_with(temp_audio_file)

    @pytest.mark.asyncio
    async def test_speech_to_text(self, client, temp_audio_file):
        """Тест распознавания речи"""
        # Мокируем валидацию и загрузку данных
        with patch.object(client, '_validate_audio_file', new_callable=AsyncMock) as mock_validate, \
                patch.object(client, '_load_audio_data', new_callable=AsyncMock) as mock_load, \
                patch.object(client, '_call_asr_api', new_callable=AsyncMock) as mock_api, \
                patch.object(client, '_process_stt_result', new_callable=AsyncMock) as mock_process:

            mock_load.return_value = "base64_audio_data"
            mock_api.return_value = {"text": "Привет мир", "confidence": 0.95}
            mock_process.return_value = {
                "text": "Привет мир",
                "confidence": 0.95,
                "language": "ru",
                "duration": 5.0
            }

            result = await client.speech_to_text(temp_audio_file, language="ru")

            assert result["text"] == "Привет мир"
            assert result["confidence"] == 0.95
            assert result["language"] == "ru"

            mock_validate.assert_called_once()
            mock_load.assert_called_once()
            mock_api.assert_called_once()
            mock_process.assert_called_once()

    @pytest.mark.asyncio
    async def test_text_to_speech(self, client):
        """Тест синтеза речи"""
        with patch.object(client, '_call_asr_api', new_callable=AsyncMock) as mock_api, \
                patch.object(client, '_extract_audio_data', new_callable=AsyncMock) as mock_extract:

            mock_api.return_value = {"audio_data": "base64_audio"}
            mock_extract.return_value = b"audio_bytes_data"

            result = await client.text_to_speech("Привет мир")

            assert result == b"audio_bytes_data"
            mock_api.assert_called_once()
            mock_extract.assert_called_once()

    @pytest.mark.asyncio
    async def test_text_to_speech_empty_text(self, client):
        """Тест синтеза речи с пустым текстом"""
        with pytest.raises(ValidationError, match="Текст для синтеза не может быть пустым"):
            await client.text_to_speech("")

    @pytest.mark.asyncio
    async def test_text_to_speech_with_output_file(self, client):
        """Тест синтеза речи с сохранением в файл"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        try:
            with patch.object(client, '_call_asr_api', new_callable=AsyncMock) as mock_api, \
                    patch.object(client, '_extract_audio_data', new_callable=AsyncMock) as mock_extract:

                mock_api.return_value = {"audio_data": "base64_audio"}
                mock_extract.return_value = b"audio_bytes_data"

                result = await client.text_to_speech("Привет мир", output_file=output_path)

                assert result == output_path
                assert output_path.exists()

                # Проверяем содержимое файла
                with open(output_path, 'rb') as f:
                    content = f.read()
                assert content == b"audio_bytes_data"

        finally:
            if output_path.exists():
                output_path.unlink()

    @pytest.mark.asyncio
    async def test_voice_activity_detection(self, client, temp_audio_file):
        """Тест детекции речевой активности"""
        with patch.object(client, '_validate_audio_file', new_callable=AsyncMock) as mock_validate, \
                patch.object(client, '_load_audio_data', new_callable=AsyncMock) as mock_load, \
                patch.object(client, '_call_asr_api', new_callable=AsyncMock) as mock_api:

            mock_load.return_value = "base64_audio_data"
            mock_api.return_value = {
                "speech_segments": [(0.0, 5.0), (10.0, 15.0)],
                "total_speech_duration": 10.0,
                "total_silence_duration": 5.0,
                "speech_ratio": 0.67
            }

            result = await client.voice_activity_detection(temp_audio_file)

            assert isinstance(result, VADResult)
            assert len(result.speech_segments) == 2
            assert result.total_speech_duration == 10.0
            assert result.speech_ratio == 0.67

    @pytest.mark.asyncio
    async def test_speaker_diarization(self, client, temp_audio_file):
        """Тест диаризации спикеров"""
        with patch.object(client, '_validate_audio_file', new_callable=AsyncMock) as mock_validate, \
                patch.object(client, '_load_audio_data', new_callable=AsyncMock) as mock_load, \
                patch.object(client, '_call_asr_api', new_callable=AsyncMock) as mock_api:

            mock_load.return_value = "base64_audio_data"
            mock_api.return_value = {
                "speakers": [
                    {
                        "speaker_id": "speaker_1",
                        "total_duration": 60.0,
                        "segments_count": 10,
                        "confidence": 0.9,
                        "gender": "male"
                    }
                ],
                "segments": [],
                "total_duration": 120.0
            }

            result = await client.speaker_diarization(temp_audio_file, num_speakers=2)

            assert result["total_speakers"] == 1
            assert len(result["speakers"]) == 1
            assert result["speakers"][0].speaker_id == "speaker_1"
            assert result["speakers"][0].gender == "male"

    @pytest.mark.asyncio
    async def test_emotion_analysis(self, client, temp_audio_file):
        """Тест анализа эмоций"""
        with patch.object(client, '_validate_audio_file', new_callable=AsyncMock) as mock_validate, \
                patch.object(client, '_load_audio_data', new_callable=AsyncMock) as mock_load, \
                patch.object(client, '_call_asr_api', new_callable=AsyncMock) as mock_api:

            mock_load.return_value = "base64_audio_data"
            mock_api.return_value = {
                "dominant_emotion": "happy",
                "emotions": {"happy": 0.7, "neutral": 0.2, "sad": 0.1},
                "confidence": 0.85,
                "valence": 0.6,
                "arousal": 0.4
            }

            result = await client.emotion_analysis(temp_audio_file)

            assert isinstance(result, EmotionAnalysis)
            assert result.dominant_emotion == "happy"
            assert result.emotions["happy"] == 0.7
            assert result.confidence == 0.85

    @pytest.mark.asyncio
    async def test_transcribe_with_analysis(self, client, temp_audio_file):
        """Тест комплексной обработки аудио"""
        with patch.object(client, 'speech_to_text', new_callable=AsyncMock) as mock_stt, \
                patch.object(client, 'voice_activity_detection', new_callable=AsyncMock) as mock_vad, \
                patch.object(client, 'speaker_diarization', new_callable=AsyncMock) as mock_diarization, \
                patch.object(client, 'emotion_analysis', new_callable=AsyncMock) as mock_emotions:

            mock_stt.return_value = {
                "text": "Привет мир",
                "confidence": 0.95,
                "duration": 5.0
            }

            mock_vad.return_value = VADResult(
                speech_segments=[(0.0, 5.0)],
                total_speech_duration=5.0,
                total_silence_duration=0.0,
                speech_ratio=1.0
            )

            mock_diarization.return_value = {"total_speakers": 1}
            mock_emotions.return_value = EmotionAnalysis(
                dominant_emotion="neutral",
                emotions={"neutral": 1.0},
                confidence=0.8,
                valence=0.0,
                arousal=0.0
            )

            result = await client.transcribe_with_analysis(temp_audio_file)

            assert "transcription" in result
            assert "voice_activity" in result
            assert "summary" in result
            assert result["transcription"]["text"] == "Привет мир"

    def test_get_supported_languages(self):
        """Тест получения поддерживаемых языков"""
        languages = ASRClient.get_supported_languages()

        assert isinstance(languages, list)
        assert "ru" in languages
        assert "en" in languages
        assert len(languages) > 10

    def test_get_supported_voices(self):
        """Тест получения поддерживаемых голосов"""
        voices = ASRClient.get_supported_voices()

        assert isinstance(voices, dict)
        assert "ru" in voices
        assert "en" in voices
        assert isinstance(voices["ru"], list)
        assert len(voices["ru"]) > 0

    def test_get_supported_emotions(self):
        """Тест получения поддерживаемых эмоций"""
        emotions = ASRClient.get_supported_emotions()

        assert isinstance(emotions, list)
        assert "neutral" in emotions
        assert "happy" in emotions
        assert "sad" in emotions
        assert len(emotions) > 5

    def test_get_config_summary(self, client):
        """Тест получения сводки конфигурации"""
        summary = client.get_config_summary()

        assert isinstance(summary, dict)
        assert "default_language" in summary
        assert "quality_level" in summary
        assert "max_speakers" in summary
        assert "supported_formats" in summary
        assert "features" in summary

        assert summary["default_language"] == client.asr_config.default_language
        assert summary["max_speakers"] == client.asr_config.max_speakers

    @pytest.mark.asyncio
    async def test_streaming_not_supported(self, client):
        """Тест что обычный streaming не поддерживается"""
        with pytest.raises(NotImplementedError, match="не поддерживает обычный streaming"):
            await client.chat_completion_stream([])

    @pytest.mark.asyncio
    async def test_structured_output_not_supported(self, client):
        """Тест что structured output не поддерживается"""
        with pytest.raises(NotImplementedError, match="не поддерживает structured output"):
            await client.chat_completion_structured([], Mock())


if __name__ == "__main__":
    pytest.main([__file__])
