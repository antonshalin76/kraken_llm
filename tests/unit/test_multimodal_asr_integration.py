"""
Тесты интеграции ASR клиента с мультимодальным клиентом.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from kraken_llm.client.multimodal import MultimodalLLMClient, MultimodalConfig
from kraken_llm.client.asr import ASRClient, ASRConfig, VADResult, EmotionAnalysis
from kraken_llm.config.settings import LLMConfig


class TestMultimodalASRIntegration:
    """Тесты интеграции ASR с мультимодальным клиентом"""

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
        return MultimodalConfig()

    @pytest.fixture
    def client(self, llm_config, multimodal_config):
        """Фикстура для мультимодального клиента"""
        with patch('kraken_llm.client.multimodal.BaseLLMClient.__init__'):
            client = MultimodalLLMClient(llm_config, multimodal_config)
            client.config = llm_config
            client.multimodal_config = multimodal_config
            return client

    @pytest.fixture
    def temp_audio_file(self):
        """Фикстура для временного аудио файла"""
        wav_header = b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00'

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_header)
            temp_path = Path(f.name)

        yield temp_path

        if temp_path.exists():
            temp_path.unlink()

    def test_asr_client_lazy_initialization(self, client):
        """Тест ленивой инициализации ASR клиента"""
        # Изначально ASR клиент не создан
        assert client._asr_client is None

        # После первого обращения создается
        with patch('kraken_llm.client.asr.ASRClient') as mock_asr_class:
            mock_asr_instance = Mock()
            mock_asr_class.return_value = mock_asr_instance

            asr_client = client._get_asr_client()

            assert asr_client == mock_asr_instance
            assert client._asr_client == mock_asr_instance
            mock_asr_class.assert_called_once()

        # При повторном обращении используется существующий
        asr_client2 = client._get_asr_client()
        assert asr_client2 == mock_asr_instance

    @pytest.mark.asyncio
    async def test_speech_to_text_advanced_single_file(self, client, temp_audio_file):
        """Тест продвинутого распознавания речи для одного файла"""
        mock_asr_client = Mock()
        mock_asr_client.transcribe_with_analysis = AsyncMock()
        mock_asr_client.transcribe_with_analysis.return_value = {
            "transcription": {
                "text": "Привет мир",
                "confidence": 0.95,
                "duration": 5.0
            },
            "voice_activity": VADResult(
                speech_segments=[(0.0, 5.0)],
                total_speech_duration=5.0,
                total_silence_duration=0.0,
                speech_ratio=1.0
            ),
            "summary": {"total_duration": 5.0}
        }

        with patch.object(client, '_get_asr_client', return_value=mock_asr_client):
            result = await client.speech_to_text_advanced(
                temp_audio_file,
                language="ru",
                include_diarization=True,
                include_emotions=True
            )

            assert result["transcription"]["text"] == "Привет мир"
            assert result["file_path"] == str(temp_audio_file)

            mock_asr_client.transcribe_with_analysis.assert_called_once_with(
                temp_audio_file,
                include_diarization=True,
                include_emotions=True,
                include_vad=True,
                language="ru"
            )

    @pytest.mark.asyncio
    async def test_speech_to_text_advanced_multiple_files(self, client):
        """Тест продвинутого распознавания речи для нескольких файлов"""
        # Создаем два временных файла
        temp_files = []
        for i in range(2):
            with tempfile.NamedTemporaryFile(suffix=f"_{i}.wav", delete=False) as f:
                f.write(
                    b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00')
                temp_files.append(Path(f.name))

        try:
            mock_asr_client = Mock()
            mock_asr_client.transcribe_with_analysis = AsyncMock()
            mock_asr_client.transcribe_with_analysis.side_effect = [
                {"transcription": {"text": f"Файл {i}", "confidence": 0.9}}
                for i in range(2)
            ]

            with patch.object(client, '_get_asr_client', return_value=mock_asr_client):
                result = await client.speech_to_text_advanced(temp_files)

                assert result["files_processed"] == 2
                assert result["successful"] == 2
                assert result["failed"] == 0
                assert len(result["results"]) == 2

                for i, file_result in enumerate(result["results"]):
                    assert file_result["transcription"]["text"] == f"Файл {i}"
                    assert file_result["file_path"] == str(temp_files[i])

        finally:
            for temp_file in temp_files:
                if temp_file.exists():
                    temp_file.unlink()

    @pytest.mark.asyncio
    async def test_speech_to_text_advanced_with_error(self, client, temp_audio_file):
        """Тест обработки ошибок в продвинутом распознавании речи"""
        mock_asr_client = Mock()
        mock_asr_client.transcribe_with_analysis = AsyncMock()
        mock_asr_client.transcribe_with_analysis.side_effect = Exception(
            "ASR Error")

        with patch.object(client, '_get_asr_client', return_value=mock_asr_client):
            result = await client.speech_to_text_advanced(temp_audio_file)

            assert result["file_path"] == str(temp_audio_file)
            assert result["error"] == "ASR Error"
            assert result["success"] is False

    @pytest.mark.asyncio
    async def test_text_to_speech_multimodal(self, client):
        """Тест синтеза речи через мультимодальный клиент"""
        mock_asr_client = Mock()
        mock_asr_client.text_to_speech = AsyncMock()
        mock_asr_client.text_to_speech.return_value = b"audio_data"

        with patch.object(client, '_get_asr_client', return_value=mock_asr_client):
            result = await client.text_to_speech_multimodal(
                text="Привет мир",
                voice="elena",
                language="ru"
            )

            assert result == b"audio_data"

            mock_asr_client.text_to_speech.assert_called_once_with(
                text="Привет мир",
                voice="elena",
                language="ru",
                output_file=None
            )

    @pytest.mark.asyncio
    async def test_voice_activity_detection_multimodal(self, client, temp_audio_file):
        """Тест VAD через мультимодальный клиент"""
        mock_vad_result = VADResult(
            speech_segments=[(0.0, 5.0)],
            total_speech_duration=5.0,
            total_silence_duration=0.0,
            speech_ratio=1.0
        )

        mock_asr_client = Mock()
        mock_asr_client.voice_activity_detection = AsyncMock()
        mock_asr_client.voice_activity_detection.return_value = mock_vad_result

        with patch.object(client, '_get_asr_client', return_value=mock_asr_client):
            result = await client.voice_activity_detection_multimodal(temp_audio_file)

            assert result["file_path"] == str(temp_audio_file)
            assert result["vad_result"] == mock_vad_result
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_speaker_diarization_multimodal(self, client, temp_audio_file):
        """Тест диаризации спикеров через мультимодальный клиент"""
        mock_diarization_result = {
            "total_speakers": 2,
            "speakers": [{"speaker_id": "speaker_1"}, {"speaker_id": "speaker_2"}]
        }

        mock_asr_client = Mock()
        mock_asr_client.speaker_diarization = AsyncMock()
        mock_asr_client.speaker_diarization.return_value = mock_diarization_result

        with patch.object(client, '_get_asr_client', return_value=mock_asr_client):
            result = await client.speaker_diarization_multimodal(
                temp_audio_file,
                num_speakers=2
            )

            assert result["file_path"] == str(temp_audio_file)
            assert result["diarization_result"] == mock_diarization_result
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_multimodal_with_speech_analysis(self, client, temp_audio_file):
        """Тест комплексного мультимодального анализа с речью"""
        # Создаем временное изображение
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"fake_png_data")
            temp_image = Path(f.name)

        try:
            # Мокируем методы
            with patch.object(client, 'mixed_media_completion', new_callable=AsyncMock) as mock_visual, \
                    patch.object(client, 'speech_to_text_advanced', new_callable=AsyncMock) as mock_speech:

                mock_visual.return_value = "Анализ изображения завершен"
                mock_speech.return_value = {
                    "transcription": {
                        "text": "Привет мир",
                        "confidence": 0.95,
                        "speakers": [{"speaker_id": "speaker_1"}]
                    },
                    "emotions": EmotionAnalysis(
                        dominant_emotion="happy",
                        emotions={"happy": 0.7},
                        confidence=0.8,
                        valence=0.5,
                        arousal=0.3
                    )
                }

                media_files = {
                    "images": [temp_image],
                    "audio": [temp_audio_file]
                }

                result = await client.multimodal_with_speech_analysis(
                    text_prompt="Проанализируй медиа",
                    media_files=media_files
                )

                assert result["text_prompt"] == "Проанализируй медиа"
                assert "media_analysis" in result
                assert "speech_analysis" in result
                assert "combined_insights" in result

                # Проверяем что визуальный анализ был вызван
                mock_visual.assert_called_once()

                # Проверяем что речевой анализ был вызван
                mock_speech.assert_called_once()

                # Проверяем структуру результата
                assert result["speech_analysis"]["files_count"] == 1
                assert len(result["speech_analysis"]["results"]) == 1

                # Проверяем комбинированные инсайты
                insights = result["combined_insights"]
                assert "summary" in insights
                assert "key_findings" in insights
                assert "confidence_scores" in insights

        finally:
            if temp_image.exists():
                temp_image.unlink()

    @pytest.mark.asyncio
    async def test_multimodal_with_speech_analysis_audio_only(self, client, temp_audio_file):
        """Тест мультимодального анализа только с аудио"""
        with patch.object(client, 'speech_to_text_advanced', new_callable=AsyncMock) as mock_speech:
            mock_speech.return_value = {
                "transcription": {
                    "text": "Только аудио",
                    "confidence": 0.9,
                    "speakers": []
                }
            }

            media_files = {"audio": [temp_audio_file]}

            result = await client.multimodal_with_speech_analysis(
                text_prompt="Анализ аудио",
                media_files=media_files
            )

            assert "speech_analysis" in result
            assert "media_analysis" in result
            # Визуальный анализ не должен быть выполнен
            assert result["media_analysis"] == {}

    def test_generate_combined_insights(self, client):
        """Тест генерации комбинированных инсайтов"""
        media_analysis = {
            "visual": "Анализ изображения завершен"
        }

        speech_analysis = {
            "results": [
                {
                    "transcription": {
                        "confidence": 0.9,
                        "speakers": [{"speaker_id": "speaker_1"}, {"speaker_id": "speaker_2"}]
                    },
                    "emotions": EmotionAnalysis(
                        dominant_emotion="happy",
                        emotions={"happy": 0.7},
                        confidence=0.8,
                        valence=0.5,
                        arousal=0.3
                    )
                }
            ]
        }

        insights = client._generate_combined_insights(
            media_analysis, speech_analysis, "Тестовый промпт"
        )

        assert "summary" in insights
        assert "key_findings" in insights
        assert "recommendations" in insights
        assert "confidence_scores" in insights

        # Проверяем что найдены спикеры
        assert any(
            "2 уникальных спикеров" in finding for finding in insights["key_findings"])

        # Проверяем что есть информация об эмоциях
        assert any("happy" in finding for finding in insights["key_findings"])

        # Проверяем оценки уверенности
        assert "speech_recognition" in insights["confidence_scores"]
        assert "visual_analysis" in insights["confidence_scores"]


if __name__ == "__main__":
    pytest.main([__file__])
