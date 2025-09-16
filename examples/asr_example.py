#!/usr/bin/env python3
"""
Пример использования ASRClient для работы с речевыми технологиями.

Демонстрирует различные возможности ASR клиента:
- Распознавание речи (Speech-to-Text)
- Генерация речи (Text-to-Speech)
- Детекция речевой активности (VAD)
- Диаризация спикеров
- Анализ эмоций в речи
- Интеграция с мультимодальным клиентом
"""

import asyncio
import tempfile
from pathlib import Path
import logging
import os
from dotenv import load_dotenv

from kraken_llm.client.asr import ASRClient, ASRConfig
from kraken_llm.client.multimodal import MultimodalLLMClient, MultimodalConfig
from kraken_llm.config.settings import LLMConfig

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Создаем конфигурацию
config = LLMConfig(
    endpoint=os.getenv("ASR_ENDPOINT"),
    api_key=os.getenv("ASR_TOKEN"),
    model=os.getenv("ASR_MODEL")
)

def create_test_audio_file(filename: str, duration_seconds: int = 5) -> Path:
    """
    Создает тестовый WAV файл с минимальными данными.
    
    Args:
        filename: Имя файла
        duration_seconds: Длительность в секундах
        
    Returns:
        Путь к созданному файлу
    """
    # Создаем минимальный WAV файл
    sample_rate = 16000
    samples_count = sample_rate * duration_seconds
    
    # WAV заголовок
    wav_header = (
        b'RIFF'
        + (36 + samples_count * 2).to_bytes(4, 'little')  # Размер файла - 8
        + b'WAVE'
        + b'fmt '
        + (16).to_bytes(4, 'little')  # Размер fmt chunk
        + (1).to_bytes(2, 'little')   # Аудио формат (PCM)
        + (1).to_bytes(2, 'little')   # Количество каналов (моно)
        + sample_rate.to_bytes(4, 'little')  # Частота дискретизации
        + (sample_rate * 2).to_bytes(4, 'little')  # Байт в секунду
        + (2).to_bytes(2, 'little')   # Выравнивание блока
        + (16).to_bytes(2, 'little')  # Бит на семпл
        + b'data'
        + (samples_count * 2).to_bytes(4, 'little')  # Размер данных
    )
    
    # Простые аудио данные (тишина с небольшим шумом)
    audio_data = b'\x00\x00' * samples_count
    
    temp_path = Path(tempfile.gettempdir()) / filename
    with open(temp_path, 'wb') as f:
        f.write(wav_header + audio_data)
    
    return temp_path


async def demo_basic_asr_functionality():
    """Демонстрация базовой функциональности ASR"""
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦИЯ БАЗОВОЙ ФУНКЦИОНАЛЬНОСТИ ASR")
    print("="*60)
    
    asr_config = ASRConfig(
        default_language="ru",
        enable_timestamps=True,
        enable_speaker_diarization=True,
        max_speakers=5,
        quality_level="high"
    )
    
    # Создаем ASR клиент
    client = ASRClient(config, asr_config)
    
    # Создаем тестовый аудио файл
    audio_file = create_test_audio_file("test_speech.wav", duration_seconds=10)
    
    try:
        print(f"\nСоздан тестовый аудио файл: {audio_file}")
        print(f"Размер файла: {audio_file.stat().st_size} байт")
        
        # 1. Информация о конфигурации
        print("\n1. Конфигурация ASR клиента:")
        print("-" * 40)
        
        config_summary = client.get_config_summary()
        for key, value in config_summary.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for sub_key, sub_value in value.items():
                    print(f"  {sub_key}: {sub_value}")
            else:
                print(f"{key}: {value}")
        
        # 2. Поддерживаемые языки и голоса
        print("\n2. Поддерживаемые возможности:")
        print("-" * 40)
        
        languages = ASRClient.get_supported_languages()
        print(f"Языки ({len(languages)}): {', '.join(languages[:10])}...")
        
        voices = ASRClient.get_supported_voices()
        print(f"Голоса для русского: {', '.join(voices.get('ru', []))}")
        print(f"Голоса для английского: {', '.join(voices.get('en', []))}")
        
        emotions = ASRClient.get_supported_emotions()
        print(f"Эмоции ({len(emotions)}): {', '.join(emotions)}")
        
        # 3. Базовое распознавание речи
        print("\n3. Базовое распознавание речи:")
        print("-" * 40)
        
        try:
            stt_result = await client.speech_to_text(
                audio_file,
                language="ru",
                enable_diarization=True,
                enable_emotions=True
            )
            
            print(f"Распознанный текст: {stt_result.get('text', 'Не распознано')}")
            print(f"Уверенность: {stt_result.get('confidence', 0.0):.2f}")
            print(f"Длительность: {stt_result.get('duration', 0.0):.1f} сек")
            print(f"Язык: {stt_result.get('language', 'не определен')}")
            
            if 'segments' in stt_result:
                print(f"Сегментов: {len(stt_result['segments'])}")
            
            if 'speakers' in stt_result:
                print(f"Спикеров: {len(stt_result['speakers'])}")
                
        except Exception as e:
            print(f"Ошибка распознавания речи: {e}")
            print("Возможно, LLM сервер не поддерживает ASR функции")
        
        # 4. Синтез речи
        print("\n4. Синтез речи:")
        print("-" * 40)
        
        try:
            tts_result = await client.text_to_speech(
                text="Привет! Это тест синтеза речи на русском языке.",
                voice="elena",
                language="ru"
            )
            
            if isinstance(tts_result, bytes):
                print(f"Синтез завершен: {len(tts_result)} байт аудио данных")
            else:
                print(f"Аудио сохранено в файл: {tts_result}")
                
        except Exception as e:
            print(f"Ошибка синтеза речи: {e}")
            print("Возможно, LLM сервер не поддерживает TTS функции")
        
        # 5. Детекция речевой активности
        print("\n5. Детекция речевой активности (VAD):")
        print("-" * 40)
        
        try:
            vad_result = await client.voice_activity_detection(audio_file)
            
            print(f"Сегментов речи: {len(vad_result.speech_segments)}")
            print(f"Общая длительность речи: {vad_result.total_speech_duration:.1f} сек")
            print(f"Общая длительность тишины: {vad_result.total_silence_duration:.1f} сек")
            print(f"Доля речи: {vad_result.speech_ratio:.2%}")
            
            for i, (start, end) in enumerate(vad_result.speech_segments[:3]):
                print(f"  Сегмент {i+1}: {start:.1f}s - {end:.1f}s")
                
        except Exception as e:
            print(f"Ошибка VAD анализа: {e}")
        
        # 6. Диаризация спикеров
        print("\n6. Диаризация спикеров:")
        print("-" * 40)
        
        try:
            diarization_result = await client.speaker_diarization(
                audio_file, 
                num_speakers=3
            )
            
            print(f"Найдено спикеров: {diarization_result['total_speakers']}")
            print(f"Общая длительность: {diarization_result.get('total_duration', 0.0):.1f} сек")
            
            for speaker in diarization_result['speakers'][:3]:
                print(f"  {speaker.speaker_id}: {speaker.total_duration:.1f}s, "
                      f"{speaker.segments_count} сегментов, "
                      f"уверенность: {speaker.confidence:.2f}")
                
        except Exception as e:
            print(f"Ошибка диаризации: {e}")
        
        # 7. Анализ эмоций
        print("\n7. Анализ эмоций в речи:")
        print("-" * 40)
        
        try:
            emotion_result = await client.emotion_analysis(audio_file)
            
            print(f"Доминирующая эмоция: {emotion_result.dominant_emotion}")
            print(f"Уверенность: {emotion_result.confidence:.2f}")
            print(f"Валентность: {emotion_result.valence:.2f}")
            print(f"Возбуждение: {emotion_result.arousal:.2f}")
            
            print("Распределение эмоций:")
            for emotion, score in emotion_result.emotions.items():
                print(f"  {emotion}: {score:.2%}")
                
        except Exception as e:
            print(f"Ошибка анализа эмоций: {e}")
        
    finally:
        # Очистка
        if audio_file.exists():
            audio_file.unlink()


async def demo_comprehensive_analysis():
    """Демонстрация комплексного анализа аудио"""
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦИЯ КОМПЛЕКСНОГО АНАЛИЗА АУДИО")
    print("="*60)
    
    client = ASRClient(config)
    
    # Создаем несколько тестовых файлов
    audio_files = []
    for i in range(2):
        audio_file = create_test_audio_file(f"comprehensive_test_{i}.wav", duration_seconds=8)
        audio_files.append(audio_file)
    
    try:
        print(f"\nСоздано {len(audio_files)} тестовых аудио файлов")
        
        # Комплексный анализ каждого файла
        for i, audio_file in enumerate(audio_files):
            print(f"\n{i+1}. Анализ файла: {audio_file.name}")
            print("-" * 50)
            
            try:
                result = await client.transcribe_with_analysis(
                    audio_file,
                    include_diarization=True,
                    include_emotions=True,
                    include_vad=True,
                    language="ru"
                )
                
                # Транскрипция
                if 'transcription' in result:
                    trans = result['transcription']
                    print(f"Текст: {trans.get('text', 'Не распознано')}")
                    print(f"Уверенность: {trans.get('confidence', 0.0):.2f}")
                
                # VAD
                if 'voice_activity' in result and result['voice_activity']:
                    vad = result['voice_activity']
                    print(f"Доля речи: {vad.speech_ratio:.2%}")
                
                # Диаризация
                if 'diarization' in result and result['diarization']:
                    diar = result['diarization']
                    print(f"Спикеров: {diar.get('total_speakers', 0)}")
                
                # Эмоции
                if 'emotions' in result and result['emotions']:
                    emotions = result['emotions']
                    print(f"Эмоция: {emotions.dominant_emotion}")
                
                # Сводка
                if 'summary' in result:
                    summary = result['summary']
                    print(f"Сводка: длительность {summary.get('total_duration', 0.0):.1f}s, "
                          f"эмоция {summary.get('dominant_emotion', 'neutral')}")
                
            except Exception as e:
                print(f"Ошибка анализа файла {i+1}: {e}")
    
    finally:
        # Очистка
        for audio_file in audio_files:
            if audio_file.exists():
                audio_file.unlink()


async def demo_multimodal_integration():
    """Демонстрация интеграции ASR с мультимодальным клиентом"""
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦИЯ ИНТЕГРАЦИИ С МУЛЬТИМОДАЛЬНЫМ КЛИЕНТОМ")
    print("="*60)
    
    multimodal_config = MultimodalConfig()
    
    # Создаем мультимодальный клиент
    client = MultimodalLLMClient(config, multimodal_config)
    
    # Создаем тестовые файлы
    audio_file = create_test_audio_file("multimodal_test.wav", duration_seconds=6)
    
    try:
        print(f"\nСоздан тестовый аудио файл: {audio_file.name}")
        
        # 1. Продвинутое распознавание речи через мультимодальный клиент
        print("\n1. Продвинутое распознавание речи:")
        print("-" * 40)
        
        try:
            result = await client.speech_to_text_advanced(
                audio_file,
                language="ru",
                include_diarization=True,
                include_emotions=True,
                include_vad=True
            )
            
            print(f"Файл: {result.get('file_path', 'неизвестно')}")
            
            if 'transcription' in result:
                trans = result['transcription']
                print(f"Распознанный текст: {trans.get('text', 'Не распознано')}")
                print(f"Уверенность: {trans.get('confidence', 0.0):.2f}")
            
            if 'voice_activity' in result and result['voice_activity']:
                vad = result['voice_activity']
                print(f"Активность речи: {vad.speech_ratio:.2%}")
            
            if 'summary' in result:
                summary = result['summary']
                print(f"Длительность: {summary.get('total_duration', 0.0):.1f} сек")
                
        except Exception as e:
            print(f"Ошибка продвинутого распознавания: {e}")
        
        # 2. Синтез речи через мультимодальный клиент
        print("\n2. Синтез речи через мультимодальный клиент:")
        print("-" * 50)
        
        try:
            tts_result = await client.text_to_speech_multimodal(
                text="Это тест синтеза речи через мультимодальный клиент",
                voice="elena",
                language="ru"
            )
            
            if isinstance(tts_result, bytes):
                print(f"Синтез завершен: {len(tts_result)} байт")
            else:
                print(f"Файл сохранен: {tts_result}")
                
        except Exception as e:
            print(f"Ошибка синтеза через мультимодальный клиент: {e}")
        
        # 3. VAD через мультимодальный клиент
        print("\n3. VAD через мультимодальный клиент:")
        print("-" * 40)
        
        try:
            vad_result = await client.voice_activity_detection_multimodal(audio_file)
            
            if vad_result.get('success'):
                vad = vad_result['vad_result']
                print(f"Сегментов речи: {len(vad.speech_segments)}")
                print(f"Доля речи: {vad.speech_ratio:.2%}")
            else:
                print(f"Ошибка VAD: {vad_result.get('error', 'неизвестная ошибка')}")
                
        except Exception as e:
            print(f"Ошибка VAD через мультимодальный клиент: {e}")
        
        # 4. Диаризация через мультимодальный клиент
        print("\n4. Диаризация через мультимодальный клиент:")
        print("-" * 45)
        
        try:
            diar_result = await client.speaker_diarization_multimodal(
                audio_file, 
                num_speakers=2
            )
            
            if diar_result.get('success'):
                diar = diar_result['diarization_result']
                print(f"Найдено спикеров: {diar.get('total_speakers', 0)}")
            else:
                print(f"Ошибка диаризации: {diar_result.get('error', 'неизвестная ошибка')}")
                
        except Exception as e:
            print(f"Ошибка диаризации через мультимодальный клиент: {e}")
    
    finally:
        # Очистка
        if audio_file.exists():
            audio_file.unlink()


async def demo_streaming_capabilities():
    """Демонстрация потоковых возможностей (заглушка)"""
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦИЯ ПОТОКОВЫХ ВОЗМОЖНОСТЕЙ")
    print("="*60)
    
    print("\nПотоковое распознавание речи:")
    print("-" * 30)
    print("Функция streaming_speech_to_text реализована как заглушка")
    print("для будущего развития. В текущей версии используется")
    print("пакетная обработка аудио файлов.")
    
    print("\nДля реального потокового распознавания потребуется:")
    print("- WebSocket соединение с ASR сервером")
    print("- Буферизация аудио данных")
    print("- Обработка частичных результатов")
    print("- Управление сессиями")


async def main():
    """Главная функция демонстрации"""
    print("ДЕМОНСТРАЦИЯ ASR CLIENT")
    print("=" * 80)
    print("Этот пример показывает возможности ASR клиента Kraken")
    print("для работы с речевыми технологиями и интеграции")
    print("с мультимодальным клиентом.")
    print()
    
    try:
        # Демонстрация различных возможностей
        await demo_basic_asr_functionality()
        await demo_comprehensive_analysis()
        await demo_multimodal_integration()
        await demo_streaming_capabilities()
        
        print("\n" + "="*80)
        print("ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА УСПЕШНО!")
        print("="*80)
        print("\nASR клиент готов к использованию для:")
        print("- Распознавания речи с диаризацией спикеров")
        print("- Синтеза речи с различными голосами")
        print("- Детекции речевой активности")
        print("- Анализа эмоций в речи")
        print("- Интеграции с мультимодальными запросами")
        
    except KeyboardInterrupt:
        print("\nДемонстрация прервана пользователем")
    except Exception as e:
        print(f"\nОшибка во время демонстрации: {e}")
        logger.exception("Подробности ошибки:")


if __name__ == "__main__":
    # Запуск демонстрации
    asyncio.run(main())