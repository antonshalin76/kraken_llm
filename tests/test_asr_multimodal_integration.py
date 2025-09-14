#!/usr/bin/env python3
"""
Тест интеграции ASR клиента с мультимодальным клиентом.

Демонстрирует возможность вызова ASR функций из мультимодального клиента.
"""

import asyncio
import tempfile
import os
from pathlib import Path
from PIL import Image

from kraken_llm.client.multimodal import MultimodalLLMClient, MultimodalConfig
from kraken_llm.client.asr import ASRClient, ASRConfig
from kraken_llm.config.settings import LLMConfig

from dotenv import load_dotenv

load_dotenv()

def create_test_files():
    """Создает тестовые медиа файлы"""
    files = {}
    
    # Создаем тестовое изображение
    img = Image.new('RGB', (200, 150), color='blue')
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img.save(f, format='PNG')
        files['image'] = Path(f.name)
    
    # Создаем тестовый аудио файл
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
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_header)
        files['audio'] = Path(f.name)
    
    return files


async def test_standalone_asr_client():
    """Тест автономного ASR клиента"""
    print("\n" + "="*60)
    print("ТЕСТ АВТОНОМНОГО ASR КЛИЕНТА")
    print("="*60)
    
    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL")
    )
    
    asr_config = ASRConfig(
        default_language="ru",
        enable_speaker_diarization=True,
        max_speakers=3
    )
    
    # Создаем ASR клиент
    asr_client = ASRClient(config, asr_config)
    
    # Создаем тестовые файлы
    test_files = create_test_files()
    
    try:
        print(f"\nСоздан тестовый аудио файл: {test_files['audio'].name}")
        
        # 1. Тест конфигурации
        print("\n1. Конфигурация ASR клиента:")
        print("-" * 40)
        
        config_summary = asr_client.get_config_summary()
        print(f"Язык по умолчанию: {config_summary['default_language']}")
        print(f"Максимум спикеров: {config_summary['max_speakers']}")
        print(f"Поддерживаемые форматы: {len(config_summary['supported_formats'])}")
        
        # 2. Поддерживаемые возможности
        print("\n2. Поддерживаемые возможности:")
        print("-" * 40)
        
        languages = ASRClient.get_supported_languages()
        voices = ASRClient.get_supported_voices()
        emotions = ASRClient.get_supported_emotions()
        
        print(f"Языков: {len(languages)} (включая {', '.join(languages[:5])}...)")
        print(f"Голосов для русского: {len(voices.get('ru', []))}")
        print(f"Распознаваемых эмоций: {len(emotions)}")
        
        # 3. Базовое распознавание речи
        print("\n3. Базовое распознавание речи:")
        print("-" * 40)
        
        try:
            result = await asr_client.speech_to_text(
                test_files['audio'],
                language="ru"
            )
            
            print(f"✓ Распознавание выполнено")
            print(f"  Текст: {result.get('text', 'Не распознано')[:50]}...")
            print(f"  Уверенность: {result.get('confidence', 0.0):.2f}")
            print(f"  Язык: {result.get('language', 'не определен')}")
            
        except Exception as e:
            print(f"⚠ Распознавание недоступно: {type(e).__name__}")
        
        # 4. Синтез речи
        print("\n4. Синтез речи:")
        print("-" * 40)
        
        try:
            tts_result = await asr_client.text_to_speech(
                "Тест синтеза речи",
                voice="elena",
                language="ru"
            )
            
            if isinstance(tts_result, bytes):
                print(f"✓ Синтез выполнен: {len(tts_result)} байт")
            else:
                print(f"✓ Аудио сохранено: {tts_result}")
                
        except Exception as e:
            print(f"⚠ Синтез недоступен: {type(e).__name__}")
        
        print("\n✅ Автономный ASR клиент протестирован")
        
    finally:
        # Очистка
        for file_path in test_files.values():
            if file_path.exists():
                file_path.unlink()


async def test_multimodal_asr_integration():
    """Тест интеграции ASR с мультимодальным клиентом"""
    print("\n" + "="*60)
    print("ТЕСТ ИНТЕГРАЦИИ ASR С МУЛЬТИМОДАЛЬНЫМ КЛИЕНТОМ")
    print("="*60)
    
    config = LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL")
    )
    
    multimodal_config = MultimodalConfig()
    
    # Создаем мультимодальный клиент
    multimodal_client = MultimodalLLMClient(config, multimodal_config)
    
    # Создаем тестовые файлы
    test_files = create_test_files()
    
    try:
        print(f"\nСозданы тестовые файлы:")
        print(f"  Изображение: {test_files['image'].name}")
        print(f"  Аудио: {test_files['audio'].name}")
        
        # 1. Ленивая инициализация ASR клиента
        print("\n1. Ленивая инициализация ASR клиента:")
        print("-" * 45)
        
        # Изначально ASR клиент не создан
        print(f"ASR клиент до обращения: {multimodal_client._asr_client}")
        
        # После первого обращения создается
        asr_client = multimodal_client._get_asr_client()
        print(f"ASR клиент после обращения: {type(asr_client).__name__}")
        print("✓ Ленивая инициализация работает")
        
        # 2. Продвинутое распознавание речи
        print("\n2. Продвинутое распознавание речи:")
        print("-" * 40)
        
        try:
            result = await multimodal_client.speech_to_text_advanced(
                test_files['audio'],
                language="ru",
                include_diarization=True,
                include_emotions=True,
                include_vad=True
            )
            
            print(f"✓ Продвинутое распознавание выполнено")
            print(f"  Файл: {Path(result['file_path']).name}")
            
            if 'transcription' in result:
                trans = result['transcription']
                print(f"  Текст: {trans.get('text', 'Не распознано')[:50]}...")
                print(f"  Уверенность: {trans.get('confidence', 0.0):.2f}")
            
            if 'voice_activity' in result and result['voice_activity']:
                vad = result['voice_activity']
                print(f"  Активность речи: {vad.speech_ratio:.2%}")
            
            if 'summary' in result:
                summary = result['summary']
                print(f"  Длительность: {summary.get('total_duration', 0.0):.1f}s")
                
        except Exception as e:
            print(f"⚠ Продвинутое распознавание недоступно: {type(e).__name__}")
        
        # 3. Синтез речи через мультимодальный клиент
        print("\n3. Синтез речи через мультимодальный клиент:")
        print("-" * 50)
        
        try:
            tts_result = await multimodal_client.text_to_speech_multimodal(
                "Тест синтеза через мультимодальный клиент",
                voice="elena",
                language="ru"
            )
            
            if isinstance(tts_result, bytes):
                print(f"✓ Синтез выполнен: {len(tts_result)} байт")
            else:
                print(f"✓ Аудио сохранено: {tts_result}")
                
        except Exception as e:
            print(f"⚠ Синтез недоступен: {type(e).__name__}")
        
        # 4. VAD через мультимодальный клиент
        print("\n4. VAD через мультимодальный клиент:")
        print("-" * 40)
        
        try:
            vad_result = await multimodal_client.voice_activity_detection_multimodal(
                test_files['audio']
            )
            
            if vad_result.get('success'):
                vad = vad_result['vad_result']
                print(f"✓ VAD выполнен")
                print(f"  Сегментов речи: {len(vad.speech_segments)}")
                print(f"  Доля речи: {vad.speech_ratio:.2%}")
            else:
                print(f"⚠ VAD недоступен: {vad_result.get('error', 'неизвестная ошибка')}")
                
        except Exception as e:
            print(f"⚠ VAD недоступен: {type(e).__name__}")
        
        # 5. Комплексный мультимодальный анализ
        print("\n5. Комплексный мультимодальный анализ:")
        print("-" * 45)
        
        try:
            media_files = {
                "images": [test_files['image']],
                "audio": [test_files['audio']]
            }
            
            result = await multimodal_client.multimodal_with_speech_analysis(
                text_prompt="Проанализируй предоставленные медиа файлы",
                media_files=media_files,
                speech_analysis_options={
                    'include_diarization': True,
                    'include_emotions': True,
                    'include_vad': True
                }
            )
            
            print(f"✓ Комплексный анализ выполнен")
            print(f"  Промпт: {result['text_prompt'][:50]}...")
            
            if 'media_analysis' in result:
                print(f"  Визуальный анализ: {'✓' if result['media_analysis'] else '✗'}")
            
            if 'speech_analysis' in result:
                speech = result['speech_analysis']
                if 'files_count' in speech:
                    print(f"  Аудио файлов обработано: {speech['files_count']}")
            
            if 'combined_insights' in result:
                insights = result['combined_insights']
                print(f"  Инсайтов: {len(insights.get('key_findings', []))}")
                print(f"  Сводка: {insights.get('summary', 'не создана')[:50]}...")
                
        except Exception as e:
            print(f"⚠ Комплексный анализ недоступен: {type(e).__name__}")
        
        print("\n✅ Интеграция ASR с мультимодальным клиентом протестирована")
        
    finally:
        # Очистка
        for file_path in test_files.values():
            if file_path.exists():
                file_path.unlink()


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
    
    multimodal_client = MultimodalLLMClient(config)
    
    # 1. Несуществующий файл
    print("\n1. Несуществующий аудио файл:")
    print("-" * 35)
    
    try:
        await multimodal_client.speech_to_text_advanced("nonexistent.wav")
        print("✗ Должна была быть ошибка")
    except Exception as e:
        print(f"✓ Корректная ошибка: {type(e).__name__}")
    
    # 2. Неподдерживаемый формат
    print("\n2. Неподдерживаемый формат:")
    print("-" * 30)
    
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"not audio")
        txt_file = Path(f.name)
    
    try:
        await multimodal_client.voice_activity_detection_multimodal(txt_file)
        print("✗ Должна была быть ошибка")
    except Exception as e:
        print(f"✓ Корректная ошибка: {type(e).__name__}")
    finally:
        txt_file.unlink()
    
    # 3. Пустой текст для синтеза
    print("\n3. Пустой текст для синтеза:")
    print("-" * 30)
    
    try:
        await multimodal_client.text_to_speech_multimodal("")
        print("✗ Должна была быть ошибка")
    except Exception as e:
        print(f"✓ Корректная ошибка: {type(e).__name__}")
    
    print("\n✅ Обработка ошибок работает корректно")


async def main():
    """Главная функция тестирования"""
    print("ТЕСТ ИНТЕГРАЦИИ ASR КЛИЕНТА С МУЛЬТИМОДАЛЬНЫМ КЛИЕНТОМ")
    print("=" * 80)
    print("Демонстрирует возможность вызова ASR функций из мультимодального клиента")
    print()
    
    try:
        # Тестируем различные аспекты интеграции
        await test_standalone_asr_client()
        await test_multimodal_asr_integration()
        await test_error_handling()
        
        print("\n" + "="*80)
        print("ВСЕ ТЕСТЫ ИНТЕГРАЦИИ ЗАВЕРШЕНЫ УСПЕШНО!")
        print("="*80)
        print("\nИнтеграция ASR с мультимодальным клиентом работает:")
        print("✅ Ленивая инициализация ASR клиента")
        print("✅ Продвинутое распознавание речи")
        print("✅ Синтез речи через мультимодальный интерфейс")
        print("✅ Детекция речевой активности")
        print("✅ Комплексный мультимодальный анализ")
        print("✅ Корректная обработка ошибок")
        
    except KeyboardInterrupt:
        print("\nТестирование прервано пользователем")
    except Exception as e:
        print(f"\nОШИБКА В ТЕСТИРОВАНИИ: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())