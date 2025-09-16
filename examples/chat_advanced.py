#!/usr/bin/env python3
"""
Продвинутый терминальный чат с реальным голосовым вводом
"""
import asyncio
import sys
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent / "src"))

from kraken_llm import create_universal_client, UniversalCapability, LLMConfig, ReasoningConfig

try:
    import pyaudio
    import wave
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

def record_audio(duration=5, sample_rate=16000):
    """Записать аудио с микрофона"""
    if not AUDIO_AVAILABLE:
        return None
    
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    
    p = pyaudio.PyAudio()
    stream = p.open(format=format, channels=channels, rate=sample_rate,
                    input=True, frames_per_buffer=chunk)
    
    print(f"🎤 Запись {duration} сек...")
    frames = []
    for _ in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Сохраняем во временный файл
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wf = wave.open(temp_file.name, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    return temp_file.name

async def main():
    """Главная функция чата"""
    print("🤖 Kraken LLM Advanced Chat")
    print("Команды: 'v' - голос, 'q' - выход")
    if not AUDIO_AVAILABLE:
        print("⚠️ PyAudio не установлен - голосовой ввод недоступен")
    print()
    
    config = LLMConfig()
    
    async with create_universal_client(
        config=config,
        capabilities={
            UniversalCapability.CHAT_COMPLETION,
            UniversalCapability.STREAMING,
            UniversalCapability.REASONING,
            UniversalCapability.STT
        }
    ) as client:
        
        while True:
            try:
                user_input = input("👤 ").strip()
                
                if user_input.lower() in ['q', 'quit', 'exit']:
                    break
                elif user_input.lower() == 'v' and AUDIO_AVAILABLE:
                    audio_file = record_audio()
                    if audio_file:
                        try:
                            result = await client.speech_to_text(audio_file)
                            user_input = result.get('text', '')
                            print(f"🎤 Распознано: {user_input}")
                            os.unlink(audio_file)
                        except:
                            print("❌ Ошибка распознавания речи")
                            continue
                
                if not user_input:
                    continue
                
                print("🤖 ", end="", flush=True)
                async for chunk in client.chat_completion_stream([
                    {"role": "user", "content": user_input}
                ], max_tokens=8192):
                    print(chunk, end="", flush=True)
                print("\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n❌ {e}\n")

if __name__ == "__main__":
    asyncio.run(main())