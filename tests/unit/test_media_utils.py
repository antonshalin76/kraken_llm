"""
Тесты для утилит работы с медиа файлами.
"""

import pytest
import tempfile
from pathlib import Path
from PIL import Image
import base64

from kraken_llm.utils.media import (
    MediaUtils, 
    encode_image_to_base64,
    create_image_data_url,
    validate_image,
    get_image_dimensions
)


class TestMediaUtils:
    """Тесты для класса MediaUtils"""
    
    @pytest.fixture
    def temp_image(self):
        """Создает временное изображение для тестов"""
        img = Image.new('RGB', (100, 100), color='red')
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f, format='PNG')
            temp_path = Path(f.name)
        
        yield temp_path
        
        if temp_path.exists():
            temp_path.unlink()
    
    @pytest.fixture
    def temp_large_image(self):
        """Создает большое временное изображение"""
        img = Image.new('RGB', (2000, 1500), color='blue')
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img.save(f, format='JPEG', quality=95)
            temp_path = Path(f.name)
        
        yield temp_path
        
        if temp_path.exists():
            temp_path.unlink()
    
    @pytest.fixture
    def temp_audio(self):
        """Создает временный аудио файл"""
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
            temp_path = Path(f.name)
        
        yield temp_path
        
        if temp_path.exists():
            temp_path.unlink()
    
    def test_encode_file_to_base64(self, temp_image):
        """Тест кодирования файла в base64"""
        result = MediaUtils.encode_file_to_base64(temp_image)
        
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Проверяем что это валидный base64
        try:
            decoded = base64.b64decode(result)
            assert len(decoded) > 0
        except Exception:
            pytest.fail("Результат не является валидным base64")
    
    def test_encode_file_to_base64_not_found(self):
        """Тест кодирования несуществующего файла"""
        with pytest.raises(FileNotFoundError):
            MediaUtils.encode_file_to_base64("nonexistent.png")
    
    def test_get_file_info(self, temp_image):
        """Тест получения информации о файле"""
        info = MediaUtils.get_file_info(temp_image)
        
        assert 'filename' in info
        assert 'size' in info
        assert 'extension' in info
        assert 'mime_type' in info
        assert 'media_type' in info
        
        assert info['extension'] == 'png'
        assert info['mime_type'] == 'image/png'
        assert info['media_type'] == 'image'
        assert info['size'] > 0
    
    def test_get_file_info_audio(self, temp_audio):
        """Тест получения информации об аудио файле"""
        info = MediaUtils.get_file_info(temp_audio)
        
        assert info['extension'] == 'wav'
        assert info['media_type'] == 'audio'
        assert info['mime_type'] in ['audio/wav', 'audio/x-wav']
    
    def test_validate_file_format(self, temp_image):
        """Тест валидации формата файла"""
        # Валидный формат
        assert MediaUtils.validate_file_format(temp_image, ['png', 'jpg'])
        assert MediaUtils.validate_file_format(temp_image)  # Без ограничений
        
        # Невалидный формат
        assert not MediaUtils.validate_file_format(temp_image, ['jpg', 'gif'])
    
    def test_validate_file_size(self, temp_image):
        """Тест валидации размера файла"""
        file_size = temp_image.stat().st_size
        
        # Размер в пределах лимита
        assert MediaUtils.validate_file_size(temp_image, file_size + 100)
        
        # Размер превышает лимит
        assert not MediaUtils.validate_file_size(temp_image, file_size - 1)
    
    def test_get_image_info(self, temp_image):
        """Тест получения информации об изображении"""
        info = MediaUtils.get_image_info(temp_image)
        
        assert 'width' in info
        assert 'height' in info
        assert 'format' in info
        assert 'mode' in info
        assert 'has_transparency' in info
        assert 'size_bytes' in info
        
        assert info['width'] == 100
        assert info['height'] == 100
        assert info['format'] == 'PNG'
    
    def test_get_image_info_not_found(self):
        """Тест получения информации о несуществующем изображении"""
        with pytest.raises(FileNotFoundError):
            MediaUtils.get_image_info("nonexistent.png")
    
    def test_resize_image(self, temp_image):
        """Тест изменения размера изображения"""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = Path(f.name)
        
        try:
            result = MediaUtils.resize_image(
                temp_image,
                output_path,
                max_width=50,
                max_height=50
            )
            
            assert 'original_size' in result
            assert 'new_size' in result
            assert 'original_file_size' in result
            assert 'new_file_size' in result
            
            assert result['original_size'] == (100, 100)
            assert result['new_size'] == (50, 50)
            assert output_path.exists()
            
            # Проверяем что новое изображение имеет правильный размер
            with Image.open(output_path) as img:
                assert img.size == (50, 50)
                
        finally:
            if output_path.exists():
                output_path.unlink()
    
    def test_create_data_url(self, temp_image):
        """Тест создания data URL"""
        data_url = MediaUtils.create_data_url(temp_image)
        
        assert data_url.startswith('data:image/png;base64,')
        assert len(data_url) > 50
        
        # Извлекаем base64 часть и проверяем
        base64_part = data_url.split(',')[1]
        try:
            decoded = base64.b64decode(base64_part)
            assert len(decoded) > 0
        except Exception:
            pytest.fail("Data URL содержит невалидный base64")
    
    def test_get_supported_formats(self):
        """Тест получения поддерживаемых форматов"""
        formats = MediaUtils.get_supported_formats()
        
        assert 'images' in formats
        assert 'audio' in formats
        assert 'video' in formats
        
        assert 'png' in formats['images']
        assert 'jpg' in formats['images']
        assert 'mp3' in formats['audio']
        assert 'wav' in formats['audio']
        assert 'mp4' in formats['video']
    
    def test_validate_media_file_valid_image(self, temp_image):
        """Тест валидации валидного изображения"""
        result = MediaUtils.validate_media_file(
            temp_image,
            'image',
            max_size=1024 * 1024,  # 1MB
            allowed_formats=['png', 'jpg']
        )
        
        assert result['valid'] is True
        assert len(result['errors']) == 0
        assert 'file_info' in result
        assert result['file_info']['media_type'] == 'image'
    
    def test_validate_media_file_wrong_type(self, temp_audio):
        """Тест валидации файла неправильного типа"""
        result = MediaUtils.validate_media_file(
            temp_audio,
            'image'  # Ожидаем изображение, но передаем аудио
        )
        
        assert result['valid'] is False
        assert len(result['errors']) > 0
        assert any('Неверный тип медиа' in error for error in result['errors'])
    
    def test_validate_media_file_too_large(self, temp_image):
        """Тест валидации слишком большого файла"""
        result = MediaUtils.validate_media_file(
            temp_image,
            'image',
            max_size=100  # Очень маленький лимит
        )
        
        assert result['valid'] is False
        assert any('слишком большой' in error for error in result['errors'])
    
    def test_validate_media_file_unsupported_format(self, temp_image):
        """Тест валидации неподдерживаемого формата"""
        result = MediaUtils.validate_media_file(
            temp_image,
            'image',
            allowed_formats=['jpg', 'gif']  # PNG не разрешен
        )
        
        assert result['valid'] is False
        assert any('Неподдерживаемый формат' in error for error in result['errors'])
    
    def test_batch_process_images(self, temp_image):
        """Тест пакетной обработки изображений"""
        # Создаем временную директорию с изображениями
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            input_dir.mkdir()
            
            # Копируем тестовое изображение несколько раз
            for i in range(3):
                img = Image.new('RGB', (200, 200), color=['red', 'green', 'blue'][i])
                img.save(input_dir / f"test_{i}.png")
            
            # Обрабатываем
            results = MediaUtils.batch_process_images(
                input_dir,
                output_dir,
                max_width=100,
                max_height=100
            )
            
            assert len(results) == 3
            assert all(result['status'] == 'success' for result in results)
            assert output_dir.exists()
            
            # Проверяем что файлы созданы
            output_files = list(output_dir.glob("*.png"))
            assert len(output_files) == 3


class TestConvenienceFunctions:
    """Тесты для удобных функций"""
    
    @pytest.fixture
    def temp_image(self):
        """Создает временное изображение"""
        img = Image.new('RGB', (150, 100), color='green')
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f, format='PNG')
            temp_path = Path(f.name)
        
        yield temp_path
        
        if temp_path.exists():
            temp_path.unlink()
    
    def test_encode_image_to_base64(self, temp_image):
        """Тест быстрого кодирования изображения"""
        result = encode_image_to_base64(temp_image)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_create_image_data_url(self, temp_image):
        """Тест быстрого создания data URL"""
        result = create_image_data_url(temp_image)
        
        assert result.startswith('data:image/png;base64,')
    
    def test_validate_image(self, temp_image):
        """Тест быстрой валидации изображения"""
        assert validate_image(temp_image) is True
        assert validate_image(temp_image, max_size=100) is False  # Слишком маленький лимит
    
    def test_get_image_dimensions(self, temp_image):
        """Тест получения размеров изображения"""
        width, height = get_image_dimensions(temp_image)
        
        assert width == 150
        assert height == 100


if __name__ == "__main__":
    pytest.main([__file__])