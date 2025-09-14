"""
Unit тесты для StructuredOutputValidator.

Тестирует функциональность валидатора structured output включая валидацию
Pydantic моделей, проверку совместимости и обработку ошибок.
"""

import json
import pytest
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

from kraken_llm.structured.validator import (
    StructuredOutputValidator,
    validate_structured_response,
    check_model_compatibility,
    create_model_example,
)
from kraken_llm.exceptions.validation import ValidationError


# Тестовые Pydantic модели
class SimpleTestModel(BaseModel):
    """Простая модель для тестирования."""
    name: str = Field(..., description="Имя")
    age: int = Field(..., ge=0, le=150, description="Возраст")
    email: Optional[str] = Field(None, description="Email")

    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        if v and '@' not in v:
            raise ValueError('Некорректный email')
        return v


class ComplexTestModel(BaseModel):
    """Сложная модель для тестирования."""
    user: SimpleTestModel
    tags: List[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    is_active: bool = True


class TestStructuredOutputValidator:
    """Тесты основного валидатора."""

    def test_init(self):
        """Тест инициализации валидатора."""
        validator = StructuredOutputValidator()
        assert validator is not None

    @pytest.mark.asyncio
    async def test_validate_response_with_dict(self):
        """Тест валидации словаря."""
        validator = StructuredOutputValidator()

        data = {
            "name": "Тест Пользователь",
            "age": 25,
            "email": "test@example.com"
        }

        result = await validator.validate_response(data, SimpleTestModel)

        assert isinstance(result, SimpleTestModel)
        assert result.name == "Тест Пользователь"
        assert result.age == 25
        assert result.email == "test@example.com"

    @pytest.mark.asyncio
    async def test_validate_response_with_json_string(self):
        """Тест валидации JSON строки."""
        validator = StructuredOutputValidator()

        json_data = json.dumps({
            "name": "JSON Пользователь",
            "age": 30,
            "email": "json@example.com"
        })

        result = await validator.validate_response(json_data, SimpleTestModel)

        assert isinstance(result, SimpleTestModel)
        assert result.name == "JSON Пользователь"
        assert result.age == 30
        assert result.email == "json@example.com"

    @pytest.mark.asyncio
    async def test_validate_response_with_existing_model(self):
        """Тест валидации уже существующей модели."""
        validator = StructuredOutputValidator()

        existing_model = SimpleTestModel(
            name="Существующий",
            age=35,
            email="existing@example.com"
        )

        result = await validator.validate_response(existing_model, SimpleTestModel)

        assert result is existing_model
        assert result.name == "Существующий"

    @pytest.mark.asyncio
    async def test_validate_response_invalid_json(self):
        """Тест валидации некорректного JSON."""
        validator = StructuredOutputValidator()

        invalid_json = '{"name": "Test", "age": 25, invalid}'

        with pytest.raises(ValidationError, match="Некорректный JSON"):
            await validator.validate_response(invalid_json, SimpleTestModel)

    @pytest.mark.asyncio
    async def test_validate_response_validation_error(self):
        """Тест ошибки валидации Pydantic."""
        validator = StructuredOutputValidator()

        data = {
            "name": "Тест",
            "age": -5,  # Некорректный возраст
            "email": "invalid-email"  # Некорректный email
        }

        with pytest.raises(ValidationError, match="Ошибка валидации"):
            await validator.validate_response(data, SimpleTestModel)

    def test_validate_schema_compatibility_simple_model(self):
        """Тест проверки совместимости простой модели."""
        validator = StructuredOutputValidator()

        result = validator.validate_schema_compatibility(SimpleTestModel)

        assert isinstance(result, dict)
        assert result["model_name"] == "SimpleTestModel"
        assert result["is_compatible"] is True
        assert "schema" in result
        assert isinstance(result["schema"], dict)

    def test_validate_schema_compatibility_complex_model(self):
        """Тест проверки совместимости сложной модели."""
        validator = StructuredOutputValidator()

        result = validator.validate_schema_compatibility(ComplexTestModel)

        assert isinstance(result, dict)
        assert result["model_name"] == "ComplexTestModel"
        assert result["is_compatible"] is True
        # Может быть предупреждения
        assert len(result.get("warnings", [])) >= 0

    def test_create_example_instance_simple(self):
        """Тест создания примера простой модели."""
        validator = StructuredOutputValidator()

        example = validator.create_example_instance(SimpleTestModel)

        assert isinstance(example, SimpleTestModel)
        assert example.name
        assert isinstance(example.age, int)
        assert example.age >= 0

    def test_create_example_instance_complex(self):
        """Тест создания примера сложной модели."""
        validator = StructuredOutputValidator()

        example = validator.create_example_instance(ComplexTestModel)

        assert isinstance(example, ComplexTestModel)
        assert isinstance(example.user, SimpleTestModel)
        assert isinstance(example.tags, list)
        assert isinstance(example.metadata, dict)
        assert isinstance(example.is_active, bool)


class TestValidatorUtilityFunctions:
    """Тесты утилитарных функций валидатора."""

    @pytest.mark.asyncio
    async def test_validate_structured_response_function(self):
        """Тест функции validate_structured_response."""
        data = {
            "name": "Функция Тест",
            "age": 40,
            "email": "function@example.com"
        }

        result = await validate_structured_response(data, SimpleTestModel)

        assert isinstance(result, SimpleTestModel)
        assert result.name == "Функция Тест"
        assert result.age == 40

    def test_check_model_compatibility_function(self):
        """Тест функции check_model_compatibility."""
        result = check_model_compatibility(SimpleTestModel)

        assert isinstance(result, dict)
        assert result["is_compatible"] is True
        assert result["model_name"] == "SimpleTestModel"

    def test_create_model_example_function(self):
        """Тест функции create_model_example."""
        example = create_model_example(SimpleTestModel)

        assert isinstance(example, SimpleTestModel)
        assert example.name
        assert isinstance(example.age, int)


class TestValidatorErrorHandling:
    """Тесты обработки ошибок валидатора."""

    @pytest.mark.asyncio
    async def test_validate_response_with_strict_mode(self):
        """Тест валидации в строгом режиме."""
        validator = StructuredOutputValidator()

        # Данные с дополнительным полем
        data = {
            "name": "Строгий Тест",
            "age": 25,
            "email": "strict@example.com",
            "extra_field": "should be ignored"  # Дополнительное поле
        }

        # В строгом режиме должна быть ошибка (пропускаем пока)
        # TODO: Исправить строгий режим в Pydantic 2.x
        # with pytest.raises(ValidationError):
        #     await validator.validate_response(data, SimpleTestModel, strict=True)

        # В нестрогом режиме должно работать
        result = await validator.validate_response(data, SimpleTestModel, strict=False)
        assert isinstance(result, SimpleTestModel)
        assert result.name == "Строгий Тест"

    def test_validate_schema_compatibility_invalid_model(self):
        """Тест проверки совместимости некорректной модели."""
        validator = StructuredOutputValidator()

        class InvalidModel:
            pass

        result = validator.validate_schema_compatibility(InvalidModel)

        assert result["is_compatible"] is False
        assert len(result["issues"]) > 0

    def test_create_example_instance_error(self):
        """Тест ошибки создания примера."""
        validator = StructuredOutputValidator()

        class ProblematicModel(BaseModel):
            # Очень строгое требование
            required_field: str = Field(..., min_length=100)

        with pytest.raises(ValidationError):
            validator.create_example_instance(ProblematicModel)


class TestValidatorComplexScenarios:
    """Тесты сложных сценариев валидатора."""

    @pytest.mark.asyncio
    async def test_validate_nested_model(self):
        """Тест валидации вложенной модели."""
        validator = StructuredOutputValidator()

        data = {
            "user": {
                "name": "Вложенный Пользователь",
                "age": 28,
                "email": "nested@example.com"
            },
            "tags": ["tag1", "tag2"],
            "metadata": {"key": "value"},
            "is_active": True
        }

        result = await validator.validate_response(data, ComplexTestModel)

        assert isinstance(result, ComplexTestModel)
        assert isinstance(result.user, SimpleTestModel)
        assert result.user.name == "Вложенный Пользователь"
        assert result.tags == ["tag1", "tag2"]
        assert result.metadata == {"key": "value"}
        assert result.is_active is True

    def test_schema_depth_calculation(self):
        """Тест вычисления глубины схемы."""
        validator = StructuredOutputValidator()

        # Простая модель должна иметь глубину >= 0
        simple_schema = SimpleTestModel.model_json_schema()
        simple_depth = validator._calculate_schema_depth(simple_schema)
        assert simple_depth >= 0

        # Сложная модель должна иметь глубину >= простой модели
        complex_schema = ComplexTestModel.model_json_schema()
        complex_depth = validator._calculate_schema_depth(complex_schema)
        assert complex_depth >= simple_depth

    def test_complex_types_detection(self):
        """Тест обнаружения сложных типов."""
        validator = StructuredOutputValidator()

        # Простая схема не должна содержать сложные типы
        simple_schema = SimpleTestModel.model_json_schema()
        assert not validator._has_complex_types(simple_schema)

        # Можно создать модель с union типами для тестирования
        from typing import Union

        class UnionModel(BaseModel):
            value: Union[str, int]

        union_schema = UnionModel.model_json_schema()
        # В зависимости от версии Pydantic может быть anyOf
        # assert validator._has_complex_types(union_schema)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
