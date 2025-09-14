"""
Unit тесты для FunctionRegistry в Kraken LLM фреймворке.
"""

import pytest
import json
from unittest.mock import Mock

from kraken_llm.tools.functions import (
    FunctionRegistry,
    FunctionSchema,
    FunctionCall,
    FunctionResult,
    register_function
)


class TestFunctionRegistry:
    """Тесты для класса FunctionRegistry"""
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        self.registry = FunctionRegistry()
    
    def test_init(self):
        """Тест инициализации реестра"""
        assert len(self.registry) == 0
        assert self.registry.list_functions() == []
    
    def test_register_function_basic(self):
        """Тест базовой регистрации функции"""
        def test_func(x: int, y: str = "default") -> str:
            return f"{x}: {y}"
        
        self.registry.register_function(
            name="test_func",
            func=test_func,
            description="Тестовая функция"
        )
        
        assert len(self.registry) == 1
        assert "test_func" in self.registry
        assert "test_func" in self.registry.list_functions()
        
        schema = self.registry.get_function_schema("test_func")
        assert schema is not None
        assert schema.name == "test_func"
        assert schema.description == "Тестовая функция"
    
    def test_register_function_with_custom_parameters(self):
        """Тест регистрации функции с кастомными параметрами"""
        def test_func(data):
            return data
        
        custom_params = {
            "type": "object",
            "properties": {
                "data": {"type": "string", "description": "Входные данные"}
            },
            "required": ["data"]
        }
        
        self.registry.register_function(
            name="custom_func",
            func=test_func,
            description="Функция с кастомными параметрами",
            parameters=custom_params
        )
        
        schema = self.registry.get_function_schema("custom_func")
        assert schema.parameters == custom_params
    
    def test_auto_generate_parameters_schema(self):
        """Тест автоматической генерации схемы параметров"""
        def typed_func(count: int, name: str, active: bool = True) -> dict:
            return {"count": count, "name": name, "active": active}
        
        self.registry.register_function(
            name="typed_func",
            func=typed_func,
            description="Функция с типизированными параметрами"
        )
        
        schema = self.registry.get_function_schema("typed_func")
        params = schema.parameters
        
        assert params["type"] == "object"
        assert "count" in params["properties"]
        assert "name" in params["properties"]
        assert "active" in params["properties"]
        
        assert params["properties"]["count"]["type"] == "integer"
        assert params["properties"]["name"]["type"] == "string"
        assert params["properties"]["active"]["type"] == "boolean"
        
        # count и name обязательные, active имеет значение по умолчанию
        assert "count" in params["required"]
        assert "name" in params["required"]
        assert "active" not in params["required"]
    
    def test_get_all_schemas(self):
        """Тест получения всех схем функций"""
        def func1():
            return "result1"
        
        def func2(x: int):
            return x * 2
        
        self.registry.register_function("func1", func1, "Первая функция")
        self.registry.register_function("func2", func2, "Вторая функция")
        
        schemas = self.registry.get_all_schemas()
        assert len(schemas) == 2
        
        names = [schema["name"] for schema in schemas]
        assert "func1" in names
        assert "func2" in names
    
    def test_execute_function_success(self):
        """Тест успешного выполнения функции"""
        def add_numbers(a: int, b: int) -> int:
            return a + b
        
        self.registry.register_function("add", add_numbers, "Сложение чисел")
        
        # Тест с dict аргументами
        result = self.registry.execute_function("add", {"a": 5, "b": 3})
        assert result.success is True
        assert result.result == 8
        assert result.name == "add"
        assert result.error is None
        
        # Тест с JSON строкой
        result = self.registry.execute_function("add", '{"a": 10, "b": 20}')
        assert result.success is True
        assert result.result == 30
    
    def test_execute_function_not_found(self):
        """Тест выполнения несуществующей функции"""
        result = self.registry.execute_function("nonexistent", {})
        assert result.success is False
        assert result.result is None
        assert "не зарегистрирована" in result.error
    
    def test_execute_function_invalid_json(self):
        """Тест выполнения функции с некорректным JSON"""
        def test_func():
            return "test"
        
        self.registry.register_function("test", test_func, "Тест")
        
        result = self.registry.execute_function("test", "invalid json")
        assert result.success is False
        assert "Ошибка парсинга JSON" in result.error
    
    def test_execute_function_runtime_error(self):
        """Тест выполнения функции с ошибкой времени выполнения"""
        def error_func():
            raise ValueError("Тестовая ошибка")
        
        self.registry.register_function("error", error_func, "Функция с ошибкой")
        
        result = self.registry.execute_function("error", {})
        assert result.success is False
        assert "Тестовая ошибка" in result.error
    
    def test_unregister_function(self):
        """Тест удаления функции из реестра"""
        def test_func():
            return "test"
        
        self.registry.register_function("test", test_func, "Тест")
        assert "test" in self.registry
        
        success = self.registry.unregister_function("test")
        assert success is True
        assert "test" not in self.registry
        
        # Попытка удалить несуществующую функцию
        success = self.registry.unregister_function("nonexistent")
        assert success is False
    
    def test_clear_registry(self):
        """Тест очистки реестра"""
        def func1():
            return 1
        
        def func2():
            return 2
        
        self.registry.register_function("func1", func1, "Функция 1")
        self.registry.register_function("func2", func2, "Функция 2")
        
        assert len(self.registry) == 2
        
        self.registry.clear()
        assert len(self.registry) == 0
        assert self.registry.list_functions() == []
    
    def test_register_function_overwrite(self):
        """Тест перезаписи существующей функции"""
        def func1():
            return "original"
        
        def func2():
            return "updated"
        
        self.registry.register_function("test", func1, "Оригинальная функция")
        result1 = self.registry.execute_function("test", {})
        assert result1.result == "original"
        
        # Перезаписываем функцию
        self.registry.register_function("test", func2, "Обновленная функция")
        result2 = self.registry.execute_function("test", {})
        assert result2.result == "updated"
        
        # Проверяем, что в реестре все еще одна функция
        assert len(self.registry) == 1


class TestFunctionDecorator:
    """Тесты для декоратора register_function"""
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        # Очищаем глобальный реестр
        from kraken_llm.tools.functions import default_function_registry
        default_function_registry.clear()
    
    def test_register_function_decorator_basic(self):
        """Тест базового использования декоратора"""
        @register_function(description="Получает приветствие")
        def get_greeting(name: str) -> str:
            return f"Привет, {name}!"
        
        from kraken_llm.tools.functions import default_function_registry
        
        assert "get_greeting" in default_function_registry
        result = default_function_registry.execute_function("get_greeting", {"name": "Мир"})
        assert result.success is True
        assert result.result == "Привет, Мир!"
    
    def test_register_function_decorator_custom_name(self):
        """Тест декоратора с кастомным именем"""
        @register_function(name="custom_name", description="Функция с кастомным именем")
        def original_name():
            return "test"
        
        from kraken_llm.tools.functions import default_function_registry
        
        assert "custom_name" in default_function_registry
        assert "original_name" not in default_function_registry
    
    def test_register_function_decorator_with_docstring(self):
        """Тест декоратора с использованием docstring как описания"""
        @register_function()
        def documented_function():
            """Эта функция имеет docstring"""
            return "documented"
        
        from kraken_llm.tools.functions import default_function_registry
        
        schema = default_function_registry.get_function_schema("documented_function")
        assert "docstring" in schema.description


class TestFunctionModels:
    """Тесты для Pydantic моделей функций"""
    
    def test_function_schema_model(self):
        """Тест модели FunctionSchema"""
        schema = FunctionSchema(
            name="test_func",
            description="Тестовая функция",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"]
            }
        )
        
        assert schema.name == "test_func"
        assert schema.description == "Тестовая функция"
        assert schema.parameters["type"] == "object"
    
    def test_function_call_model(self):
        """Тест модели FunctionCall"""
        # Тест с dict аргументами
        call1 = FunctionCall(name="test", arguments={"x": 1, "y": 2})
        assert call1.name == "test"
        assert call1.arguments == {"x": 1, "y": 2}
        
        # Тест с JSON строкой
        call2 = FunctionCall(name="test", arguments='{"x": 1, "y": 2}')
        assert call2.name == "test"
        assert call2.arguments == '{"x": 1, "y": 2}'
    
    def test_function_result_model(self):
        """Тест модели FunctionResult"""
        # Успешный результат
        result1 = FunctionResult(
            name="test",
            result="success",
            success=True
        )
        assert result1.success is True
        assert result1.error is None
        
        # Результат с ошибкой
        result2 = FunctionResult(
            name="test",
            result=None,
            success=False,
            error="Тестовая ошибка"
        )
        assert result2.success is False
        assert result2.error == "Тестовая ошибка"