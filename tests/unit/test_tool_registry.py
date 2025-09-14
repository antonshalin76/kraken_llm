"""
Unit тесты для ToolRegistry в Kraken LLM фреймворке.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock

from kraken_llm.tools.tools import (
    ToolRegistry,
    ToolSchema,
    ToolFunction,
    ToolCall,
    ToolResult,
    register_tool
)


class TestToolRegistry:
    """Тесты для класса ToolRegistry"""
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        self.registry = ToolRegistry()
    
    def test_init(self):
        """Тест инициализации реестра"""
        assert len(self.registry) == 0
        assert self.registry.list_tools() == []
    
    def test_register_tool_basic(self):
        """Тест базовой регистрации инструмента"""
        def test_tool(x: int, y: str = "default") -> str:
            return f"{x}: {y}"
        
        self.registry.register_tool(
            name="test_tool",
            tool=test_tool,
            description="Тестовый инструмент"
        )
        
        assert len(self.registry) == 1
        assert "test_tool" in self.registry
        assert "test_tool" in self.registry.list_tools()
        
        schema = self.registry.get_tool_schema("test_tool")
        assert schema is not None
        assert schema.type == "function"
        assert schema.function.name == "test_tool"
        assert schema.function.description == "Тестовый инструмент"
    
    def test_register_async_tool(self):
        """Тест регистрации асинхронного инструмента"""
        async def async_tool(data: str) -> str:
            await asyncio.sleep(0.01)  # Имитация асинхронной работы
            return f"processed: {data}"
        
        self.registry.register_tool(
            name="async_tool",
            tool=async_tool,
            description="Асинхронный инструмент"
        )
        
        assert "async_tool" in self.registry
        schema = self.registry.get_tool_schema("async_tool")
        assert schema.function.name == "async_tool"
    
    def test_register_tool_with_custom_parameters(self):
        """Тест регистрации инструмента с кастомными параметрами"""
        def test_tool(data):
            return data
        
        custom_params = {
            "type": "object",
            "properties": {
                "data": {"type": "string", "description": "Входные данные"}
            },
            "required": ["data"]
        }
        
        self.registry.register_tool(
            name="custom_tool",
            tool=test_tool,
            description="Инструмент с кастомными параметрами",
            parameters=custom_params
        )
        
        schema = self.registry.get_tool_schema("custom_tool")
        assert schema.function.parameters == custom_params
    
    def test_auto_generate_parameters_schema(self):
        """Тест автоматической генерации схемы параметров"""
        def typed_tool(count: int, name: str, active: bool = True) -> dict:
            return {"count": count, "name": name, "active": active}
        
        self.registry.register_tool(
            name="typed_tool",
            tool=typed_tool,
            description="Инструмент с типизированными параметрами"
        )
        
        schema = self.registry.get_tool_schema("typed_tool")
        params = schema.function.parameters
        
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
        """Тест получения всех схем инструментов"""
        def tool1():
            return "result1"
        
        async def tool2(x: int):
            return x * 2
        
        self.registry.register_tool("tool1", tool1, "Первый инструмент")
        self.registry.register_tool("tool2", tool2, "Второй инструмент")
        
        schemas = self.registry.get_all_schemas()
        assert len(schemas) == 2
        
        names = [schema["function"]["name"] for schema in schemas]
        assert "tool1" in names
        assert "tool2" in names
        
        # Проверяем формат OpenAI
        for schema in schemas:
            assert schema["type"] == "function"
            assert "function" in schema
            assert "name" in schema["function"]
            assert "description" in schema["function"]
            assert "parameters" in schema["function"]
    
    @pytest.mark.asyncio
    async def test_execute_tool_success(self):
        """Тест успешного выполнения инструмента"""
        def add_numbers(a: int, b: int) -> int:
            return a + b
        
        self.registry.register_tool("add", add_numbers, "Сложение чисел")
        
        # Тест с dict аргументами
        result = await self.registry.execute_tool("add", {"a": 5, "b": 3}, "call_1")
        assert result.success is True
        assert result.result == 8
        assert result.name == "add"
        assert result.tool_call_id == "call_1"
        assert result.error is None
        
        # Тест с JSON строкой
        result = await self.registry.execute_tool("add", '{"a": 10, "b": 20}', "call_2")
        assert result.success is True
        assert result.result == 30
        assert result.tool_call_id == "call_2"
    
    @pytest.mark.asyncio
    async def test_execute_async_tool_success(self):
        """Тест успешного выполнения асинхронного инструмента"""
        async def async_multiply(x: int, y: int) -> int:
            await asyncio.sleep(0.01)
            return x * y
        
        self.registry.register_tool("multiply", async_multiply, "Умножение чисел")
        
        result = await self.registry.execute_tool("multiply", {"x": 4, "y": 5}, "async_call")
        assert result.success is True
        assert result.result == 20
        assert result.tool_call_id == "async_call"
    
    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self):
        """Тест выполнения несуществующего инструмента"""
        result = await self.registry.execute_tool("nonexistent", {}, "call_1")
        assert result.success is False
        assert result.result is None
        assert "не зарегистрирован" in result.error
        assert result.tool_call_id == "call_1"
    
    @pytest.mark.asyncio
    async def test_execute_tool_invalid_json(self):
        """Тест выполнения инструмента с некорректным JSON"""
        def test_tool():
            return "test"
        
        self.registry.register_tool("test", test_tool, "Тест")
        
        result = await self.registry.execute_tool("test", "invalid json", "call_1")
        assert result.success is False
        assert "Ошибка парсинга JSON" in result.error
    
    @pytest.mark.asyncio
    async def test_execute_tool_runtime_error(self):
        """Тест выполнения инструмента с ошибкой времени выполнения"""
        def error_tool():
            raise ValueError("Тестовая ошибка")
        
        self.registry.register_tool("error", error_tool, "Инструмент с ошибкой")
        
        result = await self.registry.execute_tool("error", {}, "call_1")
        assert result.success is False
        assert "Тестовая ошибка" in result.error
    
    @pytest.mark.asyncio
    async def test_execute_parallel_tools(self):
        """Тест параллельного выполнения инструментов"""
        def fast_tool(value: int) -> int:
            return value * 2
        
        async def slow_tool(value: int) -> int:
            await asyncio.sleep(0.01)
            return value + 10
        
        self.registry.register_tool("fast", fast_tool, "Быстрый инструмент")
        self.registry.register_tool("slow", slow_tool, "Медленный инструмент")
        
        tool_calls = [
            ToolCall(
                id="call_1",
                type="function",
                function={"name": "fast", "arguments": {"value": 5}}
            ),
            ToolCall(
                id="call_2", 
                type="function",
                function={"name": "slow", "arguments": {"value": 3}}
            )
        ]
        
        results = await self.registry.execute_parallel_tools(tool_calls)
        
        assert len(results) == 2
        assert all(result.success for result in results)
        
        # Результаты могут прийти в любом порядке из-за параллельности
        results_by_id = {result.tool_call_id: result for result in results}
        
        assert results_by_id["call_1"].result == 10  # 5 * 2
        assert results_by_id["call_2"].result == 13  # 3 + 10
    
    @pytest.mark.asyncio
    async def test_execute_parallel_tools_with_errors(self):
        """Тест параллельного выполнения с ошибками"""
        def success_tool() -> str:
            return "success"
        
        def error_tool():
            raise RuntimeError("Ошибка инструмента")
        
        self.registry.register_tool("success", success_tool, "Успешный инструмент")
        self.registry.register_tool("error", error_tool, "Инструмент с ошибкой")
        
        tool_calls = [
            ToolCall(
                id="call_1",
                type="function", 
                function={"name": "success", "arguments": {}}
            ),
            ToolCall(
                id="call_2",
                type="function",
                function={"name": "error", "arguments": {}}
            )
        ]
        
        results = await self.registry.execute_parallel_tools(tool_calls)
        
        assert len(results) == 2
        
        results_by_id = {result.tool_call_id: result for result in results}
        
        assert results_by_id["call_1"].success is True
        assert results_by_id["call_1"].result == "success"
        
        assert results_by_id["call_2"].success is False
        assert "Ошибка инструмента" in results_by_id["call_2"].error
    
    @pytest.mark.asyncio
    async def test_execute_parallel_tools_empty_list(self):
        """Тест параллельного выполнения с пустым списком"""
        results = await self.registry.execute_parallel_tools([])
        assert results == []
    
    def test_unregister_tool(self):
        """Тест удаления инструмента из реестра"""
        def test_tool():
            return "test"
        
        self.registry.register_tool("test", test_tool, "Тест")
        assert "test" in self.registry
        
        success = self.registry.unregister_tool("test")
        assert success is True
        assert "test" not in self.registry
        
        # Попытка удалить несуществующий инструмент
        success = self.registry.unregister_tool("nonexistent")
        assert success is False
    
    def test_clear_registry(self):
        """Тест очистки реестра"""
        def tool1():
            return 1
        
        def tool2():
            return 2
        
        self.registry.register_tool("tool1", tool1, "Инструмент 1")
        self.registry.register_tool("tool2", tool2, "Инструмент 2")
        
        assert len(self.registry) == 2
        
        self.registry.clear()
        assert len(self.registry) == 0
        assert self.registry.list_tools() == []


class TestToolDecorator:
    """Тесты для декоратора register_tool"""
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        # Очищаем глобальный реестр
        from kraken_llm.tools.tools import default_tool_registry
        default_tool_registry.clear()
    
    @pytest.mark.asyncio
    async def test_register_tool_decorator_basic(self):
        """Тест базового использования декоратора"""
        @register_tool(description="Получает приветствие")
        def get_greeting(name: str) -> str:
            return f"Привет, {name}!"
        
        from kraken_llm.tools.tools import default_tool_registry
        
        assert "get_greeting" in default_tool_registry
        result = await default_tool_registry.execute_tool("get_greeting", {"name": "Мир"}, "test_call")
        assert result.success is True
        assert result.result == "Привет, Мир!"
    
    def test_register_tool_decorator_custom_name(self):
        """Тест декоратора с кастомным именем"""
        @register_tool(name="custom_name", description="Инструмент с кастомным именем")
        def original_name():
            return "test"
        
        from kraken_llm.tools.tools import default_tool_registry
        
        assert "custom_name" in default_tool_registry
        assert "original_name" not in default_tool_registry
    
    def test_register_tool_decorator_async(self):
        """Тест декоратора с асинхронной функцией"""
        @register_tool(description="Асинхронный инструмент")
        async def async_tool(data: str) -> str:
            await asyncio.sleep(0.01)
            return f"processed: {data}"
        
        from kraken_llm.tools.tools import default_tool_registry
        
        assert "async_tool" in default_tool_registry


class TestToolModels:
    """Тесты для Pydantic моделей инструментов"""
    
    def test_tool_function_model(self):
        """Тест модели ToolFunction"""
        tool_func = ToolFunction(
            name="test_tool",
            description="Тестовый инструмент",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"]
            }
        )
        
        assert tool_func.name == "test_tool"
        assert tool_func.description == "Тестовый инструмент"
        assert tool_func.parameters["type"] == "object"
    
    def test_tool_schema_model(self):
        """Тест модели ToolSchema"""
        tool_func = ToolFunction(
            name="test",
            description="Тест",
            parameters={"type": "object", "properties": {}}
        )
        
        schema = ToolSchema(function=tool_func)
        assert schema.type == "function"
        assert schema.function == tool_func
    
    def test_tool_call_model(self):
        """Тест модели ToolCall"""
        call = ToolCall(
            id="call_123",
            type="function",
            function={"name": "test", "arguments": {"x": 1}}
        )
        
        assert call.id == "call_123"
        assert call.type == "function"
        assert call.function["name"] == "test"
        assert call.function["arguments"] == {"x": 1}
    
    def test_tool_result_model(self):
        """Тест модели ToolResult"""
        # Успешный результат
        result1 = ToolResult(
            tool_call_id="call_1",
            name="test",
            result="success",
            success=True
        )
        assert result1.success is True
        assert result1.error is None
        
        # Результат с ошибкой
        result2 = ToolResult(
            tool_call_id="call_2",
            name="test",
            result=None,
            success=False,
            error="Тестовая ошибка"
        )
        assert result2.success is False
        assert result2.error == "Тестовая ошибка"