"""
Unit тесты для FunctionToolExecutor в Kraken LLM фреймворке.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from kraken_llm.tools.executor import (
    FunctionToolExecutor,
    ExecutionContext,
    ExecutionResult,
    execute_function_calls,
    execute_tool_calls
)
from kraken_llm.tools.functions import FunctionRegistry
from kraken_llm.tools.tools import ToolRegistry


class TestFunctionToolExecutor:
    """Тесты для класса FunctionToolExecutor"""

    def setup_method(self):
        """Настройка перед каждым тестом"""
        self.function_registry = FunctionRegistry()
        self.tool_registry = ToolRegistry()
        self.executor = FunctionToolExecutor(
            function_registry=self.function_registry,
            tool_registry=self.tool_registry
        )

    def test_init_with_registries(self):
        """Тест инициализации с переданными реестрами"""
        # Проверяем, что executor использует переданные реестры
        # Регистрируем функцию в нашем реестре
        def test_func():
            return "test"

        self.function_registry.register_function("test", test_func, "Тест")

        # Проверяем, что executor видит эту функцию
        functions = self.executor.get_available_functions()
        assert len(functions) == 1
        assert functions[0]["name"] == "test"

    def test_init_with_default_registries(self):
        """Тест инициализации с реестрами по умолчанию"""
        executor = FunctionToolExecutor()

        from kraken_llm.tools.functions import default_function_registry
        from kraken_llm.tools.tools import default_tool_registry

        assert executor.function_registry is default_function_registry
        assert executor.tool_registry is default_tool_registry

    @pytest.mark.asyncio
    async def test_execute_function_calls_success(self):
        """Тест успешного выполнения function calls"""
        # Регистрируем тестовые функции
        def add(a: int, b: int) -> int:
            return a + b

        def multiply(x: int, y: int) -> int:
            return x * y

        self.function_registry.register_function("add", add, "Сложение")
        self.function_registry.register_function(
            "multiply", multiply, "Умножение")

        # Подготавливаем вызовы функций
        function_calls = [
            {"name": "add", "arguments": {"a": 5, "b": 3}},
            {"name": "multiply", "arguments": {"x": 4, "y": 2}}
        ]

        context = ExecutionContext(request_id="test_request")

        # Выполняем
        result = await self.executor.execute_function_calls(function_calls, context)

        assert result.success is True
        assert len(result.results) == 2
        assert result.context == context
        assert result.execution_time is not None

        # Проверяем результаты
        assert result.results[0].success is True
        assert result.results[0].result == 8  # 5 + 3
        assert result.results[0].name == "add"

        assert result.results[1].success is True
        assert result.results[1].result == 8  # 4 * 2
        assert result.results[1].name == "multiply"

    @pytest.mark.asyncio
    async def test_execute_function_calls_with_error(self):
        """Тест выполнения function calls с ошибкой"""
        def success_func() -> str:
            return "success"

        def error_func():
            raise ValueError("Тестовая ошибка")

        self.function_registry.register_function(
            "success", success_func, "Успешная функция")
        self.function_registry.register_function(
            "error", error_func, "Функция с ошибкой")

        function_calls = [
            {"name": "success", "arguments": {}},
            {"name": "error", "arguments": {}}
        ]

        result = await self.executor.execute_function_calls(function_calls)

        assert result.success is False  # Общий статус False из-за ошибки
        assert len(result.results) == 2

        assert result.results[0].success is True
        assert result.results[0].result == "success"

        assert result.results[1].success is False
        assert "Тестовая ошибка" in result.results[1].error

    @pytest.mark.asyncio
    async def test_execute_tool_calls_success(self):
        """Тест успешного выполнения tool calls"""
        # Регистрируем тестовые инструменты
        def subtract(a: int, b: int) -> int:
            return a - b

        async def async_divide(x: int, y: int) -> float:
            await asyncio.sleep(0.01)
            return x / y

        self.tool_registry.register_tool("subtract", subtract, "Вычитание")
        self.tool_registry.register_tool("divide", async_divide, "Деление")

        # Подготавливаем вызовы инструментов
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "subtract", "arguments": {"a": 10, "b": 3}}
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "divide", "arguments": {"x": 15, "y": 3}}
            }
        ]

        context = ExecutionContext(user_id="test_user")

        # Выполняем
        result = await self.executor.execute_tool_calls(tool_calls, context)

        assert result.success is True
        assert len(result.results) == 2
        assert result.context == context
        assert result.execution_time is not None

        # Результаты могут прийти в любом порядке из-за параллельности
        results_by_id = {r.tool_call_id: r for r in result.results}

        assert results_by_id["call_1"].success is True
        assert results_by_id["call_1"].result == 7  # 10 - 3
        assert results_by_id["call_1"].name == "subtract"

        assert results_by_id["call_2"].success is True
        assert results_by_id["call_2"].result == 5.0  # 15 / 3
        assert results_by_id["call_2"].name == "divide"

    @pytest.mark.asyncio
    async def test_execute_tool_calls_with_error(self):
        """Тест выполнения tool calls с ошибкой"""
        def success_tool() -> str:
            return "success"

        async def error_tool():
            raise RuntimeError("Асинхронная ошибка")

        self.tool_registry.register_tool(
            "success", success_tool, "Успешный инструмент")
        self.tool_registry.register_tool(
            "error", error_tool, "Инструмент с ошибкой")

        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "success", "arguments": {}}
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "error", "arguments": {}}
            }
        ]

        result = await self.executor.execute_tool_calls(tool_calls)

        assert result.success is False  # Общий статус False из-за ошибки
        assert len(result.results) == 2

        results_by_id = {r.tool_call_id: r for r in result.results}

        assert results_by_id["call_1"].success is True
        assert results_by_id["call_1"].result == "success"

        assert results_by_id["call_2"].success is False
        assert "Асинхронная ошибка" in results_by_id["call_2"].error

    @pytest.mark.asyncio
    async def test_execute_mixed_calls(self):
        """Тест выполнения смешанных function и tool calls"""
        # Регистрируем функцию и инструмент
        def func_add(a: int, b: int) -> int:
            return a + b

        async def tool_multiply(x: int, y: int) -> int:
            await asyncio.sleep(0.01)
            return x * y

        self.function_registry.register_function(
            "add", func_add, "Функция сложения")
        self.tool_registry.register_tool(
            "multiply", tool_multiply, "Инструмент умножения")

        function_calls = [{"name": "add", "arguments": {"a": 2, "b": 3}}]
        tool_calls = [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "multiply", "arguments": {"x": 4, "y": 5}}
        }]

        context = ExecutionContext(session_id="test_session")

        result = await self.executor.execute_mixed_calls(
            function_calls=function_calls,
            tool_calls=tool_calls,
            context=context
        )

        assert result.success is True
        assert len(result.results) == 2
        assert result.context == context

        # Проверяем, что есть результаты обоих типов
        function_results = [r for r in result.results if hasattr(
            r, 'name') and not hasattr(r, 'tool_call_id')]
        tool_results = [
            r for r in result.results if hasattr(r, 'tool_call_id')]

        assert len(function_results) == 1
        assert len(tool_results) == 1

        assert function_results[0].result == 5  # 2 + 3
        assert tool_results[0].result == 20  # 4 * 5

    @pytest.mark.asyncio
    async def test_execute_mixed_calls_empty(self):
        """Тест выполнения смешанных calls с пустыми списками"""
        result = await self.executor.execute_mixed_calls()

        assert result.success is True
        assert len(result.results) == 0
        assert result.execution_time is not None

    def test_get_available_functions(self):
        """Тест получения доступных функций"""
        def test_func():
            return "test"

        self.function_registry.register_function(
            "test", test_func, "Тестовая функция")

        functions = self.executor.get_available_functions()
        assert len(functions) == 1
        assert functions[0]["name"] == "test"
        assert functions[0]["description"] == "Тестовая функция"

    def test_get_available_tools(self):
        """Тест получения доступных инструментов"""
        def test_tool():
            return "test"

        self.tool_registry.register_tool(
            "test", test_tool, "Тестовый инструмент")

        tools = self.executor.get_available_tools()
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "test"
        assert tools[0]["function"]["description"] == "Тестовый инструмент"

    def test_get_all_capabilities(self):
        """Тест получения всех возможностей"""
        def test_func():
            return "func"

        def test_tool():
            return "tool"

        self.function_registry.register_function("func", test_func, "Функция")
        self.tool_registry.register_tool("tool", test_tool, "Инструмент")

        capabilities = self.executor.get_all_capabilities()

        assert "functions" in capabilities
        assert "tools" in capabilities
        assert "function_count" in capabilities
        assert "tool_count" in capabilities

        assert capabilities["function_count"] == 1
        assert capabilities["tool_count"] == 1
        assert len(capabilities["functions"]) == 1
        assert len(capabilities["tools"]) == 1

    @pytest.mark.asyncio
    async def test_validate_calls_valid(self):
        """Тест валидации корректных вызовов"""
        def test_func():
            return "test"

        def test_tool():
            return "test"

        self.function_registry.register_function("func", test_func, "Функция")
        self.tool_registry.register_tool("tool", test_tool, "Инструмент")

        function_calls = [{"name": "func", "arguments": {}}]
        tool_calls = [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "tool", "arguments": {}}
        }]

        validation = await self.executor.validate_calls(function_calls, tool_calls)

        assert validation["valid"] is True
        assert len(validation["errors"]) == 0

    @pytest.mark.asyncio
    async def test_validate_calls_invalid(self):
        """Тест валидации некорректных вызовов"""
        # Не регистрируем никаких функций/инструментов

        function_calls = [
            {"name": "nonexistent", "arguments": {}},
            {"arguments": {}}  # Отсутствует имя
        ]

        tool_calls = [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "nonexistent_tool", "arguments": "invalid json"}
        }]

        validation = await self.executor.validate_calls(function_calls, tool_calls)

        assert validation["valid"] is False
        assert len(validation["errors"]) > 0

        # Проверяем, что есть ошибки для всех проблем
        error_messages = " ".join(validation["errors"])
        assert "не зарегистрирована" in error_messages or "не зарегистрирован" in error_messages
        assert "отсутствует имя" in error_messages


class TestExecutionContext:
    """Тесты для модели ExecutionContext"""

    def test_execution_context_basic(self):
        """Тест базового создания контекста"""
        context = ExecutionContext()
        assert context.request_id is None
        assert context.user_id is None
        assert context.session_id is None
        assert context.metadata == {}

    def test_execution_context_with_data(self):
        """Тест создания контекста с данными"""
        context = ExecutionContext(
            request_id="req_123",
            user_id="user_456",
            session_id="session_789",
            metadata={"key": "value"}
        )

        assert context.request_id == "req_123"
        assert context.user_id == "user_456"
        assert context.session_id == "session_789"
        assert context.metadata == {"key": "value"}


class TestExecutionResult:
    """Тесты для модели ExecutionResult"""

    def test_execution_result_basic(self):
        """Тест базового создания результата"""
        result = ExecutionResult(success=True, results=[])
        assert result.success is True
        assert result.results == []
        assert result.execution_time is None
        assert result.context is None


class TestGlobalFunctions:
    """Тесты для глобальных функций"""

    def setup_method(self):
        """Настройка перед каждым тестом"""
        # Очищаем глобальные реестры
        from kraken_llm.tools.functions import default_function_registry
        from kraken_llm.tools.tools import default_tool_registry

        default_function_registry.clear()
        default_tool_registry.clear()

    @pytest.mark.asyncio
    async def test_execute_function_calls_global(self):
        """Тест глобальной функции execute_function_calls"""
        from kraken_llm.tools.functions import default_function_registry

        def test_func() -> str:
            return "global_test"

        default_function_registry.register_function(
            "test", test_func, "Глобальная функция")

        function_calls = [{"name": "test", "arguments": {}}]
        result = await execute_function_calls(function_calls)

        assert result.success is True
        assert len(result.results) == 1
        assert result.results[0].result == "global_test"

    @pytest.mark.asyncio
    async def test_execute_tool_calls_global(self):
        """Тест глобальной функции execute_tool_calls"""
        from kraken_llm.tools.tools import default_tool_registry

        async def test_tool() -> str:
            return "global_tool"

        default_tool_registry.register_tool(
            "test", test_tool, "Глобальный инструмент")

        tool_calls = [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "test", "arguments": {}}
        }]

        result = await execute_tool_calls(tool_calls)

        assert result.success is True
        assert len(result.results) == 1
        assert result.results[0].result == "global_tool"
