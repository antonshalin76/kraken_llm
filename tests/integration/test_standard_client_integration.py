"""
Integration tests for StandardLLMClient.

These tests verify the StandardLLMClient works correctly with real API calls,
including function calling, tool calling, and error handling.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from kraken_llm.client.standard import StandardLLMClient
from kraken_llm.config.settings import LLMConfig
from kraken_llm.exceptions.api import APIError
from kraken_llm.exceptions.validation import ValidationError


class TestStandardLLMClientIntegration:
    """Integration tests for StandardLLMClient with real API interactions."""

    @pytest.fixture
    def config(self):
        """Test configuration."""
        return LLMConfig(
            endpoint="http://localhost:8080",
            api_key="test_key",
            model="test_model",
            temperature=0.7,
            max_tokens=100,
        )

    @pytest.fixture
    def client(self, config):
        """StandardLLMClient instance for testing."""
        return StandardLLMClient(config)

    @pytest.fixture
    def sample_messages(self):
        """Sample messages for testing."""
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]

    @pytest.mark.asyncio
    async def test_basic_chat_completion(self, client, sample_messages):
        """Test basic chat completion functionality."""
        # Mock the AsyncOpenAI client
        mock_response = MagicMock()
        mock_response.id = "test_response_id"
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello! I'm doing well, thank you for asking."
        mock_response.choices[0].message.function_call = None
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"

        client.openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response)
        # Mock server support check to avoid real network calls
        client._server_supports_non_streaming = AsyncMock(return_value=True)

        # Execute chat completion
        result = await client.chat_completion(
            messages=sample_messages,
            temperature=0.8,
            max_tokens=150
        )

        # Verify result
        assert result == "Hello! I'm doing well, thank you for asking."

        # Verify API call parameters
        call_args = client.openai_client.chat.completions.create.call_args
        assert call_args[1]["messages"] == sample_messages
        assert call_args[1]["temperature"] == 0.8
        assert call_args[1]["max_tokens"] == 150
        assert call_args[1]["model"] == "test_model"
        assert call_args[1]["stream"] is False

    @pytest.mark.asyncio
    async def test_function_calling(self, client, sample_messages):
        """Test function calling functionality."""
        # Register a test function
        def test_function(name: str, age: int) -> str:
            return f"Hello {name}, you are {age} years old!"

        client.register_function(
            name="greet_person",
            function=test_function,
            description="Greet a person with their name and age"
        )

        # Mock the AsyncOpenAI response with function call
        mock_response = MagicMock()
        mock_response.id = "test_response_id"
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = None
        mock_response.choices[0].message.function_call = MagicMock()
        mock_response.choices[0].message.function_call.name = "greet_person"
        mock_response.choices[0].message.function_call.arguments = '{"name": "Alice", "age": 30}'
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "function_call"

        client.openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response)
        # Mock server support check to avoid real network calls
        client._server_supports_non_streaming = AsyncMock(return_value=True)

        # Execute chat completion with functions
        functions = [{
            "name": "greet_person",
            "description": "Greet a person with their name and age",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name", "age"]
            }
        }]

        result = await client.chat_completion(
            messages=sample_messages,
            functions=functions,
            function_call="auto"
        )

        # Verify function execution result
        assert result == "Hello Alice, you are 30 years old!"

        # Verify API call included functions
        call_args = client.openai_client.chat.completions.create.call_args
        assert call_args[1]["functions"] == functions
        assert call_args[1]["function_call"] == "auto"

    @pytest.mark.asyncio
    async def test_async_function_calling(self, client, sample_messages):
        """Test async function calling functionality."""
        # Register an async test function
        async def async_test_function(query: str) -> str:
            await asyncio.sleep(0.01)  # Simulate async work
            return f"Processed query: {query}"

        client.register_function(
            name="process_query",
            function=async_test_function,
            description="Process a query asynchronously"
        )

        # Mock the AsyncOpenAI response with function call
        mock_response = MagicMock()
        mock_response.id = "test_response_id"
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = None
        mock_response.choices[0].message.function_call = MagicMock()
        mock_response.choices[0].message.function_call.name = "process_query"
        mock_response.choices[0].message.function_call.arguments = '{"query": "test query"}'
        mock_response.choices[0].message.tool_calls = None

        client.openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response)
        # Mock server support check to avoid real network calls
        client._server_supports_non_streaming = AsyncMock(return_value=True)

        # Execute chat completion
        result = await client.chat_completion(
            messages=sample_messages,
            functions=[{
                "name": "process_query",
                "description": "Process a query asynchronously",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }
            }]
        )

        # Verify async function execution result
        assert result == "Processed query: test query"

    @pytest.mark.asyncio
    async def test_tool_calling(self, client, sample_messages):
        """Test tool calling functionality."""
        # Register a test tool
        def calculator_tool(operation: str, a: float, b: float) -> float:
            if operation == "add":
                return a + b
            elif operation == "multiply":
                return a * b
            else:
                raise ValueError(f"Unknown operation: {operation}")

        client.register_tool(
            name="calculator",
            tool=calculator_tool,
            description="Perform basic math operations"
        )

        # Mock the AsyncOpenAI response with tool calls
        mock_response = MagicMock()
        mock_response.id = "test_response_id"
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = None
        mock_response.choices[0].message.function_call = None
        mock_response.choices[0].message.tool_calls = [MagicMock()]

        tool_call = mock_response.choices[0].message.tool_calls[0]
        tool_call.id = "call_123"
        tool_call.type = "function"
        tool_call.function = MagicMock()
        tool_call.function.name = "calculator"
        tool_call.function.arguments = '{"operation": "add", "a": 5, "b": 3}'

        client.openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response)
        # Mock server support check to avoid real network calls
        client._server_supports_non_streaming = AsyncMock(return_value=True)

        # Execute chat completion with tools
        tools = [{
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Perform basic math operations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string", "enum": ["add", "multiply"]},
                        "a": {"type": "number"},
                        "b": {"type": "number"}
                    },
                    "required": ["operation", "a", "b"]
                }
            }
        }]

        result = await client.chat_completion(
            messages=sample_messages,
            tools=tools,
            tool_choice="auto"
        )

        # Verify tool execution result
        assert "Tool calculator: 8" in result

        # Verify API call included tools
        call_args = client.openai_client.chat.completions.create.call_args
        assert call_args[1]["tools"] == tools
        assert call_args[1]["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self, client, sample_messages):
        """Test multiple parallel tool calls."""
        # Register multiple tools
        def add_tool(a: float, b: float) -> float:
            return a + b

        def multiply_tool(a: float, b: float) -> float:
            return a * b

        client.register_tool("add", add_tool)
        client.register_tool("multiply", multiply_tool)

        # Mock response with multiple tool calls
        mock_response = MagicMock()
        mock_response.id = "test_response_id"
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = None
        mock_response.choices[0].message.function_call = None
        mock_response.choices[0].message.tool_calls = [
            MagicMock(), MagicMock()]

        # First tool call
        tool_call_1 = mock_response.choices[0].message.tool_calls[0]
        tool_call_1.id = "call_1"
        tool_call_1.type = "function"
        tool_call_1.function = MagicMock()
        tool_call_1.function.name = "add"
        tool_call_1.function.arguments = '{"a": 5, "b": 3}'

        # Second tool call
        tool_call_2 = mock_response.choices[0].message.tool_calls[1]
        tool_call_2.id = "call_2"
        tool_call_2.type = "function"
        tool_call_2.function = MagicMock()
        tool_call_2.function.name = "multiply"
        tool_call_2.function.arguments = '{"a": 4, "b": 6}'

        client.openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response)
        # Mock server support check to avoid real network calls
        client._server_supports_non_streaming = AsyncMock(return_value=True)

        # Execute chat completion
        result = await client.chat_completion(
            messages=sample_messages,
            tools=[
                {"type": "function", "function": {"name": "add"}},
                {"type": "function", "function": {"name": "multiply"}}
            ]
        )

        # Verify both tools were executed
        assert "Tool add: 8" in result
        assert "Tool multiply: 24" in result

    @pytest.mark.asyncio
    async def test_validation_errors(self, client):
        """Test parameter validation."""
        # Test empty messages
        with pytest.raises(ValidationError, match="Список сообщений не может быть пустым"):
            await client.chat_completion(messages=[])

        # Test streaming not supported
        with pytest.raises(ValidationError, match="не поддерживает streaming"):
            await client.chat_completion(
                messages=[{"role": "user", "content": "test"}],
                stream=True
            )

        # Test function_call without functions
        with pytest.raises(ValidationError, match="function_call указан, но functions не предоставлены"):
            await client.chat_completion(
                messages=[{"role": "user", "content": "test"}],
                function_call="auto"
            )

        # Test tool_choice without tools
        with pytest.raises(ValidationError, match="tool_choice указан, но tools не предоставлены"):
            await client.chat_completion(
                messages=[{"role": "user", "content": "test"}],
                tool_choice="auto"
            )

        # Test functions and tools together
        with pytest.raises(ValidationError, match="Нельзя использовать functions и tools одновременно"):
            await client.chat_completion(
                messages=[{"role": "user", "content": "test"}],
                functions=[{"name": "test"}],
                tools=[{"type": "function", "function": {"name": "test"}}]
            )

        # Test invalid message structure
        with pytest.raises(ValidationError, match="должно содержать 'role' и 'content'"):
            await client.chat_completion(
                messages=[{"role": "user"}]  # Missing content
            )

        # Test invalid role
        with pytest.raises(ValidationError, match="Некорректная роль"):
            await client.chat_completion(
                messages=[{"role": "invalid", "content": "test"}]
            )

    @pytest.mark.asyncio
    async def test_function_execution_errors(self, client, sample_messages):
        """Test error handling in function execution."""
        # Register a function that raises an error
        def error_function():
            raise ValueError("Test error")

        client.register_function("error_func", error_function)

        # Mock response with function call
        mock_response = MagicMock()
        mock_response.id = "test_response_id"
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = None
        mock_response.choices[0].message.function_call = MagicMock()
        mock_response.choices[0].message.function_call.name = "error_func"
        mock_response.choices[0].message.function_call.arguments = "{}"
        mock_response.choices[0].message.tool_calls = None

        client.openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response)
        # Mock server support check to avoid real network calls
        client._server_supports_non_streaming = AsyncMock(return_value=True)

        # Execute and verify error handling
        result = await client.chat_completion(
            messages=sample_messages,
            functions=[{"name": "error_func"}]
        )

        assert "Ошибка выполнения функции error_func: Test error" in result

    @pytest.mark.asyncio
    async def test_invalid_function_arguments(self, client, sample_messages):
        """Test handling of invalid function arguments."""
        client.register_function("test_func", lambda x: x)

        # Mock response with invalid JSON arguments
        mock_response = MagicMock()
        mock_response.id = "test_response_id"
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = None
        mock_response.choices[0].message.function_call = MagicMock()
        mock_response.choices[0].message.function_call.name = "test_func"
        mock_response.choices[0].message.function_call.arguments = "invalid json"
        mock_response.choices[0].message.tool_calls = None

        client.openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response)
        # Mock server support check to avoid real network calls
        client._server_supports_non_streaming = AsyncMock(return_value=True)

        result = await client.chat_completion(
            messages=sample_messages,
            functions=[{"name": "test_func"}]
        )

        assert "некорректные аргументы функции" in result

    @pytest.mark.asyncio
    async def test_unregistered_function_call(self, client, sample_messages):
        """Test calling unregistered function."""
        # Mock response with unregistered function call
        mock_response = MagicMock()
        mock_response.id = "test_response_id"
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = None
        mock_response.choices[0].message.function_call = MagicMock()
        mock_response.choices[0].message.function_call.name = "unknown_func"
        mock_response.choices[0].message.function_call.arguments = "{}"
        mock_response.choices[0].message.tool_calls = None

        client.openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response)
        # Mock server support check to avoid real network calls
        client._server_supports_non_streaming = AsyncMock(return_value=True)

        result = await client.chat_completion(
            messages=sample_messages,
            functions=[{"name": "unknown_func"}]
        )

        assert "функция unknown_func не найдена" in result

    @pytest.mark.asyncio
    async def test_registry_methods(self, client):
        """Test function and tool registry methods."""
        # Test empty registries
        assert client.get_registered_functions() == []
        assert client.get_registered_tools() == []

        # Register functions and tools
        client.register_function("func1", lambda: None)
        client.register_function("func2", lambda: None)
        client.register_tool("tool1", lambda: None)
        client.register_tool("tool2", lambda: None)

        # Test registry contents
        assert set(client.get_registered_functions()) == {"func1", "func2"}
        assert set(client.get_registered_tools()) == {"tool1", "tool2"}

    @pytest.mark.asyncio
    async def test_unsupported_methods(self, client, sample_messages):
        """Test that unsupported methods raise NotImplementedError."""
        # Test streaming method
        with pytest.raises(NotImplementedError, match="не поддерживает streaming"):
            async for _ in client.chat_completion_stream(messages=sample_messages):
                pass

        # Test structured output method
        from pydantic import BaseModel

        class TestModel(BaseModel):
            text: str

        with pytest.raises(NotImplementedError, match="не поддерживает structured output"):
            await client.chat_completion_structured(
                messages=sample_messages,
                response_model=TestModel
            )

    @pytest.mark.asyncio
    async def test_client_cleanup(self, client):
        """Test client cleanup and context manager."""
        # Test manual cleanup
        await client.close()

        # Test context manager
        config = LLMConfig(api_key="test_key")
        async with StandardLLMClient(config) as client:
            assert client is not None
        # Client should be closed automatically

    @pytest.mark.asyncio
    async def test_empty_response_handling(self, client, sample_messages):
        """Test handling of empty responses."""
        # Mock empty response
        mock_response = MagicMock()
        mock_response.id = "test_response_id"
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = None
        mock_response.choices[0].message.function_call = None
        mock_response.choices[0].message.tool_calls = None

        client.openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response)
        # Mock server support check to avoid real network calls
        client._server_supports_non_streaming = AsyncMock(return_value=True)

        result = await client.chat_completion(messages=sample_messages)

        assert result == ""

    @pytest.mark.asyncio
    async def test_no_choices_response(self, client, sample_messages):
        """Test handling of response with no choices."""
        # Mock response with no choices
        mock_response = MagicMock()
        mock_response.id = "test_response_id"
        mock_response.choices = []

        client.openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response)
        # Mock server support check to avoid real network calls
        client._server_supports_non_streaming = AsyncMock(return_value=True)

        with pytest.raises(APIError, match="Ответ API не содержит choices"):
            await client.chat_completion(messages=sample_messages)
