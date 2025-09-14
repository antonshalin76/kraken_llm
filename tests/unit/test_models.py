"""
Unit тесты для моделей данных Kraken LLM фреймворка.

Тестирует валидацию, сериализацию и десериализацию всех Pydantic моделей.
"""

import pytest
from datetime import datetime
from typing import Dict, Any
import json

from kraken_llm.models import (
    # Модели запросов
    MessageRole,
    ChatMessage,
    FunctionDefinition,
    ToolDefinition,
    ResponseFormat,
    ChatCompletionRequest,
    StreamingChatRequest,
    StructuredOutputRequest,
    FunctionCallingRequest,
    ToolCallingRequest,
    MultimodalMessage,
    MultimodalRequest,
    EmbeddingsRequest,
    ASRRequest,
    TTSRequest,
    BatchRequest,

    # Модели ответов
    FinishReason,
    Usage,
    ResponseChatMessage,
    Choice,
    ChatCompletionResponse,
    StreamDelta,
    StreamChoice,
    ChatCompletionStreamResponse,
    EmbeddingData,
    EmbeddingsResponse,
    ASRSegment,
    ASRResponse,
    TTSResponse,
    ErrorDetail,
    ErrorResponse,
    BatchResult,
    BatchResponse,
    HealthCheckResponse,

    # Модели для function/tool calling
    ParameterType,
    FunctionParameter,
    FunctionSchema,
    ToolFunctionCall,
    ToolCallModel,
    ExecutionStatus,
    ExecutionResult,
    FunctionRegistry,
    ToolRegistry,
    ExecutionContext,
    BatchExecutionRequest,
    BatchExecutionResponse,

    # Модели для потоковых операций
    StreamEventType,
    StreamChunk,
    StreamEvent,
    StreamState,
    StreamMetrics,
    StreamBuffer,
    StreamProcessor,
    SSEParser,
    StreamingRequest,
    StreamingResponse,
    StreamingSession,
)


class TestRequestModels:
    """Тесты моделей запросов."""

    def test_chat_message_valid(self):
        """Тест валидного сообщения чата."""
        message = ChatMessage(
            role=MessageRole.USER,
            content="Привет, как дела?"
        )

        assert message.role == MessageRole.USER
        assert message.content == "Привет, как дела?"
        assert message.name is None

    def test_chat_message_system_without_content(self):
        """Тест системного сообщения без содержимого (должно вызвать ошибку)."""
        with pytest.raises(ValueError, match="Содержимое обязательно для роли MessageRole.SYSTEM"):
            ChatMessage(role=MessageRole.SYSTEM, content="")

    def test_chat_message_tool_without_call_id(self):
        """Тест tool сообщения без call_id (должно вызвать ошибку)."""
        with pytest.raises(ValueError, match="tool_call_id обязателен для роли tool"):
            ChatMessage(
                role=MessageRole.TOOL,
                content="Результат выполнения",
                tool_call_id=None
            )

    def test_function_definition_valid(self):
        """Тест валидного определения функции."""
        func_def = FunctionDefinition(
            name="get_weather",
            description="Получить погоду для города",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Название города"}
                },
                "required": ["city"]
            }
        )

        assert func_def.name == "get_weather"
        assert func_def.description == "Получить погоду для города"
        assert func_def.parameters is not None

    def test_function_definition_invalid_name(self):
        """Тест определения функции с некорректным именем."""
        with pytest.raises(ValueError, match="должно содержать только буквы, цифры и подчеркивания"):
            FunctionDefinition(
                name="get-weather",  # Дефис недопустим
                description="Получить погоду"
            )

    def test_chat_completion_request_valid(self):
        """Тест валидного запроса chat completion."""
        request = ChatCompletionRequest(
            messages=[
                ChatMessage(role=MessageRole.USER, content="Привет!")
            ],
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=100
        )

        assert len(request.messages) == 1
        assert request.model == "gpt-3.5-turbo"
        assert request.temperature == 0.7
        assert request.max_tokens == 100

    def test_chat_completion_request_empty_messages(self):
        """Тест запроса с пустым списком сообщений."""
        with pytest.raises(ValueError, match="Список сообщений не может быть пустым"):
            ChatCompletionRequest(messages=[])

    def test_chat_completion_request_no_user_messages(self):
        """Тест запроса без пользовательских сообщений."""
        with pytest.raises(ValueError, match="Должно быть хотя бы одно сообщение от пользователя"):
            ChatCompletionRequest(
                messages=[
                    ChatMessage(role=MessageRole.SYSTEM, content="Ты помощник")
                ]
            )

    def test_chat_completion_request_functions_and_tools_conflict(self):
        """Тест конфликта между functions и tools."""
        with pytest.raises(ValueError, match="Нельзя использовать functions и tools одновременно"):
            ChatCompletionRequest(
                messages=[
                    ChatMessage(role=MessageRole.USER, content="Привет!")
                ],
                functions=[
                    FunctionDefinition(name="test_func", description="Тест")
                ],
                tools=[
                    ToolDefinition(function=FunctionDefinition(
                        name="test_tool", description="Тест"))
                ]
            )

    def test_embeddings_request_valid(self):
        """Тест валидного запроса embeddings."""
        request = EmbeddingsRequest(
            input="Тестовый текст для векторизации",
            model="text-embedding-ada-002"
        )

        assert request.input == "Тестовый текст для векторизации"
        assert request.model == "text-embedding-ada-002"
        assert request.encoding_format == "float"

    def test_embeddings_request_empty_input(self):
        """Тест запроса embeddings с пустым входом."""
        with pytest.raises(ValueError, match="Входной текст не может быть пустым"):
            EmbeddingsRequest(input="")

    def test_asr_request_valid(self):
        """Тест валидного запроса ASR."""
        request = ASRRequest(
            file="audio.wav",
            model="whisper-1",
            language="ru"
        )

        assert request.file == "audio.wav"
        assert request.model == "whisper-1"
        assert request.language == "ru"
        assert request.response_format == "json"

    def test_tts_request_valid(self):
        """Тест валидного запроса TTS."""
        request = TTSRequest(
            input="Привет, мир!",
            model="tts-1",
            voice="alloy"
        )

        assert request.input == "Привет, мир!"
        assert request.model == "tts-1"
        assert request.voice == "alloy"

    def test_tts_request_too_long_input(self):
        """Тест запроса TTS со слишком длинным текстом."""
        long_text = "А" * 5000  # Больше 4096 символов

        with pytest.raises(ValueError, match="не может быть длиннее 4096 символов"):
            TTSRequest(input=long_text)


class TestResponseModels:
    """Тесты моделей ответов."""

    def test_usage_valid(self):
        """Тест валидной модели использования токенов."""
        usage = Usage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30
        )

        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30

    def test_chat_completion_response_valid(self):
        """Тест валидного ответа chat completion."""
        import time

        response = ChatCompletionResponse(
            id="chatcmpl-123",
            object="chat.completion",
            created=int(time.time()),
            model="gpt-3.5-turbo",
            choices=[
                Choice(
                    index=0,
                    message=ResponseChatMessage(
                        role="assistant",
                        content="Привет! Как дела?"
                    ),
                    finish_reason=FinishReason.STOP
                )
            ],
            usage=Usage(
                prompt_tokens=10,
                completion_tokens=15,
                total_tokens=25
            )
        )

        assert response.id == "chatcmpl-123"
        assert len(response.choices) == 1
        assert response.choices[0].message.content == "Привет! Как дела?"

    def test_chat_completion_response_empty_choices(self):
        """Тест ответа без вариантов выбора."""
        import time

        with pytest.raises(ValueError, match="Должен быть хотя бы один вариант ответа"):
            ChatCompletionResponse(
                id="chatcmpl-123",
                object="chat.completion",
                created=int(time.time()),
                model="gpt-3.5-turbo",
                choices=[]
            )

    def test_embeddings_response_valid(self):
        """Тест валидного ответа embeddings."""
        response = EmbeddingsResponse(
            object="list",
            data=[
                EmbeddingData(
                    object="embedding",
                    embedding=[0.1, 0.2, 0.3, -0.1, -0.2],
                    index=0
                )
            ],
            model="text-embedding-ada-002",
            usage=Usage(
                prompt_tokens=5,
                completion_tokens=0,
                total_tokens=5
            )
        )

        assert len(response.data) == 1
        assert len(response.data[0].embedding) == 5
        assert response.data[0].embedding[0] == 0.1

    def test_asr_response_valid(self):
        """Тест валидного ответа ASR."""
        response = ASRResponse(
            task="transcribe",
            language="ru",
            duration=10.5,
            text="Привет, как дела?",
            segments=[
                ASRSegment(
                    id=0,
                    seek=0,
                    start=0.0,
                    end=2.5,
                    text="Привет,",
                    tokens=[1234, 5678],
                    temperature=0.0,
                    avg_logprob=-0.5,
                    compression_ratio=1.2,
                    no_speech_prob=0.1
                )
            ]
        )

        assert response.task == "transcribe"
        assert response.language == "ru"
        assert response.duration == 10.5
        assert len(response.segments) == 1

    def test_error_response_valid(self):
        """Тест валидного ответа с ошибкой."""
        error_response = ErrorResponse(
            error=ErrorDetail(
                code="invalid_request_error",
                message="Некорректный запрос",
                param="messages",
                type="invalid_request_error"
            )
        )

        assert error_response.error.code == "invalid_request_error"
        assert error_response.error.message == "Некорректный запрос"


class TestToolModels:
    """Тесты моделей для function/tool calling."""

    def test_function_parameter_valid(self):
        """Тест валидного параметра функции."""
        param = FunctionParameter(
            type=ParameterType.STRING,
            description="Название города",
            min_length=1,
            max_length=100
        )

        assert param.type == ParameterType.STRING
        assert param.description == "Название города"
        assert param.min_length == 1
        assert param.max_length == 100

    def test_function_parameter_invalid_length_for_non_string(self):
        """Тест параметра с длиной для не-строкового типа."""
        with pytest.raises(ValueError, match="применимо только к строковым параметрам"):
            FunctionParameter(
                type=ParameterType.INTEGER,
                min_length=1
            )

    def test_function_schema_valid(self):
        """Тест валидной схемы функции."""
        schema = FunctionSchema(
            name="get_weather",
            description="Получить погоду для города",
            parameters={
                "city": FunctionParameter(
                    type=ParameterType.STRING,
                    description="Название города"
                )
            },
            required_parameters=["city"]
        )

        assert schema.name == "get_weather"
        assert "city" in schema.parameters
        assert "city" in schema.required_parameters

    def test_function_schema_to_openai_format(self):
        """Тест конвертации схемы в формат OpenAI."""
        schema = FunctionSchema(
            name="get_weather",
            description="Получить погоду",
            parameters={
                "city": FunctionParameter(
                    type=ParameterType.STRING,
                    description="Город"
                )
            },
            required_parameters=["city"]
        )

        openai_format = schema.to_openai_format()

        assert openai_format["name"] == "get_weather"
        assert openai_format["description"] == "Получить погоду"
        assert "parameters" in openai_format
        assert openai_format["parameters"]["type"] == "object"
        assert "city" in openai_format["parameters"]["properties"]
        assert openai_format["parameters"]["required"] == ["city"]

    def test_function_call_from_openai_format(self):
        """Тест создания вызова функции из формата OpenAI."""
        call = ToolFunctionCall.from_openai_format(
            name="get_weather",
            arguments_json='{"city": "Москва"}',
            call_id="call_123"
        )

        assert call.name == "get_weather"
        assert call.arguments == {"city": "Москва"}
        assert call.call_id == "call_123"

    def test_function_call_invalid_json(self):
        """Тест создания вызова функции с некорректным JSON."""
        with pytest.raises(ValueError, match="Некорректный JSON в аргументах"):
            ToolFunctionCall.from_openai_format(
                name="get_weather",
                arguments_json='{"city": "Москва"',  # Некорректный JSON
                call_id="call_123"
            )

    def test_execution_result_success(self):
        """Тест успешного результата выполнения."""
        result = ExecutionResult(
            function_name="get_weather",
            status=ExecutionStatus.SUCCESS,
            result={"temperature": 20, "condition": "sunny"},
            execution_time=0.5
        )

        assert result.success is True
        assert result.function_name == "get_weather"
        assert result.result["temperature"] == 20

    def test_execution_result_error(self):
        """Тест результата выполнения с ошибкой."""
        result = ExecutionResult(
            function_name="get_weather",
            status=ExecutionStatus.ERROR,
            error="API недоступен",
            error_type="NetworkError",
            execution_time=1.0
        )

        assert result.success is False
        assert result.error == "API недоступен"
        assert result.error_type == "NetworkError"

    def test_function_registry_register_function(self):
        """Тест регистрации функции в реестре."""
        registry = FunctionRegistry()

        def test_function(city: str) -> str:
            """Тестовая функция для получения погоды."""
            return f"Погода в {city}: солнечно"

        schema = registry.register_function(test_function)

        assert schema.name == "test_function"
        assert "city" in schema.parameters
        assert schema.parameters["city"].type == ParameterType.STRING
        assert "city" in schema.required_parameters

    def test_execution_context_permissions(self):
        """Тест контекста выполнения с разрешениями."""
        context = ExecutionContext(
            user_id="user_123",
            permissions=["read_weather", "write_calendar"]
        )

        assert context.has_permission("read_weather") is True
        assert context.has_permission("delete_files") is False


class TestStreamingModels:
    """Тесты моделей для потоковых операций."""

    def test_stream_chunk_valid(self):
        """Тест валидного chunk потока."""
        chunk = StreamChunk(
            id="chunk_1",
            sequence=0,
            content="Привет",
            metadata={"model": "gpt-3.5-turbo"}
        )

        assert chunk.id == "chunk_1"
        assert chunk.sequence == 0
        assert chunk.content == "Привет"
        assert chunk.has_content is True
        assert chunk.is_final is False

    def test_stream_chunk_final(self):
        """Тест финального chunk потока."""
        chunk = StreamChunk(
            id="chunk_final",
            sequence=5,
            content="",
            finish_reason="stop"
        )

        assert chunk.is_final is True
        assert chunk.has_content is False

    def test_stream_buffer_add_chunk(self):
        """Тест добавления chunk в буфер."""
        buffer = StreamBuffer()

        chunk1 = StreamChunk(id="1", sequence=0, content="Привет")
        chunk2 = StreamChunk(id="2", sequence=1, content=" мир")

        buffer.add_chunk(chunk1)
        buffer.add_chunk(chunk2)

        assert len(buffer.chunks) == 2
        assert buffer.aggregated_content == "Привет мир"
        assert buffer.metrics.total_chunks == 2
        assert buffer.metrics.total_characters == 10

    def test_stream_metrics_update(self):
        """Тест обновления метрик потока."""
        metrics = StreamMetrics()

        chunk = StreamChunk(
            id="test",
            sequence=0,
            content="Тест",
            timestamp=datetime.now()
        )

        metrics.update_with_chunk(chunk)

        assert metrics.total_chunks == 1
        assert metrics.total_characters == 4
        assert metrics.start_time is not None

    def test_sse_parser_parse_line(self):
        """Тест парсинга строки SSE."""
        # Тест обычных данных
        result = SSEParser.parse_sse_line('data: {"content": "Привет"}')
        assert result == {"content": "Привет"}

        # Тест маркера завершения
        result = SSEParser.parse_sse_line('data: [DONE]')
        assert result == {"type": "done"}

        # Тест пустой строки
        result = SSEParser.parse_sse_line('')
        assert result is None

        # Тест комментария
        result = SSEParser.parse_sse_line(': это комментарий')
        assert result is None

    def test_sse_parser_parse_openai_chunk(self):
        """Тест парсинга chunk от OpenAI."""
        data = {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "gpt-3.5-turbo",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Привет"},
                    "finish_reason": None
                }
            ]
        }

        chunk = SSEParser.parse_openai_chunk(data, 0)

        assert chunk is not None
        assert chunk.id == "chatcmpl-123"
        assert chunk.content == "Привет"
        assert chunk.sequence == 0
        assert chunk.metadata["model"] == "gpt-3.5-turbo"

    def test_streaming_session_process_data(self):
        """Тест обработки данных в потоковой сессии."""
        session = StreamingSession(
            session_id="session_123",
            request=StreamingRequest()
        )

        # Имитируем SSE данные
        sse_data = 'data: {"choices": [{"delta": {"content": "Привет"}}]}'

        # Тест асинхронной обработки
        import asyncio

        async def test_process():
            events = await session.process_stream_data(sse_data)
            return events

        events = asyncio.run(test_process())

        # Проверяем, что события были созданы
        assert isinstance(events, list)


if __name__ == "__main__":
    # Запуск тестов
    pytest.main([__file__, "-v"])
