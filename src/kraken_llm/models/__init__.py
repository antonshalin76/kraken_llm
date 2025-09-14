"""
Модели данных Kraken LLM фреймворка

Pydantic модели для запросов, ответов, function/tool calling и потоковых операций.
"""

# Модели запросов
from .requests import (
    MessageRole,
    ChatMessage,
    FunctionDefinition,
    ToolDefinition,
    ResponseFormat,
    ChatCompletionRequest,
    StreamingRequest as StreamingChatRequest,
    StructuredOutputRequest,
    FunctionCallingRequest,
    ToolCallingRequest,
    MultimodalMessage,
    MultimodalRequest,
    EmbeddingsRequest,
    ASRRequest,
    TTSRequest,
    BatchRequest,
)

# Модели ответов
from .responses import (
    FinishReason,
    Usage,
    FunctionCall,
    ToolCall,
    ChatMessage as ResponseChatMessage,
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
)

# Модели для function/tool calling
from .tools import (
    ParameterType,
    FunctionParameter,
    FunctionSchema,
    FunctionCall as ToolFunctionCall,
    ToolCall as ToolCallModel,
    ExecutionStatus,
    ExecutionResult,
    FunctionRegistry,
    ToolRegistry,
    ExecutionContext,
    BatchExecutionRequest,
    BatchExecutionResponse,
)

# Модели для потоковых операций
from .streaming import (
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

__all__ = [
    # Модели запросов
    "MessageRole",
    "ChatMessage",
    "FunctionDefinition",
    "ToolDefinition",
    "ResponseFormat",
    "ChatCompletionRequest",
    "StreamingChatRequest",
    "StructuredOutputRequest",
    "FunctionCallingRequest",
    "ToolCallingRequest",
    "MultimodalMessage",
    "MultimodalRequest",
    "EmbeddingsRequest",
    "ASRRequest",
    "TTSRequest",
    "BatchRequest",
    
    # Модели ответов
    "FinishReason",
    "Usage",
    "FunctionCall",
    "ToolCall",
    "ResponseChatMessage",
    "Choice",
    "ChatCompletionResponse",
    "StreamDelta",
    "StreamChoice",
    "ChatCompletionStreamResponse",
    "EmbeddingData",
    "EmbeddingsResponse",
    "ASRSegment",
    "ASRResponse",
    "TTSResponse",
    "ErrorDetail",
    "ErrorResponse",
    "BatchResult",
    "BatchResponse",
    "HealthCheckResponse",
    
    # Модели для function/tool calling
    "ParameterType",
    "FunctionParameter",
    "FunctionSchema",
    "ToolFunctionCall",
    "ToolCallModel",
    "ExecutionStatus",
    "ExecutionResult",
    "FunctionRegistry",
    "ToolRegistry",
    "ExecutionContext",
    "BatchExecutionRequest",
    "BatchExecutionResponse",
    
    # Модели для потоковых операций
    "StreamEventType",
    "StreamChunk",
    "StreamEvent",
    "StreamState",
    "StreamMetrics",
    "StreamBuffer",
    "StreamProcessor",
    "SSEParser",
    "StreamingRequest",
    "StreamingResponse",
    "StreamingSession",
]