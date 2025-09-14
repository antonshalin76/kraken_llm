"""
Модуль для работы с function calling и tool calling в Kraken LLM фреймворке.

Этот модуль предоставляет реестры для регистрации функций и инструментов,
а также исполнитель для их выполнения в контексте OpenAI API.
"""

from .functions import (
    FunctionRegistry,
    FunctionSchema,
    FunctionCall,
    FunctionResult,
    default_function_registry,
    register_function
)

from .tools import (
    ToolRegistry,
    ToolSchema,
    ToolFunction,
    ToolCall,
    ToolResult,
    default_tool_registry,
    register_tool
)

from .executor import (
    FunctionToolExecutor,
    ExecutionContext,
    ExecutionResult,
    default_executor,
    execute_function_calls,
    execute_tool_calls
)

__all__ = [
    # Function calling
    "FunctionRegistry",
    "FunctionSchema", 
    "FunctionCall",
    "FunctionResult",
    "default_function_registry",
    "register_function",
    
    # Tool calling
    "ToolRegistry",
    "ToolSchema",
    "ToolFunction",
    "ToolCall", 
    "ToolResult",
    "default_tool_registry",
    "register_tool",
    
    # Execution
    "FunctionToolExecutor",
    "ExecutionContext",
    "ExecutionResult",
    "default_executor",
    "execute_function_calls",
    "execute_tool_calls"
]