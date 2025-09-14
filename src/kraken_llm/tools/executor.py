"""
Исполнитель функций и инструментов для Kraken LLM фреймворка.

Этот модуль предоставляет унифицированный интерфейс для выполнения
как function calling, так и tool calling операций.
"""

import json
import asyncio
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
import logging

from .functions import FunctionRegistry, FunctionCall, FunctionResult, default_function_registry
from .tools import ToolRegistry, ToolCall, ToolResult, default_tool_registry

logger = logging.getLogger(__name__)


class ExecutionContext(BaseModel):
    """Контекст выполнения функций/инструментов"""
    request_id: Optional[str] = Field(None, description="ID запроса")
    user_id: Optional[str] = Field(None, description="ID пользователя")
    session_id: Optional[str] = Field(None, description="ID сессии")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Дополнительные метаданные")


class ExecutionResult(BaseModel):
    """Результат выполнения функций/инструментов"""
    success: bool = Field(..., description="Общий статус выполнения")
    results: List[Union[FunctionResult, ToolResult]] = Field(..., description="Результаты выполнения")
    execution_time: Optional[float] = Field(None, description="Время выполнения в секундах")
    context: Optional[ExecutionContext] = Field(None, description="Контекст выполнения")


class FunctionToolExecutor:
    """
    Унифицированный исполнитель функций и инструментов.
    
    Предоставляет единый интерфейс для выполнения как function calling,
    так и tool calling операций с поддержкой параллельного выполнения.
    """
    
    def __init__(
        self,
        function_registry: Optional[FunctionRegistry] = None,
        tool_registry: Optional[ToolRegistry] = None
    ):
        """
        Инициализация исполнителя.
        
        Args:
            function_registry: Реестр функций (если None, используется глобальный)
            tool_registry: Реестр инструментов (если None, используется глобальный)
        """
        self.function_registry = function_registry if function_registry is not None else default_function_registry
        self.tool_registry = tool_registry if tool_registry is not None else default_tool_registry
        logger.info("Инициализирован FunctionToolExecutor")
    
    async def execute_function_calls(
        self,
        function_calls: List[Dict[str, Any]],
        context: Optional[ExecutionContext] = None
    ) -> ExecutionResult:
        """
        Выполнение function calls.
        
        Args:
            function_calls: Список вызовов функций в формате OpenAI
            context: Контекст выполнения
            
        Returns:
            Результат выполнения всех функций
        """
        import time
        start_time = time.time()
        
        logger.info(f"Начинается выполнение {len(function_calls)} function calls")
        
        results = []
        all_success = True
        
        try:
            # Выполняем функции последовательно (function calling не поддерживает параллельность)
            for func_call_data in function_calls:
                function_call = FunctionCall(
                    name=func_call_data.get("name", ""),
                    arguments=func_call_data.get("arguments", {})
                )
                
                result = self.function_registry.execute_function(
                    name=function_call.name,
                    arguments=function_call.arguments
                )
                
                results.append(result)
                if not result.success:
                    all_success = False
            
            execution_time = time.time() - start_time
            logger.info(f"Завершено выполнение function calls за {execution_time:.3f}s")
            
            return ExecutionResult(
                success=all_success,
                results=results,
                execution_time=execution_time,
                context=context
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Ошибка при выполнении function calls: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Создаем результаты с ошибками для всех вызовов
            error_results = [
                FunctionResult(
                    name=func_call.get("name", "unknown"),
                    result=None,
                    success=False,
                    error=error_msg
                )
                for func_call in function_calls
            ]
            
            return ExecutionResult(
                success=False,
                results=error_results,
                execution_time=execution_time,
                context=context
            )
    
    async def execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        context: Optional[ExecutionContext] = None
    ) -> ExecutionResult:
        """
        Выполнение tool calls с поддержкой параллельности.
        
        Args:
            tool_calls: Список вызовов инструментов в формате OpenAI
            context: Контекст выполнения
            
        Returns:
            Результат выполнения всех инструментов
        """
        import time
        start_time = time.time()
        
        logger.info(f"Начинается выполнение {len(tool_calls)} tool calls")
        
        try:
            # Преобразуем в формат ToolCall
            parsed_tool_calls = []
            for tool_call_data in tool_calls:
                tool_call = ToolCall(
                    id=tool_call_data.get("id", ""),
                    type=tool_call_data.get("type", "function"),
                    function=tool_call_data.get("function", {})
                )
                parsed_tool_calls.append(tool_call)
            
            # Выполняем инструменты параллельно
            results = await self.tool_registry.execute_parallel_tools(parsed_tool_calls)
            
            execution_time = time.time() - start_time
            all_success = all(result.success for result in results)
            
            logger.info(f"Завершено выполнение tool calls за {execution_time:.3f}s")
            
            return ExecutionResult(
                success=all_success,
                results=results,
                execution_time=execution_time,
                context=context
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Ошибка при выполнении tool calls: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Создаем результаты с ошибками для всех вызовов
            error_results = [
                ToolResult(
                    tool_call_id=tool_call.get("id", ""),
                    name=tool_call.get("function", {}).get("name", "unknown"),
                    result=None,
                    success=False,
                    error=error_msg
                )
                for tool_call in tool_calls
            ]
            
            return ExecutionResult(
                success=False,
                results=error_results,
                execution_time=execution_time,
                context=context
            )
    
    async def execute_mixed_calls(
        self,
        function_calls: Optional[List[Dict[str, Any]]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        context: Optional[ExecutionContext] = None
    ) -> ExecutionResult:
        """
        Выполнение смешанных function и tool calls.
        
        Args:
            function_calls: Список вызовов функций
            tool_calls: Список вызовов инструментов
            context: Контекст выполнения
            
        Returns:
            Объединенный результат выполнения
        """
        import time
        start_time = time.time()
        
        logger.info("Начинается выполнение смешанных function/tool calls")
        
        all_results = []
        all_success = True
        
        try:
            # Выполняем function calls если есть
            if function_calls:
                func_result = await self.execute_function_calls(function_calls, context)
                all_results.extend(func_result.results)
                if not func_result.success:
                    all_success = False
            
            # Выполняем tool calls если есть
            if tool_calls:
                tool_result = await self.execute_tool_calls(tool_calls, context)
                all_results.extend(tool_result.results)
                if not tool_result.success:
                    all_success = False
            
            execution_time = time.time() - start_time
            logger.info(f"Завершено выполнение смешанных calls за {execution_time:.3f}s")
            
            return ExecutionResult(
                success=all_success,
                results=all_results,
                execution_time=execution_time,
                context=context
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Ошибка при выполнении смешанных calls: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return ExecutionResult(
                success=False,
                results=all_results,  # Возвращаем то, что успели выполнить
                execution_time=execution_time,
                context=context
            )
    
    def get_available_functions(self) -> List[Dict[str, Any]]:
        """
        Получение списка доступных функций для OpenAI API.
        
        Returns:
            Список схем функций в формате OpenAI
        """
        return self.function_registry.get_all_schemas()
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Получение списка доступных инструментов для OpenAI API.
        
        Returns:
            Список схем инструментов в формате OpenAI
        """
        return self.tool_registry.get_all_schemas()
    
    def get_all_capabilities(self) -> Dict[str, Any]:
        """
        Получение всех доступных возможностей (функций и инструментов).
        
        Returns:
            Словарь с функциями и инструментами
        """
        return {
            "functions": self.get_available_functions(),
            "tools": self.get_available_tools(),
            "function_count": len(self.function_registry),
            "tool_count": len(self.tool_registry)
        }
    
    async def validate_calls(
        self,
        function_calls: Optional[List[Dict[str, Any]]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Валидация вызовов функций и инструментов.
        
        Args:
            function_calls: Список вызовов функций для валидации
            tool_calls: Список вызовов инструментов для валидации
            
        Returns:
            Результат валидации с деталями
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Валидация function calls
        if function_calls:
            for i, func_call in enumerate(function_calls):
                func_name = func_call.get("name")
                if not func_name:
                    validation_result["valid"] = False
                    validation_result["errors"].append(f"Function call {i}: отсутствует имя функции")
                    continue
                
                if func_name not in self.function_registry:
                    validation_result["valid"] = False
                    validation_result["errors"].append(f"Function call {i}: функция '{func_name}' не зарегистрирована")
                
                # Проверка аргументов
                arguments = func_call.get("arguments")
                if isinstance(arguments, str):
                    try:
                        json.loads(arguments)
                    except json.JSONDecodeError:
                        validation_result["valid"] = False
                        validation_result["errors"].append(f"Function call {i}: некорректный JSON в аргументах")
        
        # Валидация tool calls
        if tool_calls:
            for i, tool_call in enumerate(tool_calls):
                tool_id = tool_call.get("id")
                if not tool_id:
                    validation_result["warnings"].append(f"Tool call {i}: отсутствует ID")
                
                function_data = tool_call.get("function", {})
                tool_name = function_data.get("name")
                
                if not tool_name:
                    validation_result["valid"] = False
                    validation_result["errors"].append(f"Tool call {i}: отсутствует имя инструмента")
                    continue
                
                if tool_name not in self.tool_registry:
                    validation_result["valid"] = False
                    validation_result["errors"].append(f"Tool call {i}: инструмент '{tool_name}' не зарегистрирован")
                
                # Проверка аргументов
                arguments = function_data.get("arguments")
                if isinstance(arguments, str):
                    try:
                        json.loads(arguments)
                    except json.JSONDecodeError:
                        validation_result["valid"] = False
                        validation_result["errors"].append(f"Tool call {i}: некорректный JSON в аргументах")
        
        return validation_result


# Глобальный экземпляр исполнителя для удобства использования
default_executor = FunctionToolExecutor()


async def execute_function_calls(
    function_calls: List[Dict[str, Any]],
    context: Optional[ExecutionContext] = None
) -> ExecutionResult:
    """
    Удобная функция для выполнения function calls через глобальный исполнитель.
    
    Args:
        function_calls: Список вызовов функций
        context: Контекст выполнения
        
    Returns:
        Результат выполнения
    """
    return await default_executor.execute_function_calls(function_calls, context)


async def execute_tool_calls(
    tool_calls: List[Dict[str, Any]],
    context: Optional[ExecutionContext] = None
) -> ExecutionResult:
    """
    Удобная функция для выполнения tool calls через глобальный исполнитель.
    
    Args:
        tool_calls: Список вызовов инструментов
        context: Контекст выполнения
        
    Returns:
        Результат выполнения
    """
    return await default_executor.execute_tool_calls(tool_calls, context)