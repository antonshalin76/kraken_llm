"""
Реестр инструментов для tool calling в Kraken LLM фреймворке.

Этот модуль предоставляет ToolRegistry для регистрации и выполнения
инструментов в контексте OpenAI tool calling с поддержкой параллельного выполнения.
"""

import json
import asyncio
import inspect
from typing import Dict, Callable, Any, List, Optional, Union, Awaitable
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class ToolFunction(BaseModel):
    """Определение функции инструмента"""
    name: str = Field(..., description="Имя функции")
    description: str = Field(..., description="Описание функции")
    parameters: Dict[str, Any] = Field(..., description="Параметры функции в JSON Schema формате")


class ToolSchema(BaseModel):
    """Схема инструмента для OpenAI tool calling"""
    type: str = Field(default="function", description="Тип инструмента")
    function: ToolFunction = Field(..., description="Определение функции")


class ToolCall(BaseModel):
    """Модель для вызова инструмента"""
    id: str = Field(..., description="Уникальный ID вызова инструмента")
    type: str = Field(default="function", description="Тип инструмента")
    function: Dict[str, Any] = Field(..., description="Данные вызова функции")


class ToolResult(BaseModel):
    """Результат выполнения инструмента"""
    tool_call_id: str = Field(..., description="ID вызова инструмента")
    name: str = Field(..., description="Имя выполненного инструмента")
    result: Any = Field(..., description="Результат выполнения")
    success: bool = Field(..., description="Успешность выполнения")
    error: Optional[str] = Field(None, description="Сообщение об ошибке, если есть")


class ToolRegistry:
    """
    Реестр инструментов для tool calling.
    
    Позволяет регистрировать Python функции как инструменты и выполнять их
    параллельно в контексте OpenAI tool calling API.
    """
    
    def __init__(self):
        """Инициализация реестра инструментов"""
        self._tools: Dict[str, Callable] = {}
        self._schemas: Dict[str, ToolSchema] = {}
        logger.info("Инициализирован ToolRegistry")
    
    def register_tool(
        self,
        name: str,
        tool: Callable,
        description: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Регистрация инструмента в реестре.
        
        Args:
            name: Имя инструмента для использования в LLM
            tool: Python функция для выполнения (может быть async)
            description: Описание инструмента для LLM
            parameters: JSON Schema параметров (если None, генерируется автоматически)
        """
        if name in self._tools:
            logger.warning(f"Инструмент '{name}' уже зарегистрирован, перезаписываем")
        
        # Автоматическая генерация схемы параметров если не предоставлена
        if parameters is None:
            parameters = self._generate_parameters_schema(tool)
        
        # Создание схемы инструмента
        tool_function = ToolFunction(
            name=name,
            description=description,
            parameters=parameters
        )
        
        schema = ToolSchema(
            type="function",
            function=tool_function
        )
        
        # Регистрация
        self._tools[name] = tool
        self._schemas[name] = schema
        
        logger.info(f"Зарегистрирован инструмент '{name}': {description}")
    
    def _generate_parameters_schema(self, tool: Callable) -> Dict[str, Any]:
        """
        Автоматическая генерация JSON Schema для параметров инструмента.
        
        Args:
            tool: Python функция
            
        Returns:
            JSON Schema для параметров инструмента
        """
        try:
            signature = inspect.signature(tool)
            properties = {}
            required = []
            
            for param_name, param in signature.parameters.items():
                # Пропускаем self и *args, **kwargs
                if param_name in ['self'] or param.kind in [param.VAR_POSITIONAL, param.VAR_KEYWORD]:
                    continue
                
                # Определяем тип параметра
                param_type = "string"  # по умолчанию
                if param.annotation != param.empty:
                    if param.annotation == int:
                        param_type = "integer"
                    elif param.annotation == float:
                        param_type = "number"
                    elif param.annotation == bool:
                        param_type = "boolean"
                    elif param.annotation == list:
                        param_type = "array"
                    elif param.annotation == dict:
                        param_type = "object"
                
                properties[param_name] = {
                    "type": param_type,
                    "description": f"Параметр {param_name}"
                }
                
                # Если нет значения по умолчанию, параметр обязательный
                if param.default == param.empty:
                    required.append(param_name)
            
            return {
                "type": "object",
                "properties": properties,
                "required": required
            }
            
        except Exception as e:
            logger.warning(f"Не удалось сгенерировать схему для инструмента: {e}")
            return {
                "type": "object",
                "properties": {},
                "required": []
            }
    
    def get_tool_schema(self, name: str) -> Optional[ToolSchema]:
        """
        Получение схемы инструмента по имени.
        
        Args:
            name: Имя инструмента
            
        Returns:
            Схема инструмента или None если не найден
        """
        return self._schemas.get(name)
    
    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """
        Получение всех схем инструментов в формате OpenAI.
        
        Returns:
            Список схем инструментов для передачи в OpenAI API
        """
        schemas = []
        for schema in self._schemas.values():
            schemas.append({
                "type": schema.type,
                "function": {
                    "name": schema.function.name,
                    "description": schema.function.description,
                    "parameters": schema.function.parameters
                }
            })
        return schemas
    
    async def execute_tool(self, name: str, arguments: Union[str, Dict[str, Any]], tool_call_id: str = "") -> ToolResult:
        """
        Выполнение зарегистрированного инструмента.
        
        Args:
            name: Имя инструмента
            arguments: Аргументы инструмента (JSON строка или словарь)
            tool_call_id: ID вызова инструмента
            
        Returns:
            Результат выполнения инструмента
        """
        if name not in self._tools:
            error_msg = f"Инструмент '{name}' не зарегистрирован"
            logger.error(error_msg)
            return ToolResult(
                tool_call_id=tool_call_id,
                name=name,
                result=None,
                success=False,
                error=error_msg
            )
        
        try:
            # Парсинг аргументов если это строка
            if isinstance(arguments, str):
                try:
                    parsed_args = json.loads(arguments)
                except json.JSONDecodeError as e:
                    error_msg = f"Ошибка парсинга JSON аргументов: {e}"
                    logger.error(error_msg)
                    return ToolResult(
                        tool_call_id=tool_call_id,
                        name=name,
                        result=None,
                        success=False,
                        error=error_msg
                    )
            else:
                parsed_args = arguments
            
            # Выполнение инструмента
            tool = self._tools[name]
            logger.info(f"Выполняется инструмент '{name}' с аргументами: {parsed_args}")
            
            # Проверяем, является ли функция асинхронной
            if asyncio.iscoroutinefunction(tool):
                if isinstance(parsed_args, dict):
                    result = await tool(**parsed_args)
                else:
                    result = await tool(parsed_args)
            else:
                if isinstance(parsed_args, dict):
                    result = tool(**parsed_args)
                else:
                    result = tool(parsed_args)
            
            logger.info(f"Инструмент '{name}' выполнен успешно")
            return ToolResult(
                tool_call_id=tool_call_id,
                name=name,
                result=result,
                success=True,
                error=None
            )
            
        except Exception as e:
            error_msg = f"Ошибка выполнения инструмента '{name}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            return ToolResult(
                tool_call_id=tool_call_id,
                name=name,
                result=None,
                success=False,
                error=error_msg
            )
    
    async def execute_parallel_tools(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """
        Параллельное выполнение нескольких инструментов.
        
        Args:
            tool_calls: Список вызовов инструментов
            
        Returns:
            Список результатов выполнения инструментов
        """
        if not tool_calls:
            return []
        
        logger.info(f"Начинается параллельное выполнение {len(tool_calls)} инструментов")
        
        # Создаем задачи для параллельного выполнения
        tasks = []
        for tool_call in tool_calls:
            function_name = tool_call.function.get("name", "")
            function_args = tool_call.function.get("arguments", {})
            
            task = self.execute_tool(
                name=function_name,
                arguments=function_args,
                tool_call_id=tool_call.id
            )
            tasks.append(task)
        
        # Выполняем все задачи параллельно
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Обрабатываем результаты и исключения
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    error_msg = f"Исключение при выполнении инструмента: {str(result)}"
                    logger.error(error_msg)
                    processed_results.append(ToolResult(
                        tool_call_id=tool_calls[i].id,
                        name=tool_calls[i].function.get("name", "unknown"),
                        result=None,
                        success=False,
                        error=error_msg
                    ))
                else:
                    processed_results.append(result)
            
            logger.info(f"Завершено параллельное выполнение {len(tool_calls)} инструментов")
            return processed_results
            
        except Exception as e:
            error_msg = f"Ошибка при параллельном выполнении инструментов: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Возвращаем ошибки для всех вызовов
            return [
                ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.function.get("name", "unknown"),
                    result=None,
                    success=False,
                    error=error_msg
                )
                for tool_call in tool_calls
            ]
    
    def list_tools(self) -> List[str]:
        """
        Получение списка имен всех зарегистрированных инструментов.
        
        Returns:
            Список имен инструментов
        """
        return list(self._tools.keys())
    
    def unregister_tool(self, name: str) -> bool:
        """
        Удаление инструмента из реестра.
        
        Args:
            name: Имя инструмента для удаления
            
        Returns:
            True если инструмент был удален, False если не найден
        """
        if name in self._tools:
            del self._tools[name]
            del self._schemas[name]
            logger.info(f"Инструмент '{name}' удален из реестра")
            return True
        return False
    
    def clear(self) -> None:
        """Очистка всех зарегистрированных инструментов"""
        count = len(self._tools)
        self._tools.clear()
        self._schemas.clear()
        logger.info(f"Очищен реестр инструментов, удалено {count} инструментов")
    
    def __len__(self) -> int:
        """Количество зарегистрированных инструментов"""
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        """Проверка наличия инструмента в реестре"""
        return name in self._tools
    
    def __repr__(self) -> str:
        """Строковое представление реестра"""
        return f"ToolRegistry({len(self._tools)} tools: {list(self._tools.keys())})"


# Глобальный экземпляр реестра для удобства использования
default_tool_registry = ToolRegistry()


def register_tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None
):
    """
    Декоратор для регистрации инструментов в глобальном реестре.
    
    Args:
        name: Имя инструмента (если None, используется имя Python функции)
        description: Описание инструмента (если None, используется docstring)
        parameters: JSON Schema параметров (если None, генерируется автоматически)
    
    Example:
        @register_tool(description="Выполняет математические вычисления")
        async def calculate(expression: str) -> float:
            return eval(expression)
    """
    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or f"Инструмент {tool_name}"
        
        default_tool_registry.register_tool(
            name=tool_name,
            tool=func,
            description=tool_description,
            parameters=parameters
        )
        
        return func
    
    return decorator