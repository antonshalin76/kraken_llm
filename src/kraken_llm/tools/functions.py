"""
Реестр функций для function calling в Kraken LLM фреймворке.

Этот модуль предоставляет FunctionRegistry для регистрации и выполнения
функций в контексте OpenAI function calling.
"""

import json
import inspect
from typing import Dict, Callable, Any, List, Optional, Union
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class FunctionSchema(BaseModel):
    """Схема функции для OpenAI function calling"""
    name: str = Field(..., description="Имя функции")
    description: str = Field(..., description="Описание функции")
    parameters: Dict[str, Any] = Field(..., description="Параметры функции в JSON Schema формате")


class FunctionCall(BaseModel):
    """Модель для вызова функции"""
    name: str = Field(..., description="Имя вызываемой функции")
    arguments: Union[str, Dict[str, Any]] = Field(..., description="Аргументы функции")


class FunctionResult(BaseModel):
    """Результат выполнения функции"""
    name: str = Field(..., description="Имя выполненной функции")
    result: Any = Field(..., description="Результат выполнения")
    success: bool = Field(..., description="Успешность выполнения")
    error: Optional[str] = Field(None, description="Сообщение об ошибке, если есть")


class FunctionRegistry:
    """
    Реестр функций для function calling.
    
    Позволяет регистрировать Python функции и автоматически генерировать
    схемы для OpenAI function calling API.
    """
    
    def __init__(self):
        """Инициализация реестра функций"""
        self._functions: Dict[str, Callable] = {}
        self._schemas: Dict[str, FunctionSchema] = {}
        logger.info("Инициализирован FunctionRegistry")
    
    def register_function(
        self,
        name: str,
        func: Callable,
        description: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Регистрация функции в реестре.
        
        Args:
            name: Имя функции для использования в LLM
            func: Python функция для выполнения
            description: Описание функции для LLM
            parameters: JSON Schema параметров (если None, генерируется автоматически)
        """
        if name in self._functions:
            logger.warning(f"Функция '{name}' уже зарегистрирована, перезаписываем")
        
        # Автоматическая генерация схемы параметров если не предоставлена
        if parameters is None:
            parameters = self._generate_parameters_schema(func)
        
        # Создание схемы функции
        schema = FunctionSchema(
            name=name,
            description=description,
            parameters=parameters
        )
        
        # Регистрация
        self._functions[name] = func
        self._schemas[name] = schema
        
        logger.info(f"Зарегистрирована функция '{name}': {description}")
    
    def _generate_parameters_schema(self, func: Callable) -> Dict[str, Any]:
        """
        Автоматическая генерация JSON Schema для параметров функции.
        
        Args:
            func: Python функция
            
        Returns:
            JSON Schema для параметров функции
        """
        try:
            signature = inspect.signature(func)
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
            logger.warning(f"Не удалось сгенерировать схему для функции: {e}")
            return {
                "type": "object",
                "properties": {},
                "required": []
            }
    
    def get_function_schema(self, name: str) -> Optional[FunctionSchema]:
        """
        Получение схемы функции по имени.
        
        Args:
            name: Имя функции
            
        Returns:
            Схема функции или None если не найдена
        """
        return self._schemas.get(name)
    
    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """
        Получение всех схем функций в формате OpenAI.
        
        Returns:
            Список схем функций для передачи в OpenAI API
        """
        schemas = []
        for schema in self._schemas.values():
            schemas.append({
                "name": schema.name,
                "description": schema.description,
                "parameters": schema.parameters
            })
        return schemas
    
    def execute_function(self, name: str, arguments: Union[str, Dict[str, Any]]) -> FunctionResult:
        """
        Выполнение зарегистрированной функции.
        
        Args:
            name: Имя функции
            arguments: Аргументы функции (JSON строка или словарь)
            
        Returns:
            Результат выполнения функции
        """
        if name not in self._functions:
            error_msg = f"Функция '{name}' не зарегистрирована"
            logger.error(error_msg)
            return FunctionResult(
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
                    return FunctionResult(
                        name=name,
                        result=None,
                        success=False,
                        error=error_msg
                    )
            else:
                parsed_args = arguments
            
            # Выполнение функции
            func = self._functions[name]
            logger.info(f"Выполняется функция '{name}' с аргументами: {parsed_args}")
            
            if isinstance(parsed_args, dict):
                result = func(**parsed_args)
            else:
                result = func(parsed_args)
            
            logger.info(f"Функция '{name}' выполнена успешно")
            return FunctionResult(
                name=name,
                result=result,
                success=True,
                error=None
            )
            
        except Exception as e:
            error_msg = f"Ошибка выполнения функции '{name}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            return FunctionResult(
                name=name,
                result=None,
                success=False,
                error=error_msg
            )
    
    def list_functions(self) -> List[str]:
        """
        Получение списка имен всех зарегистрированных функций.
        
        Returns:
            Список имен функций
        """
        return list(self._functions.keys())
    
    def unregister_function(self, name: str) -> bool:
        """
        Удаление функции из реестра.
        
        Args:
            name: Имя функции для удаления
            
        Returns:
            True если функция была удалена, False если не найдена
        """
        if name in self._functions:
            del self._functions[name]
            del self._schemas[name]
            logger.info(f"Функция '{name}' удалена из реестра")
            return True
        return False
    
    def clear(self) -> None:
        """Очистка всех зарегистрированных функций"""
        count = len(self._functions)
        self._functions.clear()
        self._schemas.clear()
        logger.info(f"Очищен реестр функций, удалено {count} функций")
    
    def __len__(self) -> int:
        """Количество зарегистрированных функций"""
        return len(self._functions)
    
    def __contains__(self, name: str) -> bool:
        """Проверка наличия функции в реестре"""
        return name in self._functions
    
    def __repr__(self) -> str:
        """Строковое представление реестра"""
        return f"FunctionRegistry({len(self._functions)} functions: {list(self._functions.keys())})"


# Глобальный экземпляр реестра для удобства использования
default_function_registry = FunctionRegistry()


def register_function(
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None
):
    """
    Декоратор для регистрации функций в глобальном реестре.
    
    Args:
        name: Имя функции (если None, используется имя Python функции)
        description: Описание функции (если None, используется docstring)
        parameters: JSON Schema параметров (если None, генерируется автоматически)
    
    Example:
        @register_function(description="Получает текущую погоду")
        def get_weather(city: str) -> str:
            return f"Погода в {city}: солнечно"
    """
    def decorator(func: Callable) -> Callable:
        func_name = name or func.__name__
        func_description = description or func.__doc__ or f"Функция {func_name}"
        
        default_function_registry.register_function(
            name=func_name,
            func=func,
            description=func_description,
            parameters=parameters
        )
        
        return func
    
    return decorator