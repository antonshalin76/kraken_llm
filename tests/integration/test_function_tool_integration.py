"""
Интеграционные тесты для системы function и tool calling в Kraken LLM фреймворке.

Эти тесты демонстрируют реальные примеры использования функций и инструментов.
"""

import pytest
import asyncio
import json
import math
from datetime import datetime
from typing import List, Dict, Any

from kraken_llm.tools import (
    FunctionRegistry,
    ToolRegistry,
    FunctionToolExecutor,
    ExecutionContext,
    register_function,
    register_tool
)


class TestRealWorldFunctions:
    """Тесты с реальными примерами функций"""
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        self.registry = FunctionRegistry()
        
        # Регистрируем набор полезных функций
        self._register_math_functions()
        self._register_string_functions()
        self._register_utility_functions()
    
    def _register_math_functions(self):
        """Регистрация математических функций"""
        
        def calculate_area_circle(radius: float) -> float:
            """Вычисляет площадь круга по радиусу"""
            return math.pi * radius ** 2
        
        def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
            """Вычисляет расстояние между двумя точками"""
            return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        def factorial(n: int) -> int:
            """Вычисляет факториал числа"""
            if n < 0:
                raise ValueError("Факториал определен только для неотрицательных чисел")
            return math.factorial(n)
        
        self.registry.register_function(
            "calculate_area_circle",
            calculate_area_circle,
            "Вычисляет площадь круга по радиусу"
        )
        
        self.registry.register_function(
            "calculate_distance", 
            calculate_distance,
            "Вычисляет расстояние между двумя точками на плоскости"
        )
        
        self.registry.register_function(
            "factorial",
            factorial,
            "Вычисляет факториал числа"
        )
    
    def _register_string_functions(self):
        """Регистрация функций для работы со строками"""
        
        def reverse_string(text: str) -> str:
            """Переворачивает строку"""
            return text[::-1]
        
        def count_words(text: str) -> int:
            """Подсчитывает количество слов в тексте"""
            return len(text.split())
        
        def capitalize_words(text: str) -> str:
            """Делает первую букву каждого слова заглавной"""
            return text.title()
        
        self.registry.register_function(
            "reverse_string",
            reverse_string,
            "Переворачивает строку задом наперед"
        )
        
        self.registry.register_function(
            "count_words",
            count_words,
            "Подсчитывает количество слов в тексте"
        )
        
        self.registry.register_function(
            "capitalize_words",
            capitalize_words,
            "Делает первую букву каждого слова заглавной"
        )
    
    def _register_utility_functions(self):
        """Регистрация утилитарных функций"""
        
        def get_current_time() -> str:
            """Возвращает текущее время"""
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        def format_json(data: dict) -> str:
            """Форматирует словарь в красивый JSON"""
            return json.dumps(data, indent=2, ensure_ascii=False)
        
        def validate_email(email: str) -> bool:
            """Простая валидация email адреса"""
            return "@" in email and "." in email.split("@")[-1]
        
        self.registry.register_function(
            "get_current_time",
            get_current_time,
            "Возвращает текущую дату и время"
        )
        
        self.registry.register_function(
            "format_json",
            format_json,
            "Форматирует словарь в красивый JSON"
        )
        
        self.registry.register_function(
            "validate_email",
            validate_email,
            "Проверяет корректность email адреса"
        )
    
    def test_math_functions(self):
        """Тест математических функций"""
        # Тест площади круга
        result = self.registry.execute_function("calculate_area_circle", {"radius": 5.0})
        assert result.success is True
        assert abs(result.result - (math.pi * 25)) < 0.001
        
        # Тест расстояния между точками
        result = self.registry.execute_function(
            "calculate_distance", 
            {"x1": 0, "y1": 0, "x2": 3, "y2": 4}
        )
        assert result.success is True
        assert result.result == 5.0  # 3-4-5 треугольник
        
        # Тест факториала
        result = self.registry.execute_function("factorial", {"n": 5})
        assert result.success is True
        assert result.result == 120
    
    def test_string_functions(self):
        """Тест функций для работы со строками"""
        # Тест переворота строки
        result = self.registry.execute_function("reverse_string", {"text": "Привет"})
        assert result.success is True
        assert result.result == "тевирП"
        
        # Тест подсчета слов
        result = self.registry.execute_function("count_words", {"text": "Это тестовая строка"})
        assert result.success is True
        assert result.result == 3
        
        # Тест капитализации
        result = self.registry.execute_function("capitalize_words", {"text": "привет мир"})
        assert result.success is True
        assert result.result == "Привет Мир"
    
    def test_utility_functions(self):
        """Тест утилитарных функций"""
        # Тест получения времени
        result = self.registry.execute_function("get_current_time", {})
        assert result.success is True
        assert isinstance(result.result, str)
        assert len(result.result) == 19  # YYYY-MM-DD HH:MM:SS
        
        # Тест форматирования JSON
        test_data = {"name": "Тест", "value": 42}
        result = self.registry.execute_function("format_json", {"data": test_data})
        assert result.success is True
        assert "name" in result.result
        assert "Тест" in result.result
        
        # Тест валидации email
        result = self.registry.execute_function("validate_email", {"email": "test@example.com"})
        assert result.success is True
        assert result.result is True
        
        result = self.registry.execute_function("validate_email", {"email": "invalid-email"})
        assert result.success is True
        assert result.result is False
    
    def test_function_error_handling(self):
        """Тест обработки ошибок в функциях"""
        # Тест факториала с отрицательным числом
        result = self.registry.execute_function("factorial", {"n": -1})
        assert result.success is False
        assert "неотрицательных чисел" in result.error
    
    def test_get_all_function_schemas(self):
        """Тест получения всех схем функций"""
        schemas = self.registry.get_all_schemas()
        
        # Проверяем, что все функции зарегистрированы
        function_names = [schema["name"] for schema in schemas]
        expected_functions = [
            "calculate_area_circle", "calculate_distance", "factorial",
            "reverse_string", "count_words", "capitalize_words",
            "get_current_time", "format_json", "validate_email"
        ]
        
        for func_name in expected_functions:
            assert func_name in function_names
        
        # Проверяем формат схем
        for schema in schemas:
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema
            assert schema["parameters"]["type"] == "object"


class TestRealWorldTools:
    """Тесты с реальными примерами инструментов"""
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        self.registry = ToolRegistry()
        
        # Регистрируем набор полезных инструментов
        self._register_async_tools()
        self._register_data_processing_tools()
        self._register_file_simulation_tools()
    
    def _register_async_tools(self):
        """Регистрация асинхронных инструментов"""
        
        async def fetch_weather(city: str) -> dict:
            """Имитирует получение данных о погоде"""
            await asyncio.sleep(0.1)  # Имитация сетевого запроса
            return {
                "city": city,
                "temperature": 22,
                "condition": "солнечно",
                "humidity": 65
            }
        
        async def translate_text(text: str, target_language: str = "en") -> str:
            """Имитирует перевод текста"""
            await asyncio.sleep(0.05)
            translations = {
                "en": {"Привет": "Hello", "Мир": "World"},
                "fr": {"Привет": "Bonjour", "Мир": "Monde"}
            }
            return translations.get(target_language, {}).get(text, f"[{target_language}] {text}")
        
        async def calculate_hash(data: str, algorithm: str = "md5") -> str:
            """Имитирует вычисление хеша"""
            await asyncio.sleep(0.02)
            import hashlib
            if algorithm == "md5":
                return hashlib.md5(data.encode()).hexdigest()
            elif algorithm == "sha256":
                return hashlib.sha256(data.encode()).hexdigest()
            else:
                raise ValueError(f"Неподдерживаемый алгоритм: {algorithm}")
        
        self.registry.register_tool(
            "fetch_weather",
            fetch_weather,
            "Получает данные о погоде для указанного города"
        )
        
        self.registry.register_tool(
            "translate_text",
            translate_text,
            "Переводит текст на указанный язык"
        )
        
        self.registry.register_tool(
            "calculate_hash",
            calculate_hash,
            "Вычисляет хеш строки с использованием указанного алгоритма"
        )
    
    def _register_data_processing_tools(self):
        """Регистрация инструментов для обработки данных"""
        
        async def process_numbers(numbers: List[int], operation: str = "sum") -> float:
            """Обрабатывает список чисел"""
            await asyncio.sleep(0.01)
            
            if operation == "sum":
                return sum(numbers)
            elif operation == "average":
                return sum(numbers) / len(numbers) if numbers else 0
            elif operation == "max":
                return max(numbers) if numbers else 0
            elif operation == "min":
                return min(numbers) if numbers else 0
            else:
                raise ValueError(f"Неподдерживаемая операция: {operation}")
        
        async def filter_data(data: List[dict], field: str, value: Any) -> List[dict]:
            """Фильтрует список словарей по полю и значению"""
            await asyncio.sleep(0.01)
            return [item for item in data if item.get(field) == value]
        
        async def sort_data(data: List[dict], field: str, reverse: bool = False) -> List[dict]:
            """Сортирует список словарей по полю"""
            await asyncio.sleep(0.01)
            return sorted(data, key=lambda x: x.get(field, 0), reverse=reverse)
        
        self.registry.register_tool(
            "process_numbers",
            process_numbers,
            "Выполняет математические операции над списком чисел"
        )
        
        self.registry.register_tool(
            "filter_data",
            filter_data,
            "Фильтрует список объектов по указанному полю и значению"
        )
        
        self.registry.register_tool(
            "sort_data",
            sort_data,
            "Сортирует список объектов по указанному полю"
        )
    
    def _register_file_simulation_tools(self):
        """Регистрация инструментов для имитации работы с файлами"""
        
        # Имитируем файловую систему в памяти
        self._mock_filesystem = {}
        
        async def write_file(filename: str, content: str) -> str:
            """Имитирует запись в файл"""
            await asyncio.sleep(0.01)
            self._mock_filesystem[filename] = content
            return f"Файл '{filename}' записан, размер: {len(content)} символов"
        
        async def read_file(filename: str) -> str:
            """Имитирует чтение файла"""
            await asyncio.sleep(0.01)
            if filename not in self._mock_filesystem:
                raise FileNotFoundError(f"Файл '{filename}' не найден")
            return self._mock_filesystem[filename]
        
        async def list_files() -> List[str]:
            """Имитирует получение списка файлов"""
            await asyncio.sleep(0.01)
            return list(self._mock_filesystem.keys())
        
        self.registry.register_tool(
            "write_file",
            write_file,
            "Записывает содержимое в файл"
        )
        
        self.registry.register_tool(
            "read_file",
            read_file,
            "Читает содержимое файла"
        )
        
        self.registry.register_tool(
            "list_files",
            list_files,
            "Возвращает список всех файлов"
        )
    
    @pytest.mark.asyncio
    async def test_async_tools(self):
        """Тест асинхронных инструментов"""
        # Тест получения погоды
        result = await self.registry.execute_tool(
            "fetch_weather", 
            {"city": "Москва"}, 
            "weather_call"
        )
        assert result.success is True
        assert result.result["city"] == "Москва"
        assert "temperature" in result.result
        
        # Тест перевода
        result = await self.registry.execute_tool(
            "translate_text",
            {"text": "Привет", "target_language": "en"},
            "translate_call"
        )
        assert result.success is True
        assert result.result == "Hello"
        
        # Тест вычисления хеша
        result = await self.registry.execute_tool(
            "calculate_hash",
            {"data": "test", "algorithm": "md5"},
            "hash_call"
        )
        assert result.success is True
        assert len(result.result) == 32  # MD5 хеш
    
    @pytest.mark.asyncio
    async def test_data_processing_tools(self):
        """Тест инструментов обработки данных"""
        # Тест обработки чисел
        result = await self.registry.execute_tool(
            "process_numbers",
            {"numbers": [1, 2, 3, 4, 5], "operation": "sum"},
            "sum_call"
        )
        assert result.success is True
        assert result.result == 15
        
        result = await self.registry.execute_tool(
            "process_numbers",
            {"numbers": [1, 2, 3, 4, 5], "operation": "average"},
            "avg_call"
        )
        assert result.success is True
        assert result.result == 3.0
        
        # Тест фильтрации данных
        test_data = [
            {"name": "Alice", "age": 25},
            {"name": "Bob", "age": 30},
            {"name": "Charlie", "age": 25}
        ]
        
        result = await self.registry.execute_tool(
            "filter_data",
            {"data": test_data, "field": "age", "value": 25},
            "filter_call"
        )
        assert result.success is True
        assert len(result.result) == 2
        assert all(item["age"] == 25 for item in result.result)
    
    @pytest.mark.asyncio
    async def test_file_simulation_tools(self):
        """Тест инструментов имитации файловой системы"""
        # Тест записи файла
        result = await self.registry.execute_tool(
            "write_file",
            {"filename": "test.txt", "content": "Тестовое содержимое"},
            "write_call"
        )
        assert result.success is True
        assert "записан" in result.result
        
        # Тест чтения файла
        result = await self.registry.execute_tool(
            "read_file",
            {"filename": "test.txt"},
            "read_call"
        )
        assert result.success is True
        assert result.result == "Тестовое содержимое"
        
        # Тест списка файлов
        result = await self.registry.execute_tool(
            "list_files",
            {},
            "list_call"
        )
        assert result.success is True
        assert "test.txt" in result.result
    
    @pytest.mark.asyncio
    async def test_parallel_tool_execution(self):
        """Тест параллельного выполнения инструментов"""
        from kraken_llm.tools.tools import ToolCall
        
        # Создаем несколько вызовов инструментов
        tool_calls = [
            ToolCall(
                id="call_1",
                type="function",
                function={"name": "fetch_weather", "arguments": {"city": "Москва"}}
            ),
            ToolCall(
                id="call_2",
                type="function", 
                function={"name": "calculate_hash", "arguments": {"data": "test1"}}
            ),
            ToolCall(
                id="call_3",
                type="function",
                function={"name": "process_numbers", "arguments": {"numbers": [1, 2, 3]}}
            )
        ]
        
        # Выполняем параллельно
        import time
        start_time = time.time()
        results = await self.registry.execute_parallel_tools(tool_calls)
        execution_time = time.time() - start_time
        
        # Проверяем результаты
        assert len(results) == 3
        assert all(result.success for result in results)
        
        # Параллельное выполнение должно быть быстрее последовательного
        # (учитывая, что каждый инструмент имеет задержку)
        assert execution_time < 0.2  # Должно быть значительно меньше суммы задержек
        
        # Проверяем результаты по ID
        results_by_id = {result.tool_call_id: result for result in results}
        
        assert "city" in results_by_id["call_1"].result
        assert len(results_by_id["call_2"].result) == 32  # MD5 хеш
        assert results_by_id["call_3"].result == 6  # Сумма [1, 2, 3]


class TestIntegratedFunctionToolExecution:
    """Тесты интегрированного выполнения функций и инструментов"""
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        self.function_registry = FunctionRegistry()
        self.tool_registry = ToolRegistry()
        self.executor = FunctionToolExecutor(
            function_registry=self.function_registry,
            tool_registry=self.tool_registry
        )
        
        # Регистрируем смешанный набор функций и инструментов
        self._register_mixed_capabilities()
    
    def _register_mixed_capabilities(self):
        """Регистрация смешанных возможностей"""
        
        # Синхронные функции
        def calculate_tax(amount: float, rate: float = 0.2) -> float:
            """Вычисляет налог"""
            return amount * rate
        
        def format_currency(amount: float, currency: str = "RUB") -> str:
            """Форматирует сумму в валюте"""
            return f"{amount:.2f} {currency}"
        
        # Асинхронные инструменты
        async def fetch_exchange_rate(from_currency: str, to_currency: str) -> float:
            """Имитирует получение курса валют"""
            await asyncio.sleep(0.05)
            rates = {
                ("USD", "RUB"): 75.5,
                ("EUR", "RUB"): 85.2,
                ("RUB", "USD"): 1/75.5,
                ("RUB", "EUR"): 1/85.2
            }
            return rates.get((from_currency, to_currency), 1.0)
        
        async def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
            """Конвертирует валюту"""
            await asyncio.sleep(0.03)
            # Используем fetch_exchange_rate
            rate = await fetch_exchange_rate(from_currency, to_currency)
            converted_amount = amount * rate
            return {
                "original_amount": amount,
                "original_currency": from_currency,
                "converted_amount": converted_amount,
                "target_currency": to_currency,
                "exchange_rate": rate
            }
        
        # Регистрируем функции
        self.function_registry.register_function(
            "calculate_tax", calculate_tax, "Вычисляет налог с суммы"
        )
        self.function_registry.register_function(
            "format_currency", format_currency, "Форматирует сумму в указанной валюте"
        )
        
        # Регистрируем инструменты
        self.tool_registry.register_tool(
            "fetch_exchange_rate", fetch_exchange_rate, "Получает курс обмена валют"
        )
        self.tool_registry.register_tool(
            "convert_currency", convert_currency, "Конвертирует сумму из одной валюты в другую"
        )
    
    @pytest.mark.asyncio
    async def test_mixed_execution_workflow(self):
        """Тест рабочего процесса с функциями и инструментами"""
        context = ExecutionContext(
            request_id="workflow_test",
            user_id="test_user",
            metadata={"workflow": "currency_calculation"}
        )
        
        # Шаг 1: Вычисляем налог (функция)
        function_calls = [
            {"name": "calculate_tax", "arguments": {"amount": 1000.0, "rate": 0.13}}
        ]
        
        # Шаг 2: Конвертируем валюту (инструмент)
        tool_calls = [
            {
                "id": "convert_call",
                "type": "function",
                "function": {
                    "name": "convert_currency",
                    "arguments": {
                        "amount": 130.0,  # Результат налога
                        "from_currency": "RUB",
                        "to_currency": "USD"
                    }
                }
            }
        ]
        
        # Выполняем смешанные вызовы
        result = await self.executor.execute_mixed_calls(
            function_calls=function_calls,
            tool_calls=tool_calls,
            context=context
        )
        
        assert result.success is True
        assert len(result.results) == 2
        assert result.context.request_id == "workflow_test"
        
        # Проверяем результаты
        function_result = next(r for r in result.results if hasattr(r, 'name') and not hasattr(r, 'tool_call_id'))
        tool_result = next(r for r in result.results if hasattr(r, 'tool_call_id'))
        
        assert function_result.result == 130.0  # 1000 * 0.13
        assert tool_result.result["original_amount"] == 130.0
        assert tool_result.result["target_currency"] == "USD"
    
    @pytest.mark.asyncio
    async def test_complex_parallel_workflow(self):
        """Тест сложного параллельного рабочего процесса"""
        # Множественные функции и инструменты
        function_calls = [
            {"name": "calculate_tax", "arguments": {"amount": 1000.0}},
            {"name": "calculate_tax", "arguments": {"amount": 2000.0}},
            {"name": "format_currency", "arguments": {"amount": 1500.0, "currency": "USD"}}
        ]
        
        tool_calls = [
            {
                "id": "rate_1",
                "type": "function",
                "function": {
                    "name": "fetch_exchange_rate",
                    "arguments": {"from_currency": "USD", "to_currency": "RUB"}
                }
            },
            {
                "id": "rate_2", 
                "type": "function",
                "function": {
                    "name": "fetch_exchange_rate",
                    "arguments": {"from_currency": "EUR", "to_currency": "RUB"}
                }
            },
            {
                "id": "convert_1",
                "type": "function",
                "function": {
                    "name": "convert_currency",
                    "arguments": {
                        "amount": 100.0,
                        "from_currency": "USD",
                        "to_currency": "RUB"
                    }
                }
            }
        ]
        
        result = await self.executor.execute_mixed_calls(
            function_calls=function_calls,
            tool_calls=tool_calls
        )
        
        assert result.success is True
        assert len(result.results) == 6  # 3 функции + 3 инструмента
        
        # Все результаты должны быть успешными
        assert all(r.success for r in result.results)
    
    def test_capabilities_overview(self):
        """Тест получения обзора всех возможностей"""
        capabilities = self.executor.get_all_capabilities()
        
        assert capabilities["function_count"] == 2
        assert capabilities["tool_count"] == 2
        
        # Проверяем наличие всех зарегистрированных возможностей
        function_names = [f["name"] for f in capabilities["functions"]]
        tool_names = [t["function"]["name"] for t in capabilities["tools"]]
        
        assert "calculate_tax" in function_names
        assert "format_currency" in function_names
        assert "fetch_exchange_rate" in tool_names
        assert "convert_currency" in tool_names
    
    @pytest.mark.asyncio
    async def test_validation_comprehensive(self):
        """Тест комплексной валидации"""
        # Корректные вызовы
        valid_function_calls = [
            {"name": "calculate_tax", "arguments": {"amount": 100}}
        ]
        
        valid_tool_calls = [
            {
                "id": "test_call",
                "type": "function",
                "function": {"name": "fetch_exchange_rate", "arguments": {"from_currency": "USD", "to_currency": "RUB"}}
            }
        ]
        
        validation = await self.executor.validate_calls(valid_function_calls, valid_tool_calls)
        assert validation["valid"] is True
        assert len(validation["errors"]) == 0
        
        # Некорректные вызовы
        invalid_function_calls = [
            {"name": "nonexistent_function", "arguments": {}}
        ]
        
        invalid_tool_calls = [
            {
                "id": "test_call",
                "type": "function", 
                "function": {"name": "nonexistent_tool", "arguments": {}}
            }
        ]
        
        validation = await self.executor.validate_calls(invalid_function_calls, invalid_tool_calls)
        assert validation["valid"] is False
        assert len(validation["errors"]) >= 2  # По ошибке для каждого несуществующего вызова


class TestDecoratorIntegration:
    """Тесты интеграции с декораторами"""
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        # Очищаем глобальные реестры
        from kraken_llm.tools.functions import default_function_registry
        from kraken_llm.tools.tools import default_tool_registry
        
        default_function_registry.clear()
        default_tool_registry.clear()
    
    def test_decorator_registration(self):
        """Тест регистрации через декораторы"""
        
        @register_function(description="Складывает два числа")
        def add_numbers(a: int, b: int) -> int:
            return a + b
        
        @register_tool(description="Умножает число на 2")
        async def double_number(x: int) -> int:
            return x * 2
        
        from kraken_llm.tools.functions import default_function_registry
        from kraken_llm.tools.tools import default_tool_registry
        
        assert "add_numbers" in default_function_registry
        assert "double_number" in default_tool_registry
        
        # Проверяем, что функции работают
        func_result = default_function_registry.execute_function("add_numbers", {"a": 3, "b": 4})
        assert func_result.success is True
        assert func_result.result == 7
    
    @pytest.mark.asyncio
    async def test_decorator_with_executor(self):
        """Тест использования декораторов с исполнителем"""
        
        @register_function(description="Вычисляет квадрат числа")
        def square(x: int) -> int:
            return x ** 2
        
        @register_tool(description="Вычисляет куб числа")
        async def cube(x: int) -> int:
            await asyncio.sleep(0.01)
            return x ** 3
        
        # Используем глобальный исполнитель
        from kraken_llm.tools.executor import default_executor
        
        function_calls = [{"name": "square", "arguments": {"x": 5}}]
        tool_calls = [{
            "id": "cube_call",
            "type": "function",
            "function": {"name": "cube", "arguments": {"x": 3}}
        }]
        
        result = await default_executor.execute_mixed_calls(
            function_calls=function_calls,
            tool_calls=tool_calls
        )
        
        assert result.success is True
        assert len(result.results) == 2
        
        # Проверяем результаты
        function_result = next(r for r in result.results if hasattr(r, 'name') and not hasattr(r, 'tool_call_id'))
        tool_result = next(r for r in result.results if hasattr(r, 'tool_call_id'))
        
        assert function_result.result == 25  # 5^2
        assert tool_result.result == 27  # 3^3