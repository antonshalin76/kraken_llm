#!/usr/bin/env python3
"""
Пример использования системы function и tool calling в Kraken LLM фреймворке.

Этот пример демонстрирует:
1. Регистрацию функций и инструментов
2. Выполнение function calls
3. Выполнение tool calls с параллельностью
4. Использование декораторов
5. Смешанное выполнение функций и инструментов
"""

import asyncio
import json
from kraken_llm.tools import (
    FunctionRegistry,
    ToolRegistry,
    FunctionToolExecutor,
    ExecutionContext,
    register_function,
    register_tool
)


def demo_function_registry():
    """Демонстрация работы с реестром функций"""
    print("\n=== Демонстрация FunctionRegistry ===\n")

    # Создаем реестр функций
    registry = FunctionRegistry()

    # Регистрируем математические функции
    def add(a: int, b: int) -> int:
        """Складывает два числа"""
        return a + b

    def multiply(x: float, y: float) -> float:
        """Умножает два числа"""
        return x * y

    def power(base: float, exponent: int = 2) -> float:
        """Возводит число в степень"""
        return base ** exponent

    registry.register_function("add", add, "Складывает два числа")
    registry.register_function("multiply", multiply, "Умножает два числа")
    registry.register_function("power", power, "Возводит число в степень")

    print(f"Зарегистрировано функций: {len(registry)}")
    print(f"Список функций: {registry.list_functions()}")

    # Получаем схемы для OpenAI API
    schemas = registry.get_all_schemas()
    print(f"\nСхемы функций для OpenAI API:")
    for schema in schemas:
        print(f"- {schema['name']}: {schema['description']}")

    # Выполняем функции
    print(f"\nВыполнение функций:")

    result = registry.execute_function("add", {"a": 5, "b": 3})
    print(f"add(5, 3) = {result.result} (успех: {result.success})")

    result = registry.execute_function("multiply", {"x": 4.5, "y": 2.0})
    print(f"multiply(4.5, 2.0) = {result.result} (успех: {result.success})")

    result = registry.execute_function("power", {"base": 2.0, "exponent": 3})
    print(f"power(2.0, 3) = {result.result} (успех: {result.success})")


async def demo_tool_registry():
    """Демонстрация работы с реестром инструментов"""
    print("\n=== Демонстрация ToolRegistry ===\n")

    # Создаем реестр инструментов
    registry = ToolRegistry()

    # Регистрируем асинхронные инструменты
    async def fetch_data(url: str) -> dict:
        """Имитирует получение данных по URL"""
        await asyncio.sleep(0.1)  # Имитация сетевого запроса
        return {
            "url": url,
            "status": 200,
            "data": f"Данные с {url}",
            "timestamp": "2024-01-01T12:00:00Z"
        }

    async def process_text(text: str, operation: str = "upper") -> str:
        """Обрабатывает текст"""
        await asyncio.sleep(0.05)
        if operation == "upper":
            return text.upper()
        elif operation == "lower":
            return text.lower()
        elif operation == "reverse":
            return text[::-1]
        else:
            return text

    def calculate_hash(data: str) -> str:
        """Вычисляет хеш строки (синхронная функция в реестре инструментов)"""
        import hashlib
        return hashlib.md5(data.encode()).hexdigest()

    registry.register_tool("fetch_data", fetch_data, "Получает данные по URL")
    registry.register_tool("process_text", process_text, "Обрабатывает текст")
    registry.register_tool(
        "calculate_hash", calculate_hash, "Вычисляет MD5 хеш")

    print(f"Зарегистрировано инструментов: {len(registry)}")
    print(f"Список инструментов: {registry.list_tools()}")

    # Получаем схемы для OpenAI API
    schemas = registry.get_all_schemas()
    print(f"\nСхемы инструментов для OpenAI API:")
    for schema in schemas:
        func = schema['function']
        print(f"- {func['name']}: {func['description']}")

    # Выполняем инструменты последовательно
    print(f"\nПоследовательное выполнение инструментов:")

    result = await registry.execute_tool("fetch_data", {"url": "https://google.com"}, "call_1")
    print(
        f"fetch_data результат: {result.result['status']} - {result.result['data']}")

    result = await registry.execute_tool("process_text", {"text": "Привет Мир", "operation": "upper"}, "call_2")
    print(f"process_text результат: {result.result}")

    result = await registry.execute_tool("calculate_hash", {"data": "test"}, "call_3")
    print(f"calculate_hash результат: {result.result}")

    # Демонстрация параллельного выполнения
    print(f"\nПараллельное выполнение инструментов:")

    from kraken_llm.tools.tools import ToolCall

    tool_calls = [
        ToolCall(
            id="parallel_1",
            type="function",
            function={"name": "fetch_data", "arguments": {
                "url": "https://api1.example.com"}}
        ),
        ToolCall(
            id="parallel_2",
            type="function",
            function={"name": "fetch_data", "arguments": {
                "url": "https://api2.example.com"}}
        ),
        ToolCall(
            id="parallel_3",
            type="function",
            function={"name": "process_text", "arguments": {
                "text": "Параллельная обработка", "operation": "reverse"}}
        )
    ]

    import time
    start_time = time.time()
    results = await registry.execute_parallel_tools(tool_calls)
    execution_time = time.time() - start_time

    print(
        f"Выполнено {len(results)} инструментов за {execution_time:.3f} секунд")
    for result in results:
        print(
            f"- {result.tool_call_id}: {result.success} - {str(result.result)[:50]}...")


async def demo_executor():
    """Демонстрация работы с исполнителем"""
    print("\n=== Демонстрация FunctionToolExecutor ===\n")
    print("ПРИМЕЧАНИЕ: Эта демонстрация создает НОВЫЕ реестры с ДРУГИМИ функциями и инструментами")
    print("для демонстрации независимости компонентов.\n")

    # Создаем реестры и исполнитель
    func_registry = FunctionRegistry()
    tool_registry = ToolRegistry()

    # Регистрируем функции
    def calculate_tax(amount: float, rate: float = 0.2) -> float:
        """Вычисляет налог"""
        return amount * rate

    def format_currency(amount: float, currency: str = "RUB") -> str:
        """Форматирует сумму в валюте"""
        return f"{amount:.2f} {currency}"

    func_registry.register_function("calculate_tax", calculate_tax, "Вычисляет налог с суммы")
    func_registry.register_function("format_currency", format_currency, "Форматирует сумму в валюте")

    # Регистрируем инструменты
    async def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
        """Конвертирует валюту"""
        await asyncio.sleep(0.1)
        # Простые курсы для демонстрации
        rates = {
            ("USD", "RUB"): 75.0,
            ("EUR", "RUB"): 85.0,
            ("RUB", "USD"): 1/75.0,
            ("RUB", "EUR"): 1/85.0
        }
        rate = rates.get((from_currency, to_currency), 1.0)
        return {
            "original_amount": amount,
            "converted_amount": amount * rate,
            "rate": rate,
            "from_currency": from_currency,
            "to_currency": to_currency
        }

    async def send_notification(message: str, recipient: str = "admin") -> dict:
        """Отправляет уведомление"""
        await asyncio.sleep(0.05)
        return {
            "message": message,
            "recipient": recipient,
            "sent_at": "2024-01-01T12:00:00Z",
            "status": "delivered"
        }

    tool_registry.register_tool(
        "convert_currency", convert_currency, "Конвертирует валюту")
    tool_registry.register_tool(
        "send_notification", send_notification, "Отправляет уведомление")

    # Создаем исполнитель ПОСЛЕ регистрации функций и инструментов
    executor = FunctionToolExecutor(func_registry, tool_registry)

    # Проверяем регистрацию
    print(f"Функций в НОВОМ реестре: {len(func_registry)}")
    print(f"Инструментов в НОВОМ реестре: {len(tool_registry)}")

    # Демонстрация возможностей
    capabilities = executor.get_all_capabilities()
    print(f"Доступно функций: {capabilities['function_count']}")
    print(f"Доступно инструментов: {capabilities['tool_count']}")

    # Создаем контекст выполнения
    context = ExecutionContext(
        request_id="demo_request_001",
        user_id="demo_user",
        metadata={"demo": True, "version": "1.0"}
    )

    # Проверяем доступные функции и инструменты
    print(f"\nДоступные функции: {func_registry.list_functions()}")
    print(f"Доступные инструменты: {tool_registry.list_tools()}")

    # Выполняем смешанный рабочий процесс
    print(f"\nВыполнение смешанного рабочего процесса:")

    function_calls = [
        {"name": "calculate_tax", "arguments": {"amount": 1000.0, "rate": 0.13}},
        {"name": "format_currency", "arguments": {
            "amount": 130.0, "currency": "RUB"}}
    ]

    tool_calls = [
        {
            "id": "convert_1",
            "type": "function",
            "function": {
                "name": "convert_currency",
                "arguments": {
                    "amount": 130.0,
                    "from_currency": "RUB",
                    "to_currency": "USD"
                }
            }
        },
        {
            "id": "notify_1",
            "type": "function",
            "function": {
                "name": "send_notification",
                "arguments": {
                    "message": "Налог рассчитан и конвертирован",
                    "recipient": "finance_team"
                }
            }
        }
    ]

    result = await executor.execute_mixed_calls(
        function_calls=function_calls,
        tool_calls=tool_calls,
        context=context
    )

    print(f"Общий статус: {result.success}")
    print(f"Время выполнения: {result.execution_time:.3f} секунд")
    print(f"Количество результатов: {len(result.results)}")

    print(f"\nРезультаты:")
    for i, res in enumerate(result.results):
        if hasattr(res, 'tool_call_id'):
            print(
                f"{i+1}. Инструмент {res.name} (ID: {res.tool_call_id}): {res.success}")
            if res.success:
                print(f"   Результат: {res.result}")
        else:
            print(f"{i+1}. Функция {res.name}: {res.success}")
            if res.success:
                print(f"   Результат: {res.result}")


async def demo_unified_executor():
    """Демонстрация работы с исполнителем, используя ВСЕ функции и инструменты"""
    print("\n=== Демонстрация Unified FunctionToolExecutor ===\n")
    print("Эта демонстрация объединяет ВСЕ функции и инструменты из предыдущих примеров\n")

    # Создаем реестры и регистрируем ВСЕ функции и инструменты
    func_registry = FunctionRegistry()
    tool_registry = ToolRegistry()

    # Функции из demo_function_registry
    def add(a: int, b: int) -> int:
        """Складывает два числа"""
        return a + b
    
    def multiply(x: float, y: float) -> float:
        """Умножает два числа"""
        return x * y
    
    def power(base: float, exponent: int = 2) -> float:
        """Возводит число в степень"""
        return base ** exponent

    # Функции из demo_executor
    def calculate_tax(amount: float, rate: float = 0.2) -> float:
        """Вычисляет налог"""
        return amount * rate

    def format_currency(amount: float, currency: str = "RUB") -> str:
        """Форматирует сумму в валюте"""
        return f"{amount:.2f} {currency}"

    # Регистрируем все функции
    func_registry.register_function("add", add, "Складывает два числа")
    func_registry.register_function("multiply", multiply, "Умножает два числа")
    func_registry.register_function("power", power, "Возводит число в степень")
    func_registry.register_function("calculate_tax", calculate_tax, "Вычисляет налог с суммы")
    func_registry.register_function("format_currency", format_currency, "Форматирует сумму в валюте")

    # Инструменты из demo_tool_registry
    async def fetch_data(url: str) -> dict:
        """Имитирует получение данных по URL"""
        await asyncio.sleep(0.1)
        return {
            "url": url,
            "status": 200,
            "data": f"Данные с {url}",
            "timestamp": "2024-01-01T12:00:00Z"
        }
    
    async def process_text(text: str, operation: str = "upper") -> str:
        """Обрабатывает текст"""
        await asyncio.sleep(0.05)
        if operation == "upper":
            return text.upper()
        elif operation == "lower":
            return text.lower()
        elif operation == "reverse":
            return text[::-1]
        else:
            return text
    
    def calculate_hash(data: str) -> str:
        """Вычисляет хеш строки"""
        import hashlib
        return hashlib.md5(data.encode()).hexdigest()

    # Инструменты из demo_executor
    async def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
        """Конвертирует валюту"""
        await asyncio.sleep(0.1)
        rates = {
            ("USD", "RUB"): 75.0,
            ("EUR", "RUB"): 85.0,
            ("RUB", "USD"): 1/75.0,
            ("RUB", "EUR"): 1/85.0
        }
        rate = rates.get((from_currency, to_currency), 1.0)
        return {
            "original_amount": amount,
            "converted_amount": amount * rate,
            "rate": rate,
            "from_currency": from_currency,
            "to_currency": to_currency
        }

    async def send_notification(message: str, recipient: str = "admin") -> dict:
        """Отправляет уведомление"""
        await asyncio.sleep(0.05)
        return {
            "message": message,
            "recipient": recipient,
            "sent_at": "2024-01-01T12:00:00Z",
            "status": "delivered"
        }

    # Регистрируем все инструменты
    tool_registry.register_tool("fetch_data", fetch_data, "Получает данные по URL")
    tool_registry.register_tool("process_text", process_text, "Обрабатывает текст")
    tool_registry.register_tool("calculate_hash", calculate_hash, "Вычисляет MD5 хеш")
    tool_registry.register_tool("convert_currency", convert_currency, "Конвертирует валюту")
    tool_registry.register_tool("send_notification", send_notification, "Отправляет уведомление")

    # Создаем исполнитель с ВСЕМИ функциями и инструментами
    executor = FunctionToolExecutor(func_registry, tool_registry)

    # Показываем общую статистику
    capabilities = executor.get_all_capabilities()
    print(f"ВСЕГО доступно функций: {capabilities['function_count']}")
    print(f"ВСЕГО доступно инструментов: {capabilities['tool_count']}")
    print(f"Функции: {func_registry.list_functions()}")
    print(f"Инструменты: {tool_registry.list_tools()}")

    print(f"\nТеперь у нас есть доступ ко ВСЕМ функциям и инструментам из предыдущих демонстраций!")


def demo_decorators():
    """Демонстрация использования декораторов"""
    print("\n=== Демонстрация декораторов ===\n")

    # Очищаем глобальные реестры для чистоты демонстрации
    from kraken_llm.tools.functions import default_function_registry
    from kraken_llm.tools.tools import default_tool_registry

    default_function_registry.clear()
    default_tool_registry.clear()

    # Регистрируем функции через декораторы
    @register_function(description="Вычисляет площадь прямоугольника")
    def rectangle_area(width: float, height: float) -> float:
        return width * height

    @register_function(description="Проверяет, является ли число четным")
    def is_even(number: int) -> bool:
        return number % 2 == 0

    # Регистрируем инструменты через декораторы
    @register_tool(description="Генерирует случайное число")
    async def generate_random(min_val: int = 1, max_val: int = 100) -> int:
        await asyncio.sleep(0.01)
        import random
        return random.randint(min_val, max_val)

    @register_tool(description="Создает отчет")
    async def create_report(title: str, data: dict) -> dict:
        await asyncio.sleep(0.05)
        return {
            "title": title,
            "data": data,
            "created_at": "2024-01-01T12:00:00Z",
            "pages": len(str(data)) // 100 + 1
        }

    print(f"Зарегистрировано функций: {len(default_function_registry)}")
    print(f"Зарегистрировано инструментов: {len(default_tool_registry)}")

    # Используем глобальный исполнитель
    from kraken_llm.tools.executor import default_executor

    capabilities = default_executor.get_all_capabilities()
    print(f"\nВозможности глобального исполнителя:")
    print(f"- Функции: {[f['name'] for f in capabilities['functions']]}")
    print(
        f"- Инструменты: {[t['function']['name'] for t in capabilities['tools']]}")


async def main():
    """Главная функция демонстрации"""

    # Демонстрация реестра функций
    demo_function_registry()

    # Демонстрация реестра инструментов
    await demo_tool_registry()

    # Демонстрация исполнителя (с новыми функциями и инструментами)
    await demo_executor()

    # Демонстрация объединенного исполнителя (со ВСЕМИ функциями и инструментами)
    await demo_unified_executor()

    # Демонстрация декораторов
    demo_decorators()

    print("\n✅ Демонстрация завершена!")
    print("\nОсновные возможности:")
    print("- ✅ Регистрация функций с автоматической генерацией схем")
    print("- ✅ Регистрация инструментов с поддержкой async/await")
    print("- ✅ Параллельное выполнение инструментов")
    print("- ✅ Смешанное выполнение функций и инструментов")
    print("- ✅ Декораторы для удобной регистрации")
    print("- ✅ Валидация вызовов и обработка ошибок")
    print("- ✅ Контекст выполнения и метрики")


if __name__ == "__main__":
    asyncio.run(main())
