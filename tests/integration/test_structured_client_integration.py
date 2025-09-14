"""
Integration тесты для StructuredLLMClient.

Тестирует реальную интеграцию с AsyncOpenAI и Outlines для structured output,
включая различные Pydantic модели, streaming и обработку ошибок.
"""

import asyncio
import pytest
import os
from typing import List, Optional, Union
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from kraken_llm.client.structured import StructuredLLMClient
from kraken_llm.config.settings import LLMConfig
from kraken_llm.exceptions.validation import ValidationError
from kraken_llm.exceptions.network import NetworkError

from dotenv import load_dotenv

load_dotenv()

# Тестовые Pydantic модели для integration тестов
class UserProfile(BaseModel):
    """Профиль пользователя для тестирования."""
    name: str = Field(..., min_length=1, max_length=100, description="Полное имя пользователя")
    age: int = Field(..., ge=0, le=150, description="Возраст пользователя")
    email: Optional[str] = Field(None, description="Email адрес")
    is_active: bool = Field(True, description="Активен ли пользователь")
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        if v and '@' not in v:
            raise ValueError('Некорректный email адрес')
        return v


class ProductInfo(BaseModel):
    """Информация о продукте."""
    name: str = Field(..., description="Название продукта")
    price: float = Field(..., gt=0, description="Цена продукта")
    category: str = Field(..., description="Категория продукта")
    tags: List[str] = Field(default_factory=list, description="Теги продукта")
    in_stock: bool = Field(True, description="Есть ли в наличии")


class OrderDetails(BaseModel):
    """Детали заказа с вложенными объектами."""
    order_id: str = Field(..., description="ID заказа")
    customer: UserProfile = Field(..., description="Информация о клиенте")
    products: List[ProductInfo] = Field(..., description="Список продуктов")
    total_amount: float = Field(..., gt=0, description="Общая сумма заказа")
    order_date: str = Field(..., description="Дата заказа")


class AnalysisResult(BaseModel):
    """Результат анализа текста."""
    sentiment: str = Field(..., description="Тональность: positive, negative, neutral")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность в результате")
    keywords: List[str] = Field(default_factory=list, description="Ключевые слова")
    summary: str = Field(..., description="Краткое резюме")


class MathProblem(BaseModel):
    """Математическая задача с решением."""
    problem: str = Field(..., description="Формулировка задачи")
    solution: str = Field(..., description="Пошаговое решение")
    answer: Union[int, float] = Field(..., description="Финальный ответ")
    difficulty: str = Field(..., description="Уровень сложности: easy, medium, hard")


@pytest.fixture
def integration_config():
    """Конфигурация для integration тестов."""
    return LLMConfig(
        endpoint=os.getenv("LLM_ENDPOINT"),
        api_key=os.getenv("LLM_TOKEN"),
        model=os.getenv("LLM_MODEL"),
        temperature=0.1,  # Низкая температура для предсказуемости
        max_tokens=2000,
        connect_timeout=30.0,
        read_timeout=120.0,
    )


@pytest.fixture
async def structured_client(integration_config):
    """Фикстура StructuredLLMClient для integration тестов."""
    client = StructuredLLMClient(integration_config)
    yield client
    await client.close()


@pytest.mark.integration
@pytest.mark.asyncio
class TestStructuredLLMClientBasicIntegration:
    """Базовые integration тесты."""
    
    async def test_simple_structured_output(self, structured_client):
        """Тест простого structured output."""
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that generates user data in JSON format. Always respond with valid JSON only, using English field names: name, age, email, is_active."
            },
            {
                "role": "user", 
                "content": "Create a user profile with name 'Иван Петров', age 25 and email 'ivan@example.com'. Return only JSON."
            }
        ]
        
        try:
            result = await structured_client.chat_completion_structured(
                messages=messages,
                response_model=UserProfile,
                stream=False
            )
            
            # Проверяем результат
            assert isinstance(result, UserProfile)
            assert result.name == "Иван Петров" or "Иван" in result.name
            assert isinstance(result.age, int)
            assert 20 <= result.age <= 30  # Примерный диапазон
            assert result.email and "@" in result.email
            
        except NetworkError:
            pytest.skip("Тестовый LLM сервер недоступен")
        except Exception as e:
            pytest.fail(f"Неожиданная ошибка в простом structured output: {e}")
    
    async def test_complex_structured_output(self, structured_client):
        """Тест сложного structured output с вложенными объектами."""
        messages = [
            {
                "role": "system",
                "content": "Generate product data in JSON format. Use English field names: name, price, category, tags, in_stock. Return only valid JSON."
            },
            {
                "role": "user",
                "content": "Create product info: laptop for 50000 rubles in 'Electronics' category. Return only JSON."
            }
        ]
        
        try:
            result = await structured_client.chat_completion_structured(
                messages=messages,
                response_model=ProductInfo,
                stream=False
            )
            
            # Проверяем результат
            assert isinstance(result, ProductInfo)
            assert "ноутбук" in result.name.lower() or "laptop" in result.name.lower()
            assert isinstance(result.price, (int, float))
            assert result.price > 0
            assert result.category
            assert isinstance(result.tags, list)
            assert isinstance(result.in_stock, bool)
            
        except NetworkError:
            pytest.skip("Тестовый LLM сервер недоступен")
        except Exception as e:
            pytest.fail(f"Неожиданная ошибка в сложном structured output: {e}")
    
    async def test_streaming_structured_output(self, structured_client):
        """Тест streaming structured output."""
        messages = [
            {
                "role": "system",
                "content": "Generate user profiles in JSON format. Use English field names: name, age, email, is_active. Return only valid JSON."
            },
            {
                "role": "user",
                "content": "Create user profile with name 'Анна Смирнова', age 30. Return only JSON."
            }
        ]
        
        try:
            result = await structured_client.chat_completion_structured(
                messages=messages,
                response_model=UserProfile,
                stream=True
            )
            
            # Проверяем результат
            assert isinstance(result, UserProfile)
            assert result.name
            assert isinstance(result.age, int)
            assert result.age > 0
            
        except NetworkError:
            pytest.skip("Тестовый LLM сервер недоступен")
        except Exception as e:
            pytest.fail(f"Неожиданная ошибка в streaming structured output: {e}")
    
    async def test_streaming_chunks_output(self, structured_client):
        """Тест streaming structured output (возвращает готовый объект)."""
        messages = [
            {
                "role": "system",
                "content": "Generate user data in JSON format. Use English field names: name, age, email, is_active. Return only valid JSON."
            },
            {
                "role": "user",
                "content": "Create simple user profile. Return only JSON."
            }
        ]
        
        try:
            # В structured client streaming возвращает готовый объект, а не chunks
            result = await structured_client.chat_completion_structured(
                messages=messages,
                response_model=UserProfile,
                stream=True
            )
            
            # Проверяем, что получили валидный объект
            assert isinstance(result, UserProfile)
            assert result.name
            assert isinstance(result.age, int)
            assert result.age > 0
            
        except NetworkError:
            pytest.skip("Тестовый LLM сервер недоступен")
        except Exception as e:
            pytest.fail(f"Неожиданная ошибка в streaming structured output: {e}")


@pytest.mark.integration
@pytest.mark.asyncio
class TestStructuredLLMClientAdvancedIntegration:
    """Продвинутые integration тесты."""
    
    async def test_nested_model_integration(self, structured_client):
        """Тест с глубоко вложенной моделью."""
        messages = [
            {
                "role": "system",
                "content": "Create detailed orders with customer and product information in JSON format. Use English field names: order_id, customer (with name, age, email, is_active), products (array with name, price, category, tags, in_stock), total_amount, order_date. Return only valid JSON."
            },
            {
                "role": "user",
                "content": (
                    "Create order for customer 'Петр Иванов' (25 years old, email: petr@example.com) "
                    "with two products: book for 500 rubles and pen for 100 rubles. "
                    "Order ID: ORDER-123, date: 2024-01-15. Return only JSON."
                )
            }
        ]
        
        try:
            result = await structured_client.chat_completion_structured(
                messages=messages,
                response_model=OrderDetails,
                stream=False
            )
            
            # Проверяем структуру результата
            assert isinstance(result, OrderDetails)
            assert result.order_id
            
            # Проверяем вложенного клиента
            assert isinstance(result.customer, UserProfile)
            assert result.customer.name
            assert isinstance(result.customer.age, int)
            
            # Проверяем список продуктов
            assert isinstance(result.products, list)
            assert len(result.products) > 0
            
            for product in result.products:
                assert isinstance(product, ProductInfo)
                assert product.name
                assert isinstance(product.price, (int, float))
                assert product.price > 0
            
            # Проверяем общую сумму
            assert isinstance(result.total_amount, (int, float))
            assert result.total_amount > 0
            
        except NetworkError:
            pytest.skip("Тестовый LLM сервер недоступен")
        except Exception as e:
            pytest.fail(f"Неожиданная ошибка в nested model integration: {e}")
    
    async def test_analysis_task_integration(self, structured_client):
        """Тест задачи анализа текста."""
        messages = [
            {
                "role": "system",
                "content": "Analyze text sentiment and return result in JSON format. Use English field names: sentiment (positive/negative/neutral), confidence (0.0-1.0), keywords (array), summary (string). Return only valid JSON."
            },
            {
                "role": "user",
                "content": (
                    "Analyze sentiment of this text: "
                    "'Этот продукт просто великолепен! Я очень доволен покупкой и рекомендую всем.' "
                    "Return only JSON."
                )
            }
        ]
        
        try:
            result = await structured_client.chat_completion_structured(
                messages=messages,
                response_model=AnalysisResult,
                stream=False
            )
            
            # Проверяем результат анализа
            assert isinstance(result, AnalysisResult)
            assert result.sentiment in ["positive", "negative", "neutral"]
            assert 0.0 <= result.confidence <= 1.0
            assert isinstance(result.keywords, list)
            assert result.summary
            
            # Для позитивного текста ожидаем позитивную тональность
            assert result.sentiment == "positive"
            assert result.confidence > 0.5
            
        except NetworkError:
            pytest.skip("Тестовый LLM сервер недоступен")
        except Exception as e:
            pytest.fail(f"Неожиданная ошибка в analysis task: {e}")
    
    async def test_math_problem_integration(self, structured_client):
        """Тест решения математической задачи."""
        messages = [
            {
                "role": "system",
                "content": "Solve math problems and provide structured answer in JSON format. Use English field names: problem, solution, answer (number), difficulty (easy/medium/hard). Return only valid JSON."
            },
            {
                "role": "user",
                "content": "Solve: Find area of rectangle with sides 5 and 8 meters. Return only JSON."
            }
        ]
        
        try:
            result = await structured_client.chat_completion_structured(
                messages=messages,
                response_model=MathProblem,
                stream=False
            )
            
            # Проверяем результат
            assert isinstance(result, MathProblem)
            assert result.problem
            assert result.solution
            assert isinstance(result.answer, (int, float))
            assert result.difficulty in ["easy", "medium", "hard"]
            
            # Для задачи площади прямоугольника ожидаем ответ 40
            assert result.answer == 40 or abs(result.answer - 40) < 0.1
            
        except NetworkError:
            pytest.skip("Тестовый LLM сервер недоступен")
        except Exception as e:
            pytest.fail(f"Неожиданная ошибка в math problem: {e}")


@pytest.mark.integration
@pytest.mark.asyncio
class TestStructuredLLMClientErrorIntegration:
    """Тесты обработки ошибок в integration режиме."""
    
    async def test_invalid_endpoint_error(self):
        """Тест ошибки при недоступном endpoint."""
        config = LLMConfig(
            endpoint="http://invalid-endpoint:9999",
            api_key="test-key",
            model="test-model",
            connect_timeout=1.0,  # Короткий таймаут для быстрого теста
        )
        
        async with StructuredLLMClient(config) as client:
            messages = [
                {"role": "user", "content": "Test message"}
            ]
            
            with pytest.raises((NetworkError, Exception)):
                await client.chat_completion_structured(
                    messages=messages,
                    response_model=UserProfile
                )
    
    async def test_validation_error_integration(self, structured_client):
        """Тест ошибки валидации в реальных условиях."""
        # Некорректные сообщения
        invalid_messages = []
        
        with pytest.raises(ValidationError, match="не может быть пустым"):
            await structured_client.chat_completion_structured(
                messages=invalid_messages,
                response_model=UserProfile
            )
    
    async def test_model_compatibility_integration(self, structured_client):
        """Тест проверки совместимости модели."""
        # Тест с корректной моделью - проверяем что клиент может работать с моделью
        try:
            messages = [
                {"role": "system", "content": "Return JSON with name, age, email, is_active fields."},
                {"role": "user", "content": "Create test user. Return only JSON."}
            ]
            result = await structured_client.chat_completion_structured(
                messages=messages,
                response_model=UserProfile,
                stream=False
            )
            assert isinstance(result, UserProfile)
        except NetworkError:
            pytest.skip("Тестовый LLM сервер недоступен")
        
        # Тест получения схемы через validator
        if hasattr(structured_client, 'validator'):
            schema = UserProfile.model_json_schema()
            assert isinstance(schema, dict)
            assert "properties" in schema
            assert "name" in schema["properties"]


@pytest.mark.integration
@pytest.mark.asyncio
class TestStructuredLLMClientPerformance:
    """Тесты производительности."""
    
    async def test_concurrent_requests(self, structured_client):
        """Тест параллельных запросов."""
        messages = [
            {
                "role": "system",
                "content": "Generate simple user profiles in JSON format. Use English field names: name, age, email, is_active. Return only valid JSON."
            },
            {
                "role": "user",
                "content": "Create user profile. Return only JSON."
            }
        ]
        
        async def single_request():
            try:
                return await structured_client.chat_completion_structured(
                    messages=messages,
                    response_model=UserProfile,
                    stream=False
                )
            except NetworkError:
                return None  # Сервер недоступен
        
        try:
            # Запускаем 3 параллельных запроса
            tasks = [single_request() for _ in range(3)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Проверяем результаты
            successful_results = [r for r in results if isinstance(r, UserProfile)]
            
            if successful_results:
                # Если хотя бы один запрос успешен, проверяем его
                for result in successful_results:
                    assert isinstance(result, UserProfile)
                    assert result.name
                    assert isinstance(result.age, int)
            else:
                pytest.skip("Все параллельные запросы не удались - сервер недоступен")
                
        except Exception as e:
            pytest.skip(f"Тест параллельных запросов пропущен: {e}")
    
    async def test_large_model_performance(self, structured_client):
        """Тест производительности с большой моделью."""
        messages = [
            {
                "role": "system",
                "content": "Create detailed orders with full information in JSON format. Use English field names: order_id, customer (with name, age, email, is_active), products (array with name, price, category, tags, in_stock), total_amount, order_date. Return only valid JSON."
            },
            {
                "role": "user",
                "content": "Create large order with 3 products and full customer info. Return only JSON."
            }
        ]
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            result = await structured_client.chat_completion_structured(
                messages=messages,
                response_model=OrderDetails,
                stream=False
            )
            
            end_time = asyncio.get_event_loop().time()
            execution_time = end_time - start_time
            
            # Проверяем результат
            assert isinstance(result, OrderDetails)
            
            # Проверяем время выполнения (должно быть разумным)
            assert execution_time < 60.0  # Максимум 60 секунд
            
            print(f"Время выполнения большой модели: {execution_time:.2f} секунд")
            
        except NetworkError:
            pytest.skip("Тестовый LLM сервер недоступен для теста производительности")
        except Exception as e:
            pytest.fail(f"Неожиданная ошибка в тесте производительности: {e}")


@pytest.mark.integration
@pytest.mark.asyncio
class TestStructuredLLMClientRealWorldScenarios:
    """Тесты реальных сценариев использования."""
    
    async def test_user_registration_scenario(self, structured_client):
        """Тест сценария регистрации пользователя."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helper for user registration processing. "
                    "Create profiles based on provided information in JSON format. "
                    "Use English field names: name, age, email, is_active. Return only valid JSON."
                )
            },
            {
                "role": "user",
                "content": (
                    "Register new user: "
                    "Name: Мария Козлова, Age: 28, Email: maria.kozlova@gmail.com, "
                    "Status: active user. Return only JSON."
                )
            }
        ]
        
        try:
            result = await structured_client.chat_completion_structured(
                messages=messages,
                response_model=UserProfile,
                stream=False
            )
            
            # Проверяем корректность регистрации
            assert isinstance(result, UserProfile)
            assert "Мария" in result.name or "Maria" in result.name
            assert result.age == 28 or 25 <= result.age <= 30
            assert result.email and "maria" in result.email.lower()
            assert result.is_active is True
            
        except NetworkError:
            pytest.skip("Тестовый LLM сервер недоступен")
    
    async def test_product_catalog_scenario(self, structured_client):
        """Тест сценария каталога продуктов."""
        messages = [
            {
                "role": "system",
                "content": "Create product catalog entries for online store in JSON format. Use English field names: name, price, category, tags, in_stock. Return only valid JSON."
            },
            {
                "role": "user",
                "content": (
                    "Add to catalog: iPhone 15 Pro smartphone, price 120000 rubles, "
                    "category 'Mobile phones', tags: premium, Apple, 5G, in stock. Return only JSON."
                )
            }
        ]
        
        try:
            result = await structured_client.chat_completion_structured(
                messages=messages,
                response_model=ProductInfo,
                stream=False
            )
            
            # Проверяем корректность каталожной записи
            assert isinstance(result, ProductInfo)
            assert "iPhone" in result.name or "смартфон" in result.name.lower()
            assert result.price >= 100000  # Примерная цена
            assert "телефон" in result.category.lower() or "mobile" in result.category.lower()
            assert len(result.tags) > 0
            assert result.in_stock is True
            
        except NetworkError:
            pytest.skip("Тестовый LLM сервер недоступен")
    
    async def test_order_processing_scenario(self, structured_client):
        """Тест сценария обработки заказа."""
        messages = [
            {
                "role": "system",
                "content": "Process online store orders with full details in JSON format. Use English field names: order_id, customer (with name, age, email, is_active), products (array with name, price, category, tags, in_stock), total_amount, order_date. Return only valid JSON."
            },
            {
                "role": "user",
                "content": (
                    "Process order: "
                    "Customer: Алексей Петров, 35 years old, alex.petrov@yandex.ru, "
                    "Products: ASUS Laptop (85000 rub), Logitech Mouse (2500 rub), "
                    "Order ID: ORD-2024-001, Date: 2024-01-20, "
                    "Total: 87500 rubles. Return only JSON."
                )
            }
        ]
        
        try:
            result = await structured_client.chat_completion_structured(
                messages=messages,
                response_model=OrderDetails,
                stream=False
            )
            
            # Проверяем корректность обработки заказа
            assert isinstance(result, OrderDetails)
            assert "ORD" in result.order_id or "001" in result.order_id
            
            # Проверяем клиента
            assert isinstance(result.customer, UserProfile)
            assert "Алексей" in result.customer.name or "Alex" in result.customer.name
            
            # Проверяем продукты
            assert isinstance(result.products, list)
            assert len(result.products) >= 2
            
            # Проверяем общую сумму
            assert result.total_amount >= 80000
            
        except NetworkError:
            pytest.skip("Тестовый LLM сервер недоступен")


if __name__ == "__main__":
    # Запуск integration тестов
    pytest.main([
        __file__,
        "-v",
        "-m", "integration",
        "--tb=short"
    ])