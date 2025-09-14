"""
Интеграционные тесты для фабрики клиентов Kraken.

Тестирует создание различных типов клиентов через фабрику
и их базовую функциональность.
"""

import pytest
import asyncio
from typing import Dict, Any

from kraken_llm import (
    ClientFactory,
    create_client,
    create_standard_client,
    create_streaming_client,
    create_structured_client,
    create_reasoning_client,
    create_multimodal_client,
    create_adaptive_client,
    create_asr_client,
    create_embeddings_client,
    LLMConfig,
    StandardLLMClient,
    StreamingLLMClient,
    StructuredLLMClient,
    ReasoningLLMClient,
    MultimodalLLMClient,
    AdaptiveLLMClient,
    ASRClient,
    EmbeddingsClient,
)


class TestClientFactory:
    """Тесты фабрики клиентов."""
    
    def test_get_available_client_types(self):
        """Тест получения доступных типов клиентов."""
        available_types = ClientFactory.get_available_client_types()
        
        assert isinstance(available_types, dict)
        assert len(available_types) > 0
        
        # Проверяем наличие основных типов
        expected_types = [
            'standard', 'streaming', 'structured', 'reasoning',
            'multimodal', 'adaptive', 'asr', 'embeddings'
        ]
        
        for client_type in expected_types:
            assert client_type in available_types
    
    def test_create_client_with_explicit_type(self):
        """Тест создания клиента с явным указанием типа."""
        config = LLMConfig(
            endpoint="http://localhost:8080",
            api_key="test_key",
            model="test_model"
        )
        
        # Тестируем создание разных типов клиентов
        test_cases = [
            ('standard', StandardLLMClient),
            ('streaming', StreamingLLMClient),
            ('structured', StructuredLLMClient),
            ('reasoning', ReasoningLLMClient),
            ('multimodal', MultimodalLLMClient),
            ('adaptive', AdaptiveLLMClient),
            ('asr', ASRClient),
            ('embeddings', EmbeddingsClient),
        ]
        
        for client_type, expected_class in test_cases:
            client = ClientFactory.create_client(client_type, config)
            assert isinstance(client, expected_class)
            assert client.config.endpoint == "http://localhost:8080"
            assert client.config.api_key == "test_key"
            assert client.config.model == "test_model"
    
    def test_create_client_with_kwargs(self):
        """Тест создания клиента с параметрами через kwargs."""
        client = ClientFactory.create_client(
            'standard',
            endpoint="http://localhost:8080",
            api_key="test_key",
            model="test_model",
            temperature=0.8
        )
        
        assert isinstance(client, StandardLLMClient)
        assert client.config.endpoint == "http://localhost:8080"
        assert client.config.api_key == "test_key"
        assert client.config.model == "test_model"
        assert client.config.temperature == 0.8
    
    def test_auto_detect_client_type(self):
        """Тест автоматического определения типа клиента."""
        # Тест для structured output
        client = ClientFactory.create_client(
            response_model="SomeModel",
            endpoint="http://localhost:8080"
        )
        assert isinstance(client, StructuredLLMClient)
        
        # Тест для streaming
        client = ClientFactory.create_client(
            stream=True,
            endpoint="http://localhost:8080"
        )
        assert isinstance(client, StreamingLLMClient)
        
        # Тест для reasoning
        client = ClientFactory.create_client(
            reasoning_mode=True,
            endpoint="http://localhost:8080"
        )
        assert isinstance(client, ReasoningLLMClient)
        
        # Тест для multimodal
        client = ClientFactory.create_client(
            media_input=True,
            endpoint="http://localhost:8080"
        )
        assert isinstance(client, MultimodalLLMClient)
        
        # Тест для ASR
        client = ClientFactory.create_client(
            audio_file="test.wav",
            endpoint="http://localhost:8080"
        )
        assert isinstance(client, ASRClient)
        
        # Тест для embeddings
        client = ClientFactory.create_client(
            embeddings=True,
            endpoint="http://localhost:8080"
        )
        assert isinstance(client, EmbeddingsClient)
        
        # Тест по умолчанию (adaptive)
        client = ClientFactory.create_client(
            endpoint="http://localhost:8080"
        )
        assert isinstance(client, AdaptiveLLMClient)
    
    def test_create_client_invalid_type(self):
        """Тест создания клиента с неверным типом."""
        with pytest.raises(ValueError, match="Неизвестный тип клиента"):
            ClientFactory.create_client('invalid_type')
    
    def test_register_custom_client_type(self):
        """Тест регистрации кастомного типа клиента."""
        # Создаем кастомный клиент для тестирования
        class CustomTestClient(StandardLLMClient):
            pass
        
        # Регистрируем новый тип
        ClientFactory.register_client_type('custom_test', CustomTestClient)
        
        # Проверяем, что тип зарегистрирован
        available_types = ClientFactory.get_available_client_types()
        assert 'custom_test' in available_types
        assert available_types['custom_test'] == CustomTestClient
        
        # Создаем клиент нового типа
        client = ClientFactory.create_client('custom_test')
        assert isinstance(client, CustomTestClient)
    
    def test_register_invalid_client_type(self):
        """Тест регистрации неверного типа клиента."""
        class InvalidClient:
            pass
        
        with pytest.raises(ValueError, match="должен наследоваться от BaseLLMClient"):
            ClientFactory.register_client_type('invalid', InvalidClient)


class TestConvenienceFunctions:
    """Тесты удобных функций создания клиентов."""
    
    def test_create_client_function(self):
        """Тест функции create_client."""
        client = create_client(
            'standard',
            endpoint="http://localhost:8080",
            model="test_model"
        )
        
        assert isinstance(client, StandardLLMClient)
        assert client.config.endpoint == "http://localhost:8080"
        assert client.config.model == "test_model"
    
    def test_create_standard_client_function(self):
        """Тест функции create_standard_client."""
        client = create_standard_client(
            endpoint="http://localhost:8080",
            model="test_model"
        )
        
        assert isinstance(client, StandardLLMClient)
        assert client.config.endpoint == "http://localhost:8080"
        assert client.config.model == "test_model"
    
    def test_create_streaming_client_function(self):
        """Тест функции create_streaming_client."""
        client = create_streaming_client(
            endpoint="http://localhost:8080",
            model="test_model"
        )
        
        assert isinstance(client, StreamingLLMClient)
        assert client.config.endpoint == "http://localhost:8080"
        assert client.config.model == "test_model"
    
    def test_create_structured_client_function(self):
        """Тест функции create_structured_client."""
        client = create_structured_client(
            endpoint="http://localhost:8080",
            model="test_model"
        )
        
        assert isinstance(client, StructuredLLMClient)
        assert client.config.endpoint == "http://localhost:8080"
        assert client.config.model == "test_model"
    
    def test_create_reasoning_client_function(self):
        """Тест функции create_reasoning_client."""
        client = create_reasoning_client(
            endpoint="http://localhost:8080",
            model="test_model"
        )
        
        assert isinstance(client, ReasoningLLMClient)
        assert client.config.endpoint == "http://localhost:8080"
        assert client.config.model == "test_model"
    
    def test_create_multimodal_client_function(self):
        """Тест функции create_multimodal_client."""
        client = create_multimodal_client(
            endpoint="http://localhost:8080",
            model="test_model"
        )
        
        assert isinstance(client, MultimodalLLMClient)
        assert client.config.endpoint == "http://localhost:8080"
        assert client.config.model == "test_model"
    
    def test_create_adaptive_client_function(self):
        """Тест функции create_adaptive_client."""
        client = create_adaptive_client(
            endpoint="http://localhost:8080",
            model="test_model"
        )
        
        assert isinstance(client, AdaptiveLLMClient)
        assert client.config.endpoint == "http://localhost:8080"
        assert client.config.model == "test_model"
    
    def test_create_asr_client_function(self):
        """Тест функции create_asr_client."""
        client = create_asr_client(
            endpoint="http://localhost:8080",
            model="test_model"
        )
        
        assert isinstance(client, ASRClient)
        assert client.config.endpoint == "http://localhost:8080"
        assert client.config.model == "test_model"
    
    def test_create_embeddings_client_function(self):
        """Тест функции create_embeddings_client."""
        client = create_embeddings_client(
            endpoint="http://localhost:8080",
            model="test_model"
        )
        
        assert isinstance(client, EmbeddingsClient)
        assert client.config.endpoint == "http://localhost:8080"
        assert client.config.model == "test_model"


class TestClientFactoryIntegration:
    """Интеграционные тесты фабрики клиентов."""
    
    def test_client_factory_with_default_config(self):
        """Тест фабрики с конфигурацией по умолчанию."""
        client = create_client()
        
        # Должен создаться adaptive клиент по умолчанию
        assert isinstance(client, AdaptiveLLMClient)
        
        # Проверяем, что конфигурация создалась
        assert client.config is not None
        assert hasattr(client.config, 'endpoint')
        assert hasattr(client.config, 'model')
    
    def test_client_factory_config_override(self):
        """Тест переопределения конфигурации через kwargs."""
        base_config = LLMConfig(
            endpoint="http://base:8080",
            model="base_model",
            temperature=0.5
        )
        
        client = create_client(
            'standard',
            config=base_config,
            temperature=0.9,  # Переопределяем температуру
            max_tokens=2000   # Добавляем новый параметр
        )
        
        assert isinstance(client, StandardLLMClient)
        assert client.config.endpoint == "http://base:8080"  # Из базовой конфигурации
        assert client.config.model == "base_model"           # Из базовой конфигурации
        assert client.config.temperature == 0.9              # Переопределено
        assert client.config.max_tokens == 2000              # Добавлено
    
    def test_multiple_clients_independence(self):
        """Тест независимости нескольких клиентов."""
        client1 = create_client(
            'standard',
            endpoint="http://server1:8080",
            model="model1"
        )
        
        client2 = create_client(
            'streaming',
            endpoint="http://server2:8080",
            model="model2"
        )
        
        # Проверяем, что клиенты независимы
        assert isinstance(client1, StandardLLMClient)
        assert isinstance(client2, StreamingLLMClient)
        
        assert client1.config.endpoint == "http://server1:8080"
        assert client2.config.endpoint == "http://server2:8080"
        
        assert client1.config.model == "model1"
        assert client2.config.model == "model2"
        
        # Проверяем, что это разные объекты
        assert client1 is not client2
        assert client1.config is not client2.config


if __name__ == "__main__":
    # Запуск тестов
    pytest.main([__file__, "-v"])