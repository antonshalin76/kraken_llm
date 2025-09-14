#!/usr/bin/env python3
"""
Тесты для UniversalLLMClient

Проверяет функциональность универсального клиента,
включая создание, конфигурацию и использование различных возможностей.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any
from pydantic import BaseModel

from kraken_llm.config.settings import LLMConfig
from kraken_llm.client.universal import (
    UniversalLLMClient,
    UniversalClientConfig,
    UniversalCapability,
    create_universal_client,
    create_universal_client_from_report,
    create_basic_client,
    create_advanced_client,
    create_full_client
)
from kraken_llm.exceptions.validation import ValidationError


class SampleResponseModel(BaseModel):
    """Тестовая модель для structured output"""
    name: str
    value: int
    tags: List[str] = []


class TestUniversalClientConfig:
    """Тесты конфигурации универсального клиента"""
    
    def test_basic_config(self):
        """Тест базовой конфигурации"""
        config = UniversalClientConfig.basic()
        
        assert UniversalCapability.CHAT_COMPLETION in config.capabilities
        assert UniversalCapability.STREAMING in config.capabilities
        assert len(config.capabilities) == 2
    
    def test_advanced_config(self):
        """Тест продвинутой конфигурации"""
        config = UniversalClientConfig.advanced()
        
        expected_capabilities = {
            UniversalCapability.CHAT_COMPLETION,
            UniversalCapability.STREAMING,
            UniversalCapability.STRUCTURED_OUTPUT,
            UniversalCapability.FUNCTION_CALLING,
            UniversalCapability.TOOL_CALLING,
            UniversalCapability.REASONING
        }
        
        assert config.capabilities == expected_capabilities
        assert config.reasoning_config is not None
        assert config.prefer_streaming is True
    
    def test_all_capabilities_config(self):
        """Тест конфигурации со всеми возможностями"""
        config = UniversalClientConfig.all_capabilities()
        
        assert len(config.capabilities) == len(UniversalCapability)
        assert config.reasoning_config is not None
        assert config.multimodal_config is not None
        assert config.adaptive_config is not None
        assert config.asr_config is not None
    
    def test_from_capabilities_report(self):
        """Тест создания конфигурации из отчета анализатора"""
        mock_report = {
            "model_summaries": {
                "test_model": {
                    "confirmed_capabilities": [
                        {"capability": "chat_completion", "success_rate": 1.0},
                        {"capability": "streaming", "success_rate": 0.9},
                        {"capability": "structured_output_native", "success_rate": 0.8},
                    ]
                }
            }
        }
        
        config = UniversalClientConfig.from_capabilities_report(
            mock_report, model_name="test_model"
        )
        
        assert UniversalCapability.CHAT_COMPLETION in config.capabilities
        assert UniversalCapability.STREAMING in config.capabilities
        assert UniversalCapability.STRUCTURED_OUTPUT in config.capabilities
    
    def test_from_capabilities_report_no_model(self):
        """Тест создания конфигурации без указания модели"""
        mock_report = {
            "model_summaries": {
                "first_model": {
                    "confirmed_capabilities": [
                        {"capability": "chat_completion", "success_rate": 1.0},
                    ]
                }
            }
        }
        
        config = UniversalClientConfig.from_capabilities_report(mock_report)
        
        assert UniversalCapability.CHAT_COMPLETION in config.capabilities
    
    def test_from_empty_report(self):
        """Тест создания конфигурации из пустого отчета"""
        empty_report = {"model_summaries": {}}
        
        config = UniversalClientConfig.from_capabilities_report(empty_report)
        
        # Должна вернуться базовая конфигурация
        assert UniversalCapability.CHAT_COMPLETION in config.capabilities


class TestUniversalLLMClient:
    """Тесты универсального клиента"""
    
    @pytest.fixture
    def mock_config(self):
        """Мок конфигурации LLM"""
        return LLMConfig(
            endpoint="http://test.local",
            api_key="test_key",
            model="test_model"
        )
    
    @pytest.fixture
    def basic_universal_config(self):
        """Базовая конфигурация универсального клиента"""
        return UniversalClientConfig.basic()
    
    def test_init(self, mock_config, basic_universal_config):
        """Тест инициализации клиента"""
        client = UniversalLLMClient(mock_config, basic_universal_config)
        
        assert client.config == mock_config
        assert client.universal_config == basic_universal_config
        assert not client._initialized
        assert len(client._clients) == 0
    
    def test_init_without_universal_config(self, mock_config):
        """Тест инициализации без универсальной конфигурации"""
        client = UniversalLLMClient(mock_config)
        
        # Должна использоваться базовая конфигурация
        assert UniversalCapability.CHAT_COMPLETION in client.universal_config.capabilities
        assert UniversalCapability.STREAMING in client.universal_config.capabilities
    
    @pytest.mark.asyncio
    async def test_context_manager(self, mock_config, basic_universal_config):
        """Тест контекстного менеджера"""
        client = UniversalLLMClient(mock_config, basic_universal_config)
        
        with patch.object(client, '_initialize_clients', new_callable=AsyncMock) as mock_init:
            with patch.object(client, '_cleanup_clients', new_callable=AsyncMock) as mock_cleanup:
                async with client:
                    assert mock_init.called
                
                assert mock_cleanup.called
    
    @pytest.mark.asyncio
    async def test_initialize_clients(self, mock_config):
        """Тест инициализации клиентов"""
        config = UniversalClientConfig(
            capabilities={
                UniversalCapability.CHAT_COMPLETION,
                UniversalCapability.STREAMING,
                UniversalCapability.STRUCTURED_OUTPUT
            }
        )
        
        client = UniversalLLMClient(mock_config, config)
        
        # Мокаем создание клиентов
        with patch('kraken_llm.client.universal.StandardLLMClient') as mock_standard:
            with patch('kraken_llm.client.universal.StreamingLLMClient') as mock_streaming:
                with patch('kraken_llm.client.universal.StructuredLLMClient') as mock_structured:
                    
                    # Настраиваем моки
                    mock_standard.return_value.__aenter__ = AsyncMock()
                    mock_streaming.return_value.__aenter__ = AsyncMock()
                    mock_structured.return_value.__aenter__ = AsyncMock()
                    
                    await client._initialize_clients()
                    
                    assert client._initialized
                    assert 'standard' in client._clients
                    assert 'streaming' in client._clients
                    assert 'structured' in client._clients
    
    def test_get_optimal_client(self, mock_config):
        """Тест выбора оптимального клиента"""
        config = UniversalClientConfig.advanced()
        client = UniversalLLMClient(mock_config, config)
        
        # Мокаем клиенты
        client._clients = {
            'standard': Mock(),
            'streaming': Mock(),
            'structured': Mock(),
            'reasoning': Mock(),
            'adaptive': Mock()
        }
        
        # Тест выбора для structured output
        optimal = client._get_optimal_client('structured')
        assert optimal == client._clients['structured']
        
        # Тест выбора для reasoning
        optimal = client._get_optimal_client('reasoning')
        assert optimal == client._clients['reasoning']
        
        # Тест выбора для streaming
        optimal = client._get_optimal_client('chat', stream=True)
        assert optimal == client._clients['streaming']
        
        # Тест fallback с prefer_streaming=True (должен вернуть streaming)
        optimal = client._get_optimal_client('unknown')
        assert optimal is client._clients['streaming'], "Should return streaming client when prefer_streaming=True"
        
        # Тест fallback на adaptive (отключаем prefer_streaming)
        client.universal_config.prefer_streaming = False
        optimal = client._get_optimal_client('unknown')
        assert optimal is client._clients['adaptive'], "Should return adaptive client when prefer_streaming=False"
        
        # Тест fallback на standard (если adaptive недоступен)
        client._clients.pop('adaptive')  # Удаляем adaptive
        optimal = client._get_optimal_client('unknown')
        assert optimal == client._clients['standard']
    
    def test_get_optimal_client_edge_cases(self, mock_config):
        """Тест edge cases выбора оптимального клиента"""
        config = UniversalClientConfig.basic()
        client = UniversalLLMClient(mock_config, config)
        
        # Только standard клиент
        client._clients = {'standard': Mock()}
        
        # Все операции должны fallback на standard
        assert client._get_optimal_client('structured') == client._clients['standard']
        assert client._get_optimal_client('reasoning') == client._clients['standard']
        assert client._get_optimal_client('unknown') == client._clients['standard']
        
        # Тест с пустым словарем клиентов
        client._clients = {}
        assert client._get_optimal_client('any') is None
    
    def test_get_available_capabilities(self, mock_config):
        """Тест получения доступных возможностей"""
        config = UniversalClientConfig(
            capabilities={
                UniversalCapability.CHAT_COMPLETION,
                UniversalCapability.STREAMING
            }
        )
        client = UniversalLLMClient(mock_config, config)
        
        capabilities = client.get_available_capabilities()
        
        assert 'chat_completion' in capabilities
        assert 'streaming' in capabilities
        assert len(capabilities) == 2
    
    def test_get_active_clients(self, mock_config):
        """Тест получения активных клиентов"""
        client = UniversalLLMClient(mock_config)
        client._clients = {'standard': Mock(), 'streaming': Mock()}
        
        active_clients = client.get_active_clients()
        
        assert 'standard' in active_clients
        assert 'streaming' in active_clients
        assert len(active_clients) == 2
    
    def test_get_client_info(self, mock_config):
        """Тест получения информации о клиенте"""
        config = UniversalClientConfig.basic()
        client = UniversalLLMClient(mock_config, config)
        client._clients = {'standard': Mock()}
        client._initialized = True
        
        info = client.get_client_info()
        
        assert 'capabilities' in info
        assert 'active_clients' in info
        assert 'config' in info
        assert 'initialized' in info
        assert info['initialized'] is True
    
    @pytest.mark.asyncio
    async def test_chat_completion(self, mock_config):
        """Тест базового chat completion"""
        client = UniversalLLMClient(mock_config)
        
        # Мокаем инициализацию и клиент
        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = "test response"
        
        with patch.object(client, '_initialize_clients', new_callable=AsyncMock):
            with patch.object(client, '_get_optimal_client', return_value=mock_client):
                
                response = await client.chat_completion([
                    {"role": "user", "content": "test"}
                ])
                
                assert response == "test response"
                mock_client.chat_completion.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_chat_completion_no_client(self, mock_config):
        """Тест chat completion без доступного клиента"""
        client = UniversalLLMClient(mock_config)
        
        with patch.object(client, '_initialize_clients', new_callable=AsyncMock):
            with patch.object(client, '_get_optimal_client', return_value=None):
                
                with pytest.raises(Exception):  # KrakenError
                    await client.chat_completion([
                        {"role": "user", "content": "test"}
                    ])
    
    @pytest.mark.asyncio
    async def test_chat_completion_structured(self, mock_config):
        """Тест structured chat completion"""
        config = UniversalClientConfig(
            capabilities={UniversalCapability.STRUCTURED_OUTPUT}
        )
        client = UniversalLLMClient(mock_config, config)
        
        mock_client = AsyncMock()
        mock_response = SampleResponseModel(name="test", value=42)
        mock_client.chat_completion_structured.return_value = mock_response
        
        with patch.object(client, '_initialize_clients', new_callable=AsyncMock):
            with patch.object(client, '_get_optimal_client', return_value=mock_client):
                
                response = await client.chat_completion_structured([
                    {"role": "user", "content": "test"}
                ], response_model=SampleResponseModel)
                
                assert response == mock_response
                assert isinstance(response, SampleResponseModel)
                mock_client.chat_completion_structured.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_chat_completion_structured_not_configured(self, mock_config):
        """Тест structured output без конфигурации"""
        config = UniversalClientConfig(
            capabilities={UniversalCapability.CHAT_COMPLETION}  # Без STRUCTURED_OUTPUT
        )
        client = UniversalLLMClient(mock_config, config)
        
        with patch.object(client, '_initialize_clients', new_callable=AsyncMock):
            with pytest.raises(ValidationError):
                await client.chat_completion_structured([
                    {"role": "user", "content": "test"}
                ], response_model=SampleResponseModel)
    
    @pytest.mark.asyncio
    async def test_chat_completion_structured_fallback(self, mock_config):
        """Тест fallback логики для structured output"""
        config = UniversalClientConfig(
            capabilities={UniversalCapability.STRUCTURED_OUTPUT}
        )
        client = UniversalLLMClient(mock_config, config)
        
        # Мокаем клиент, который не поддерживает нативный structured output
        mock_client = AsyncMock()
        mock_client.chat_completion_structured.side_effect = Exception("Native SO failed")
        mock_client._structured_stream_outlines.return_value = SampleResponseModel(name="outlines", value=99)
        
        with patch.object(client, '_initialize_clients', new_callable=AsyncMock):
            with patch.object(client, '_get_optimal_client', return_value=mock_client):
                
                response = await client.chat_completion_structured([
                    {"role": "user", "content": "test"}
                ], response_model=SampleResponseModel)
                
                # Должен использовать Outlines fallback
                assert response.name == "outlines"
                assert response.value == 99
                mock_client._structured_stream_outlines.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_chat_completion_structured_final_fallback(self, mock_config):
        """Тест финального fallback через обычный chat completion"""
        config = UniversalClientConfig(
            capabilities={UniversalCapability.STRUCTURED_OUTPUT}
        )
        client = UniversalLLMClient(mock_config, config)
        
        # Мокаем клиент, у которого все structured методы не работают
        mock_client = AsyncMock()
        mock_client.chat_completion_structured.side_effect = Exception("Native SO failed")
        # Удаляем Outlines методы
        del mock_client._structured_stream_outlines
        del mock_client.structured_completion
        
        # Мокаем финальный fallback
        mock_fallback_response = SampleResponseModel(name="fallback", value=123)
        
        with patch.object(client, '_initialize_clients', new_callable=AsyncMock):
            with patch.object(client, '_get_optimal_client', return_value=mock_client):
                with patch.object(client, '_structured_fallback', return_value=mock_fallback_response) as mock_fallback:
                    
                    response = await client.chat_completion_structured([
                        {"role": "user", "content": "test"}
                    ], response_model=SampleResponseModel)
                    
                    # Должен использовать финальный fallback
                    assert response.name == "fallback"
                    assert response.value == 123
                    mock_fallback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_reasoning_completion(self, mock_config):
        """Тест reasoning completion"""
        config = UniversalClientConfig(
            capabilities={UniversalCapability.REASONING}
        )
        client = UniversalLLMClient(mock_config, config)
        
        mock_client = AsyncMock()
        mock_client.reasoning_completion.return_value = "reasoning response"
        
        with patch.object(client, '_initialize_clients', new_callable=AsyncMock):
            with patch.object(client, '_get_optimal_client', return_value=mock_client):
                
                response = await client.reasoning_completion([
                    {"role": "user", "content": "solve this"}
                ])
                
                assert response == "reasoning response"
                mock_client.reasoning_completion.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_reasoning_completion_fallback(self, mock_config):
        """Тест fallback для reasoning completion"""
        config = UniversalClientConfig(
            capabilities={UniversalCapability.REASONING}
        )
        client = UniversalLLMClient(mock_config, config)
        
        # Клиент без reasoning_completion метода
        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = "fallback response"
        del mock_client.reasoning_completion  # Удаляем метод
        
        with patch.object(client, '_initialize_clients', new_callable=AsyncMock):
            with patch.object(client, '_get_optimal_client', return_value=mock_client):
                with patch.object(client, 'chat_completion', return_value="fallback response") as mock_chat:
                    
                    response = await client.reasoning_completion([
                        {"role": "user", "content": "solve this"}
                    ])
                    
                    assert response == "fallback response"
                    mock_chat.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_register_function(self, mock_config):
        """Тест регистрации функции"""
        client = UniversalLLMClient(mock_config)
        
        # Мокаем клиенты
        mock_client1 = Mock()
        mock_client2 = Mock()
        mock_client3 = Mock()  # Без register_function
        
        client._clients = {
            'client1': mock_client1,
            'client2': mock_client2,
            'client3': mock_client3
        }
        
        def test_func():
            return "test"
        
        client.register_function("test_func", test_func, "Test function")
        
        # Проверяем, что функция зарегистрирована в поддерживающих клиентах
        mock_client1.register_function.assert_called_once_with(
            "test_func", test_func, "Test function", None
        )
        mock_client2.register_function.assert_called_once_with(
            "test_func", test_func, "Test function", None
        )
        
        # client3 не должен вызываться, так как у него нет register_function
    
    @pytest.mark.asyncio
    async def test_test_capabilities(self, mock_config):
        """Тест проверки возможностей"""
        config = UniversalClientConfig(
            capabilities={
                UniversalCapability.CHAT_COMPLETION,
                UniversalCapability.STREAMING
            }
        )
        client = UniversalLLMClient(mock_config, config)
        
        with patch.object(client, '_initialize_clients', new_callable=AsyncMock):
            with patch.object(client, 'chat_completion', return_value="test") as mock_chat:
                with patch.object(client, 'chat_completion_stream') as mock_stream:
                    
                    # Настраиваем async generator для streaming
                    async def mock_stream_gen():
                        yield "chunk1"
                        yield "chunk2"
                    
                    mock_stream.return_value = mock_stream_gen()
                    
                    results = await client.test_capabilities()
                    
                    assert results["chat_completion"] is True
                    assert results["streaming"] is True


class TestConvenienceFunctions:
    """Тесты удобных функций создания клиентов"""
    
    @pytest.fixture
    def mock_llm_config(self):
        """Мок LLMConfig"""
        with patch('kraken_llm.client.universal.LLMConfig') as mock:
            mock.return_value = Mock()
            yield mock
    
    def test_create_universal_client(self, mock_llm_config):
        """Тест create_universal_client"""
        capabilities = {UniversalCapability.CHAT_COMPLETION}
        
        with patch('kraken_llm.client.universal.UniversalLLMClient') as mock_client:
            client = create_universal_client(capabilities=capabilities)
            
            mock_client.assert_called_once()
            # Проверяем, что передана правильная конфигурация
            args, kwargs = mock_client.call_args
            assert args[1].capabilities == capabilities
    
    def test_create_basic_client(self, mock_llm_config):
        """Тест create_basic_client"""
        with patch('kraken_llm.client.universal.UniversalLLMClient') as mock_client:
            client = create_basic_client()
            
            mock_client.assert_called_once()
            # Проверяем базовую конфигурацию
            args, kwargs = mock_client.call_args
            config = args[1]
            assert UniversalCapability.CHAT_COMPLETION in config.capabilities
            assert UniversalCapability.STREAMING in config.capabilities
    
    def test_create_advanced_client(self, mock_llm_config):
        """Тест create_advanced_client"""
        with patch('kraken_llm.client.universal.UniversalLLMClient') as mock_client:
            client = create_advanced_client()
            
            mock_client.assert_called_once()
            # Проверяем продвинутую конфигурацию
            args, kwargs = mock_client.call_args
            config = args[1]
            assert UniversalCapability.STRUCTURED_OUTPUT in config.capabilities
            assert UniversalCapability.REASONING in config.capabilities
    
    def test_create_full_client(self, mock_llm_config):
        """Тест create_full_client"""
        with patch('kraken_llm.client.universal.UniversalLLMClient') as mock_client:
            client = create_full_client()
            
            mock_client.assert_called_once()
            # Проверяем полную конфигурацию
            args, kwargs = mock_client.call_args
            config = args[1]
            assert len(config.capabilities) == len(UniversalCapability)
    
    def test_create_universal_client_from_report(self, mock_llm_config):
        """Тест create_universal_client_from_report"""
        mock_report = {
            "model_summaries": {
                "test_model": {
                    "confirmed_capabilities": [
                        {"capability": "chat_completion", "success_rate": 1.0}
                    ]
                }
            }
        }
        
        with patch('kraken_llm.client.universal.UniversalLLMClient') as mock_client:
            client = create_universal_client_from_report(mock_report, model_name="test_model")
            
            mock_client.assert_called_once()
            # Проверяем, что конфигурация создана из отчета
            args, kwargs = mock_client.call_args
            config = args[1]
            assert UniversalCapability.CHAT_COMPLETION in config.capabilities


if __name__ == "__main__":
    pytest.main([__file__])