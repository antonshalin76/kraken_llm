"""
Unit tests for Kraken configuration system.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest
from pydantic import ValidationError

from kraken_llm.config.settings import LLMConfig
from kraken_llm.config.defaults import (
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
)


class TestLLMConfig:
    """Test cases for LLMConfig class."""
    
    def test_default_configuration(self):
        """Test that default configuration works without any parameters."""
        config = LLMConfig()
        
        assert config.endpoint == DEFAULT_ENDPOINT
        assert config.model == DEFAULT_MODEL
        assert config.api_key is None
        assert config.temperature == DEFAULT_TEMPERATURE
        assert config.max_tokens == DEFAULT_MAX_TOKENS
        assert config.stream is False
        assert config.ssl_verify is True
    
    def test_parameter_override(self):
        """Test that parameters can be overridden during initialization."""
        config = LLMConfig(
            endpoint="http://custom:8080",
            model="custom-model",
            api_key="test-key",
            temperature=0.9,
            max_tokens=2000,
            stream=True,
        )
        
        assert config.endpoint == "http://custom:8080"
        assert config.model == "custom-model"
        assert config.api_key == "test-key"
        assert config.temperature == 0.9
        assert config.max_tokens == 2000
        assert config.stream is True
    
    def test_environment_variable_loading(self, monkeypatch):
        """Test that environment variables are loaded correctly."""
        monkeypatch.setenv("LLM_ENDPOINT", "http://env:9090")
        monkeypatch.setenv("LLM_MODEL", "env-model")
        monkeypatch.setenv("LLM_API_KEY", "env-key")
        monkeypatch.setenv("LLM_TEMPERATURE", "0.5")
        monkeypatch.setenv("LLM_MAX_TOKENS", "1500")
        monkeypatch.setenv("LLM_STREAM", "true")
        
        config = LLMConfig()
        
        assert config.endpoint == "http://env:9090"
        assert config.model == "env-model"
        assert config.api_key == "env-key"
        assert config.temperature == 0.5
        assert config.max_tokens == 1500
        assert config.stream is True
    
    def test_env_file_loading(self, tmp_path):
        """Test that .env file is loaded correctly."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "LLM_ENDPOINT=http://file:7070\n"
            "LLM_MODEL=file-model\n"
            "LLM_API_KEY=file-key\n"
            "LLM_TEMPERATURE=0.3\n"
            "LLM_MAX_TOKENS=800\n"
        )
        
        # Change to temp directory so .env file is found
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            config = LLMConfig()
            
            assert config.endpoint == "http://file:7070"
            assert config.model == "file-model"
            assert config.api_key == "file-key"
            assert config.temperature == 0.3
            assert config.max_tokens == 800
        finally:
            os.chdir(original_cwd)
    
    def test_parameter_validation(self):
        """Test that parameter validation works correctly."""
        # Test temperature validation
        with pytest.raises(ValidationError):
            LLMConfig(temperature=-0.1)  # Below minimum
        
        with pytest.raises(ValidationError):
            LLMConfig(temperature=2.1)   # Above maximum
        
        # Test max_tokens validation
        with pytest.raises(ValidationError):
            LLMConfig(max_tokens=0)      # Not positive
        
        with pytest.raises(ValidationError):
            LLMConfig(max_tokens=100001) # Above maximum
        
        # Test top_p validation
        with pytest.raises(ValidationError):
            LLMConfig(top_p=-0.1)        # Below minimum
        
        with pytest.raises(ValidationError):
            LLMConfig(top_p=1.1)         # Above maximum
        
        # Test penalty validation
        with pytest.raises(ValidationError):
            LLMConfig(frequency_penalty=-2.1)  # Below minimum
        
        with pytest.raises(ValidationError):
            LLMConfig(presence_penalty=2.1)    # Above maximum
    
    def test_endpoint_normalization(self):
        """Test that endpoint URLs are normalized correctly."""
        # Test trailing slash removal
        config = LLMConfig(endpoint="http://example.com/")
        assert config.endpoint == "http://example.com"
        
        # Test multiple trailing slashes
        config = LLMConfig(endpoint="http://example.com///")
        assert config.endpoint == "http://example.com"
        
        # Test no trailing slash (should remain unchanged)
        config = LLMConfig(endpoint="http://example.com")
        assert config.endpoint == "http://example.com"
    
    def test_timeout_config_property(self):
        """Test timeout_config property returns correct values."""
        config = LLMConfig(
            connect_timeout=5.0,
            read_timeout=30.0,
            write_timeout=15.0,
        )
        
        timeout_config = config.timeout_config
        
        assert timeout_config == {
            "connect": 5.0,
            "read": 30.0,
            "write": 15.0,
        }
    
    def test_generation_params_property(self):
        """Test generation_params property returns correct values."""
        config = LLMConfig(
            temperature=0.8,
            max_tokens=1500,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            stop=["END", "STOP"],
        )
        
        params = config.generation_params
        
        assert params == {
            "temperature": 0.8,
            "max_tokens": 1500,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2,
            "stop": ["END", "STOP"],
        }
    
    def test_generation_params_without_stop(self):
        """Test generation_params property when stop is None."""
        config = LLMConfig(stop=None)
        params = config.generation_params
        
        assert "stop" not in params
    
    def test_to_openai_params(self):
        """Test to_openai_params method."""
        config = LLMConfig(
            model="test-model",
            temperature=0.7,
            max_tokens=1000,
        )
        
        params = config.to_openai_params()
        
        assert params["model"] == "test-model"
        assert params["temperature"] == 0.7
        assert params["max_tokens"] == 1000
        assert "top_p" in params
        assert "frequency_penalty" in params
        assert "presence_penalty" in params
    
    def test_to_openai_params_with_overrides(self):
        """Test to_openai_params method with parameter overrides."""
        config = LLMConfig(
            model="test-model",
            temperature=0.7,
            max_tokens=1000,
        )
        
        params = config.to_openai_params(
            temperature=0.9,
            stream=True,
            custom_param="custom_value",
        )
        
        assert params["model"] == "test-model"
        assert params["temperature"] == 0.9  # Overridden
        assert params["max_tokens"] == 1000   # From config
        assert params["stream"] is True       # Override
        assert params["custom_param"] == "custom_value"  # Custom override
    
    def test_repr_hides_api_key(self):
        """Test that __repr__ hides sensitive API key information."""
        config = LLMConfig(api_key="secret-key-123")
        repr_str = repr(config)
        
        assert "secret-key-123" not in repr_str
        assert "***" in repr_str
    
    def test_repr_without_api_key(self):
        """Test that __repr__ works correctly when no API key is set."""
        config = LLMConfig()
        repr_str = repr(config)
        
        assert "api_key=None" in repr_str
        assert "LLMConfig(" in repr_str
    
    def test_case_insensitive_env_vars(self, monkeypatch):
        """Test that environment variables are case insensitive."""
        monkeypatch.setenv("llm_endpoint", "http://lowercase:8080")
        monkeypatch.setenv("LLM_MODEL", "UPPERCASE-MODEL")
        
        config = LLMConfig()
        
        assert config.endpoint == "http://lowercase:8080"
        assert config.model == "UPPERCASE-MODEL"
    
    def test_extra_env_vars_ignored(self, monkeypatch):
        """Test that extra environment variables are ignored."""
        monkeypatch.setenv("LLM_UNKNOWN_PARAM", "should-be-ignored")
        monkeypatch.setenv("LLM_ENDPOINT", "http://valid:8080")
        
        # Should not raise an error
        config = LLMConfig()
        assert config.endpoint == "http://valid:8080"
        
        # Unknown parameter should not be accessible
        assert not hasattr(config, "unknown_param")


@pytest.fixture
def sample_config():
    """Fixture providing a sample configuration for testing."""
    return LLMConfig(
        endpoint="http://test:8080",
        model="test-model",
        api_key="test-key",
        temperature=0.8,
        max_tokens=1500,
    )


class TestConfigIntegration:
    """Integration tests for configuration system."""
    
    def test_config_with_all_parameters(self):
        """Test configuration with all parameters set."""
        config = LLMConfig(
            endpoint="http://full:8080",
            api_key="full-key",
            model="full-model",
            temperature=0.9,
            max_tokens=2000,
            top_p=0.8,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            stop=["STOP", "END"],
            stream=True,
            connect_timeout=15.0,
            read_timeout=120.0,
            write_timeout=20.0,
            ssl_verify=False,
            max_retries=5,
            retry_delay=2.0,
            log_level="DEBUG",
        )
        
        # Verify all parameters are set correctly
        assert config.endpoint == "http://full:8080"
        assert config.api_key == "full-key"
        assert config.model == "full-model"
        assert config.temperature == 0.9
        assert config.max_tokens == 2000
        assert config.top_p == 0.8
        assert config.frequency_penalty == 0.1
        assert config.presence_penalty == 0.2
        assert config.stop == ["STOP", "END"]
        assert config.stream is True
        assert config.connect_timeout == 15.0
        assert config.read_timeout == 120.0
        assert config.write_timeout == 20.0
        assert config.ssl_verify is False
        assert config.max_retries == 5
        assert config.retry_delay == 2.0
        assert config.log_level == "DEBUG"
    
    def test_config_serialization(self, sample_config):
        """Test that configuration can be serialized and deserialized."""
        # Convert to dict
        config_dict = sample_config.model_dump()
        
        # Create new config from dict
        new_config = LLMConfig(**config_dict)
        
        # Verify they're equivalent
        assert new_config.endpoint == sample_config.endpoint
        assert new_config.model == sample_config.model
        assert new_config.api_key == sample_config.api_key
        assert new_config.temperature == sample_config.temperature
        assert new_config.max_tokens == sample_config.max_tokens