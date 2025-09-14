"""
Unit tests for Kraken exception system.
"""

import pytest

from kraken_llm.exceptions import (
    KrakenError,
    NetworkError,
    ConnectionError,
    TimeoutError,
    HTTPError,
    SSLError,
    ValidationError,
    PydanticValidationError,
    JSONValidationError,
    SchemaValidationError,
    ParameterValidationError,
    APIError,
    AuthenticationError,
    RateLimitError,
    ModelError,
    ContentFilterError,
    ServiceUnavailableError,
)


class TestKrakenError:
    """Test cases for base KrakenError class."""

    def test_basic_error_creation(self):
        """Test basic error creation with message only."""
        error = KrakenError("Test error message")

        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.context == {}
        assert error.original_error is None

    def test_error_with_context(self):
        """Test error creation with context information."""
        context = {"key1": "value1", "key2": 42}
        error = KrakenError("Test error", context=context)

        assert error.context == context
        assert "key1=value1" in str(error)
        assert "key2=42" in str(error)

    def test_error_with_original_error(self):
        """Test error creation with original exception."""
        original = ValueError("Original error")
        error = KrakenError("Wrapper error", original_error=original)

        assert error.original_error is original
        assert error.message == "Wrapper error"

    def test_add_context(self):
        """Test adding context to an error."""
        error = KrakenError("Test error")
        result = error.add_context("test_key", "test_value")

        assert result is error  # Should return self for chaining
        assert error.get_context("test_key") == "test_value"
        assert "test_key=test_value" in str(error)

    def test_get_context(self):
        """Test getting context from an error."""
        error = KrakenError("Test error", context={"existing": "value"})

        assert error.get_context("existing") == "value"
        assert error.get_context("missing") is None
        assert error.get_context("missing", "default") == "default"

    def test_repr(self):
        """Test string representation of error."""
        context = {"key": "value"}
        original = ValueError("original")
        error = KrakenError("Test message", context=context,
                            original_error=original)

        repr_str = repr(error)
        assert "KrakenError" in repr_str
        assert "Test message" in repr_str
        assert "{'key': 'value'}" in repr_str
        assert "ValueError" in repr_str


class TestNetworkErrors:
    """Test cases for network-related errors."""

    def test_network_error_basic(self):
        """Test basic NetworkError creation."""
        error = NetworkError("Network failed")

        assert isinstance(error, KrakenError)
        assert error.message == "Network failed"
        assert error.endpoint is None
        assert error.status_code is None

    def test_network_error_with_details(self):
        """Test NetworkError with full details."""
        response_data = {"error": "Server error"}
        error = NetworkError(
            "Request failed",
            endpoint="http://test:8080",
            status_code=500,
            response_data=response_data,
        )

        assert error.endpoint == "http://test:8080"
        assert error.status_code == 500
        assert error.response_data == response_data
        assert error.get_context("endpoint") == "http://test:8080"
        assert error.get_context("status_code") == 500

    def test_connection_error(self):
        """Test ConnectionError creation."""
        error = ConnectionError(endpoint="http://unreachable:8080")

        assert isinstance(error, NetworkError)
        assert "Не удалось установить соединение" in error.message
        assert error.endpoint == "http://unreachable:8080"

    def test_timeout_error(self):
        """Test TimeoutError creation."""
        error = TimeoutError(
            "Read timeout",
            endpoint="http://slow:8080",
            timeout_type="read",
            timeout_value=30.0,
        )

        assert isinstance(error, NetworkError)
        assert error.timeout_type == "read"
        assert error.timeout_value == 30.0
        assert error.get_context("timeout_type") == "read"
        assert error.get_context("timeout_value") == 30.0

    def test_http_error(self):
        """Test HTTPError creation."""
        headers = {"Content-Type": "application/json"}
        response_data = {"error": "Bad request"}

        error = HTTPError(
            "HTTP error occurred",
            status_code=400,
            method="POST",
            url="http://api:8080/chat",
            headers=headers,
            response_data=response_data,
        )

        assert isinstance(error, NetworkError)
        assert error.status_code == 400
        assert error.method == "POST"
        assert error.url == "http://api:8080/chat"
        assert error.headers == headers
        assert error.response_data == response_data

    def test_ssl_error(self):
        """Test SSLError creation."""
        error = SSLError(endpoint="https://invalid-cert:8080")

        assert isinstance(error, NetworkError)
        assert "Произошла ошибка SSL/TLS" in error.message
        assert error.endpoint == "https://invalid-cert:8080"


class TestValidationErrors:
    """Test cases for validation-related errors."""

    def test_validation_error_basic(self):
        """Test basic ValidationError creation."""
        error = ValidationError("Validation failed")

        assert isinstance(error, KrakenError)
        assert error.message == "Validation failed"
        assert error.validation_errors == []
        assert error.invalid_data is None

    def test_validation_error_with_details(self):
        """Test ValidationError with full details."""
        validation_errors = [
            {"field": "name", "error": "required"},
            {"field": "age", "error": "must be positive"},
        ]
        invalid_data = {"age": -5}
        schema_info = {"type": "Person", "version": "1.0"}

        error = ValidationError(
            "Multiple validation errors",
            validation_errors=validation_errors,
            invalid_data=invalid_data,
            schema_info=schema_info,
        )

        assert error.validation_errors == validation_errors
        assert error.invalid_data == invalid_data
        assert error.schema_info == schema_info

    def test_pydantic_validation_error(self):
        """Test PydanticValidationError creation."""
        field_errors = {
            "name": ["field required"],
            "age": ["ensure this value is greater than 0"],
        }

        error = PydanticValidationError(
            "Pydantic validation failed",
            model_name="PersonModel",
            field_errors=field_errors,
        )

        assert isinstance(error, ValidationError)
        assert error.model_name == "PersonModel"
        assert error.field_errors == field_errors
        assert error.get_context("model_name") == "PersonModel"

    def test_json_validation_error(self):
        """Test JSONValidationError creation."""
        error = JSONValidationError(
            "Invalid JSON",
            json_data='{"invalid": json}',
            parse_error="Expecting ',' delimiter",
        )

        assert isinstance(error, ValidationError)
        assert error.json_data == '{"invalid": json}'
        assert error.parse_error == "Expecting ',' delimiter"
        assert error.get_context("parse_error") == "Expecting ',' delimiter"

    def test_schema_validation_error(self):
        """Test SchemaValidationError creation."""
        expected_schema = {"type": "object",
                           "properties": {"name": {"type": "string"}}}
        actual_data = {"name": 123}

        error = SchemaValidationError(
            "Schema mismatch",
            expected_schema=expected_schema,
            actual_data=actual_data,
            schema_path="$.name",
        )

        assert isinstance(error, ValidationError)
        assert error.expected_schema == expected_schema
        assert error.actual_data == actual_data
        assert error.schema_path == "$.name"

    def test_parameter_validation_error(self):
        """Test ParameterValidationError creation."""
        error = ParameterValidationError(
            "Invalid parameter value",
            parameter_name="temperature",
            parameter_value=3.0,
            valid_values=[0.0, 1.0, 2.0],
        )

        assert isinstance(error, ValidationError)
        assert error.parameter_name == "temperature"
        assert error.parameter_value == 3.0
        assert error.valid_values == [0.0, 1.0, 2.0]


class TestAPIErrors:
    """Test cases for API-related errors."""

    def test_api_error_basic(self):
        """Test basic APIError creation."""
        error = APIError("API error occurred")

        assert isinstance(error, KrakenError)
        assert error.message == "API error occurred"
        assert error.status_code is None
        assert error.error_code is None

    def test_api_error_with_details(self):
        """Test APIError with full details."""
        response_data = {
            "error": {"message": "Invalid request", "type": "invalid_request_error"}}

        error = APIError(
            "API request failed",
            status_code=400,
            error_code="invalid_request",
            error_type="invalid_request_error",
            response_data=response_data,
            request_id="req_123456",
        )

        assert error.status_code == 400
        assert error.error_code == "invalid_request"
        assert error.error_type == "invalid_request_error"
        assert error.response_data == response_data
        assert error.request_id == "req_123456"

    def test_authentication_error(self):
        """Test AuthenticationError creation."""
        error = AuthenticationError(auth_type="api_key")

        assert isinstance(error, APIError)
        assert error.status_code == 401
        assert error.auth_type == "api_key"
        assert "Authentication failed" in error.message

    def test_rate_limit_error(self):
        """Test RateLimitError creation."""
        error = RateLimitError(
            limit_type="requests_per_minute",
            limit_value=60,
            reset_time=1640995200,
            retry_after=30,
        )

        assert isinstance(error, APIError)
        assert error.status_code == 429
        assert error.limit_type == "requests_per_minute"
        assert error.limit_value == 60
        assert error.reset_time == 1640995200
        assert error.retry_after == 30

    def test_model_error(self):
        """Test ModelError creation."""
        available_models = ["gpt-3.5-turbo", "gpt-4"]

        error = ModelError(
            "Model not found",
            model_name="gpt-5",
            available_models=available_models,
        )

        assert isinstance(error, APIError)
        assert error.model_name == "gpt-5"
        assert error.available_models == available_models

    def test_content_filter_error(self):
        """Test ContentFilterError creation."""
        error = ContentFilterError(
            filter_type="safety",
            filtered_content="inappropriate content",
        )

        assert isinstance(error, APIError)
        assert error.filter_type == "safety"
        assert error.filtered_content == "inappropriate content"

    def test_service_unavailable_error(self):
        """Test ServiceUnavailableError creation."""
        error = ServiceUnavailableError(
            service_status="maintenance",
            estimated_recovery="2024-01-01T12:00:00Z",
        )

        assert isinstance(error, APIError)
        assert error.status_code == 503
        assert error.service_status == "maintenance"
        assert error.estimated_recovery == "2024-01-01T12:00:00Z"


class TestErrorInheritance:
    """Test cases for error inheritance hierarchy."""

    def test_inheritance_chain(self):
        """Test that all errors inherit from KrakenError."""
        errors = [
            NetworkError("test"),
            ConnectionError(),
            TimeoutError(),
            HTTPError("test", 500),
            SSLError(),
            ValidationError("test"),
            PydanticValidationError("test"),
            JSONValidationError("test"),
            SchemaValidationError("test"),
            ParameterValidationError("test"),
            APIError("test"),
            AuthenticationError(),
            RateLimitError(),
            ModelError("test"),
            ContentFilterError(),
            ServiceUnavailableError(),
        ]

        for error in errors:
            assert isinstance(error, KrakenError)
            assert isinstance(error, Exception)

    def test_network_error_inheritance(self):
        """Test NetworkError inheritance chain."""
        network_errors = [
            ConnectionError(),
            TimeoutError(),
            HTTPError("test", 500),
            SSLError(),
        ]

        for error in network_errors:
            assert isinstance(error, NetworkError)
            assert isinstance(error, KrakenError)

    def test_validation_error_inheritance(self):
        """Test ValidationError inheritance chain."""
        validation_errors = [
            PydanticValidationError("test"),
            JSONValidationError("test"),
            SchemaValidationError("test"),
            ParameterValidationError("test"),
        ]

        for error in validation_errors:
            assert isinstance(error, ValidationError)
            assert isinstance(error, KrakenError)

    def test_api_error_inheritance(self):
        """Test APIError inheritance chain."""
        api_errors = [
            AuthenticationError(),
            RateLimitError(),
            ModelError("test"),
            ContentFilterError(),
            ServiceUnavailableError(),
        ]

        for error in api_errors:
            assert isinstance(error, APIError)
            assert isinstance(error, KrakenError)
