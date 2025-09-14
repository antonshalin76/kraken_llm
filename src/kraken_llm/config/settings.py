"""
Конфигурация Kraken LLM с использованием Pydantic Settings

Этот модуль обеспечивает комплексное управление конфигурацией с:
- Поддержкой переменных окружения
- Загрузкой .env файлов
- Валидацией типов
- Значениями по умолчанию с описаниями
- Подсказками IDE и документацией
"""

from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .defaults import (
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    DEFAULT_API_KEY,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TOP_P,
    DEFAULT_FREQUENCY_PENALTY,
    DEFAULT_PRESENCE_PENALTY,
    DEFAULT_STOP,
    DEFAULT_STREAM,
    DEFAULT_OUTLINES_SO_MODE,
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_READ_TIMEOUT,
    DEFAULT_WRITE_TIMEOUT,
    DEFAULT_SSL_VERIFY,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY,
    DEFAULT_LOG_LEVEL,
    DEFAULT_LOG_FORMAT,
    DEFAULT_API_MODE,
    DEFAULT_API_VERSION,
    DEFAULT_CHAT_COMPLETIONS_PATH,
    DEFAULT_COMPLETIONS_PATH,
    DEFAULT_EMBEDDINGS_PATH,
    DEFAULT_MODELS_PATH,
)


class LLMConfig(BaseSettings):
    """
    Конфигурация для клиентов Kraken LLM.
    
    Все параметры имеют разумные значения по умолчанию и могут быть переопределены через:
    - Переменные окружения (с префиксом LLM_)
    - .env файл
    - Прямую передачу параметров
    
    Пример:
        # Использование значений по умолчанию
        config = LLMConfig()
        
        # Переопределение конкретных значений
        config = LLMConfig(temperature=0.9, max_tokens=2000)
        
        # Загрузка из окружения
        # LLM_ENDPOINT=http://localhost:8080 LLM_TEMPERATURE=0.5
        config = LLMConfig()
    """
    
    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Настройки подключения
    endpoint: str = Field(
        default=DEFAULT_ENDPOINT,
        description="URL эндпоинта LLM API. Базовый URL, где размещен ваш LLM API.",
        examples=["http://localhost:8080", "https://api.openai.com/v1"],
    )
    
    api_key: Optional[str] = Field(
        default=DEFAULT_API_KEY,
        description="API ключ для аутентификации. Оставьте None, если аутентификация не требуется.",
        examples=["sk-...", "auth_token_123"],
    )
    
    token: Optional[str] = Field(
        default=None,
        description="Альтернативное поле для API токена (сопоставляется с api_key)",
        examples=["auth_token_123"],
    )
    
    model: str = Field(
        default=DEFAULT_MODEL,
        description="Имя модели для использования в завершениях. Должно соответствовать доступной модели на вашем эндпоинте.",
        examples=["chat", "embedding", "multimodal", "thinking", "asr"],
    )
    
    # Параметры генерации
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description=(
            "Температура сэмплирования (0.0 до 2.0). Более высокие значения делают вывод более случайным, "
            "более низкие значения делают его более сфокусированным и детерминированным. 0.0 = детерминированный."
        ),
    )
    
    max_tokens: int = Field(
        default=DEFAULT_MAX_TOKENS,
        gt=0,
        le=100000,
        description=(
            "Максимальное количество токенов для генерации. Контролирует длину ответа. "
            "Примечание: Для некоторых моделей это включает как входные, так и выходные токены."
        ),
    )
    
    top_p: float = Field(
        default=DEFAULT_TOP_P,
        ge=0.0,
        le=1.0,
        description=(
            "Параметр nucleus sampling (0.0 до 1.0). Альтернатива температуре. "
            "0.1 означает, что рассматриваются только токены, составляющие топ 10% вероятностной массы."
        ),
    )
    
    frequency_penalty: float = Field(
        default=DEFAULT_FREQUENCY_PENALTY,
        ge=-2.0,
        le=2.0,
        description=(
            "Штраф за частоту (-2.0 до 2.0). Положительные значения штрафуют токены на основе "
            "их частоты в тексте до сих пор, уменьшая вероятность повторения."
        ),
    )
    
    presence_penalty: float = Field(
        default=DEFAULT_PRESENCE_PENALTY,
        ge=-2.0,
        le=2.0,
        description=(
            "Штраф за присутствие (-2.0 до 2.0). Положительные значения штрафуют токены, которые "
            "появились в тексте до сих пор, побуждая модель говорить о новых темах."
        ),
    )
    
    stop: Optional[List[str]] = Field(
        default=DEFAULT_STOP,
        description=(
            "Стоп-последовательности. Список строк, при встрече которых генерация остановится. "
            "Модель прекратит генерацию при встрече любой из этих последовательностей."
        ),
        examples=[["\\n", "END"], ["Human:", "AI:"]],
    )
    
    # Настройки поведения
    stream: bool = Field(
        default=DEFAULT_STREAM,
        description=(
            "Включить потоковые ответы. Когда True, ответы возвращаются по мере их "
            "генерации, а не ожидания завершения. Полезно для UI в реальном времени."
        ),
    )
    
    outlines_so_mode: bool = Field(
        default=DEFAULT_OUTLINES_SO_MODE,
        description=(
            "Включить режим структурированного вывода Outlines. Когда True, использует библиотеку Outlines "
            "для генерации структурированного вывода. Когда False, использует нативный response_format OpenAI. "
            "Outlines обеспечивает лучшее соответствие схеме, но может быть медленнее."
        ),
    )
    
    # Конфигурация OpenAI API
    api_mode: str = Field(
        default=DEFAULT_API_MODE,
        description=(
            "Режим совместимости API. Варианты: 'openai_compatible' (стандартные пути OpenAI API), "
            "'custom' (использовать пользовательские пути), 'direct' (использовать эндпоинт как есть без дополнительных путей)."
        ),
    )
    
    api_version: str = Field(
        default=DEFAULT_API_VERSION,
        description="Версия OpenAI API для использования (например, 'v1')",
    )
    
    chat_completions_path: str = Field(
        default=DEFAULT_CHAT_COMPLETIONS_PATH,
        description="Путь для эндпоинта chat completions (например, '/v1/chat/completions')",
    )
    
    completions_path: str = Field(
        default=DEFAULT_COMPLETIONS_PATH,
        description="Путь для эндпоинта completions (например, '/v1/completions')",
    )
    
    embeddings_path: str = Field(
        default=DEFAULT_EMBEDDINGS_PATH,
        description="Путь для эндпоинта embeddings (например, '/v1/embeddings')",
    )
    
    models_path: str = Field(
        default=DEFAULT_MODELS_PATH,
        description="Путь для эндпоинта models (например, '/v1/models')",
    )
    
    # Настройки таймаутов
    connect_timeout: float = Field(
        default=DEFAULT_CONNECT_TIMEOUT,
        gt=0,
        description=(
            "Таймаут подключения в секундах. Максимальное время ожидания при установлении "
            "соединения с эндпоинтом LLM API."
        ),
    )
    
    read_timeout: float = Field(
        default=DEFAULT_READ_TIMEOUT,
        gt=0,
        description=(
            "Таймаут чтения в секундах. Максимальное время ожидания ответа от "
            "LLM API после отправки запроса. Должен быть выше для более длинных генераций."
        ),
    )
    
    write_timeout: float = Field(
        default=DEFAULT_WRITE_TIMEOUT,
        gt=0,
        description=(
            "Таймаут записи в секундах. Максимальное время ожидания при отправке данных "
            "на эндпоинт LLM API."
        ),
    )
    
    # Настройки SSL
    ssl_verify: bool = Field(
        default=DEFAULT_SSL_VERIFY,
        description=(
            "Проверять SSL сертификаты. Устанавливайте False только для разработки с "
            "самоподписанными сертификатами. Всегда используйте True в продакшене."
        ),
    )
    
    # Настройки повторных попыток
    max_retries: int = Field(
        default=DEFAULT_MAX_RETRIES,
        ge=0,
        le=10,
        description=(
            "Максимальное количество попыток повтора для неудачных запросов. "
            "0 означает отсутствие повторов, более высокие значения увеличивают надежность, но задержку."
        ),
    )
    
    retry_delay: float = Field(
        default=DEFAULT_RETRY_DELAY,
        ge=0.0,
        description=(
            "Задержка между попытками повтора в секундах. Применяется экспоненциальная задержка, "
            "поэтому фактические задержки будут: delay, delay*2, delay*4, и т.д."
        ),
    )
    
    # Настройки логирования
    log_level: str = Field(
        default=DEFAULT_LOG_LEVEL,
        description=(
            "Уровень логирования для операций Kraken. Контролирует подробность вывода логов. "
            "DEBUG показывает все операции, INFO показывает важные события, WARN показывает проблемы."
        ),
        examples=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    
    log_format: str = Field(
        default=DEFAULT_LOG_FORMAT,
        description=(
            "Строка формата сообщений лога. Использует спецификаторы формата Python logging. "
            "Настройте в соответствии со стилем логирования вашего приложения."
        ),
    )
    
    def model_post_init(self, __context) -> None:
        """Пост-инициализационная валидация и настройка."""
        # Убеждаемся, что эндпоинт не заканчивается слешем для согласованного построения URL
        if self.endpoint.endswith("/"):
            self.endpoint = self.endpoint.rstrip("/")
        
        # Используем token как api_key, если api_key не установлен, но token есть
        if not self.api_key and self.token:
            self.api_key = self.token
    
    @property
    def timeout_config(self) -> dict:
        """Получить конфигурацию таймаутов для HTTP клиентов."""
        return {
            "connect": self.connect_timeout,
            "read": self.read_timeout,
            "write": self.write_timeout,
        }
    
    @property
    def generation_params(self) -> dict:
        """Получить параметры генерации для LLM запросов."""
        params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }
        
        if self.stop is not None:
            params["stop"] = self.stop
            
        return params
    
    def to_openai_params(self, **overrides) -> dict:
        """
        Преобразовать конфигурацию в параметры OpenAI API.
        
        Args:
            **overrides: Переопределения параметров для этого конкретного запроса
            
        Returns:
            Словарь параметров, подходящих для вызовов OpenAI API
        """
        params = {
            "model": self.model,
            **self.generation_params,
            **overrides,
        }
        
        # Удаляем None значения
        return {k: v for k, v in params.items() if v is not None}
    
    def __repr__(self) -> str:
        """Строковое представление, скрывающее чувствительную информацию."""
        api_key_display = "***" if self.api_key else None
        return (
            f"LLMConfig("
            f"endpoint='{self.endpoint}', "
            f"model='{self.model}', "
            f"api_key={api_key_display}, "
            f"temperature={self.temperature}, "
            f"max_tokens={self.max_tokens}"
            f")"
        )