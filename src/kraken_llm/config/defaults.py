"""
Значения конфигурации по умолчанию для Kraken LLM Framework

Этот модуль содержит все значения по умолчанию, используемые во всем фреймворке.
"""

from typing import List, Optional

# Значения по умолчанию для подключения
DEFAULT_ENDPOINT = "http://localhost:8080"
DEFAULT_MODEL = "chat"
DEFAULT_API_KEY: Optional[str] = None

# Значения по умолчанию для параметров генерации
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1000
DEFAULT_TOP_P = 1.0
DEFAULT_FREQUENCY_PENALTY = 0.0
DEFAULT_PRESENCE_PENALTY = 0.0
DEFAULT_STOP: Optional[List[str]] = None

# Значения по умолчанию для поведения
DEFAULT_STREAM = False
DEFAULT_OUTLINES_SO_MODE = False

# Значения по умолчанию для таймаутов (в секундах)
DEFAULT_CONNECT_TIMEOUT = 10.0
DEFAULT_READ_TIMEOUT = 60.0
DEFAULT_WRITE_TIMEOUT = 10.0

# Значения по умолчанию для SSL
DEFAULT_SSL_VERIFY = True

# Значения по умолчанию для повторных попыток
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0

# Значения по умолчанию для логирования
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Значения по умолчанию для путей OpenAI API
DEFAULT_API_VERSION = "v1"
DEFAULT_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
DEFAULT_COMPLETIONS_PATH = "/v1/completions"
DEFAULT_EMBEDDINGS_PATH = "/v1/embeddings"
DEFAULT_MODELS_PATH = "/v1/models"

# Значения по умолчанию для режима API
DEFAULT_API_MODE = "openai_compatible"  # openai_compatible, custom, direct