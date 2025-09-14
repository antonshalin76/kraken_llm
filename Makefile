# Makefile for Kraken LLM Framework

.PHONY: help install install-dev install-minimal test lint format clean docs

# Переменные
PYTHON := python3
PIP := pip3
PYTEST := python3 -m pytest
BLACK := python3 -m black
ISORT := python3 -m isort
MYPY := python3 -m mypy
FLAKE8 := python3 -m flake8

# Цвета для вывода
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Показать справку
	@echo "$(GREEN)Kraken LLM Framework - Команды разработки$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Установить основные зависимости
	@echo "$(GREEN)Установка основных зависимостей...$(NC)"
	$(PIP) install -r requirements.txt

install-dev: ## Установить зависимости для разработки
	@echo "$(GREEN)Установка зависимостей для разработки...$(NC)"
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .

install-minimal: ## Установить минимальные зависимости
	@echo "$(GREEN)Установка минимальных зависимостей...$(NC)"
	$(PIP) install -r requirements-minimal.txt

test: ## Запустить тесты
	@echo "$(GREEN)Запуск тестов...$(NC)"
	$(PYTEST) tests/ -v

test-cov: ## Запустить тесты с покрытием кода
	@echo "$(GREEN)Запуск тестов с анализом покрытия...$(NC)"
	$(PYTEST) tests/ -v --cov=src --cov-report=html --cov-report=term

test-unit: ## Запустить только unit тесты
	@echo "$(GREEN)Запуск unit тестов...$(NC)"
	$(PYTEST) tests/unit/ -v

test-integration: ## Запустить только интеграционные тесты
	@echo "$(GREEN)Запуск интеграционных тестов...$(NC)"
	$(PYTEST) tests/integration/ -v

lint: ## Проверить код линтерами
	@echo "$(GREEN)Проверка кода линтерами...$(NC)"
	$(FLAKE8) src/ tests/ examples/
	$(MYPY) src/

format: ## Форматировать код
	@echo "$(GREEN)Форматирование кода...$(NC)"
	$(BLACK) src/ tests/ examples/
	$(ISORT) src/ tests/ examples/

format-check: ## Проверить форматирование без изменений
	@echo "$(GREEN)Проверка форматирования...$(NC)"
	$(BLACK) --check src/ tests/ examples/
	$(ISORT) --check-only src/ tests/ examples/

clean: ## Очистить временные файлы
	@echo "$(GREEN)Очистка временных файлов...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

docs: ## Показать доступную документацию
	@echo "$(GREEN)Доступная документация:$(NC)"
	@echo "$(YELLOW)  - README.md - Полная документация$(NC)"
	@echo "$(YELLOW)  - QUICKSTART.md - Быстрый старт$(NC)"
	@echo "$(YELLOW)  - examples/ - Примеры использования$(NC)"

examples: ## Запустить примеры
	@echo "$(GREEN)Запуск примеров...$(NC)"
	@echo "$(YELLOW)Доступные примеры:$(NC)"
	@ls examples/*.py | sed 's/examples\//  - /'

run-simple: ## Запустить простой пример
	@echo "$(GREEN)Запуск простого примера...$(NC)"
	python3 examples/simple_example.py

run-streaming: ## Запустить пример потоковой передачи
	@echo "$(GREEN)Запуск примера потоковой передачи...$(NC)"
	python3 examples/streaming_example.py

run-structured: ## Запустить пример структурированного вывода
	@echo "$(GREEN)Запуск примера структурированного вывода...$(NC)"
	python3 examples/so_mode_switching_example.py

run-capabilities: ## Запустить утилиту определения возможностей моделей
	@echo "$(GREEN)Запуск утилиты определения возможностей моделей...$(NC)"
	python3 model_capabilities_detector.py

run-analyzer: ## Запустить новый анализатор возможностей моделей
	@echo "$(GREEN)Запуск анализатора возможностей моделей...$(NC)"
	python3 model_capabilities_analyzer.py

run-analyzer-quick: ## Быстрый анализ возможностей моделей
	@echo "$(GREEN)Быстрый анализ возможностей моделей...$(NC)"
	python3 model_capabilities_analyzer.py --quick

run-analyzer-md: ## Анализ с Markdown отчетом
	@echo "$(GREEN)Анализ с Markdown отчетом...$(NC)"
	python3 model_capabilities_analyzer.py --output markdown

run-media-utils: ## Запустить пример утилит для работы с медиа
	@echo "$(GREEN)Запуск примера утилит для работы с медиа...$(NC)"
	python3 examples/media_utils_example.py

run-reasoning: ## Запустить пример рассуждающих моделей
	@echo "$(GREEN)Запуск примера рассуждающих моделей...$(NC)"
	python3 examples/native_thinking_example.py

run-factory: ## Запустить пример фабрики клиентов
	@echo "$(GREEN)Запуск примера фабрики клиентов...$(NC)"
	python3 examples/client_factory_example.py

run-multimodal: ## Запустить пример мультимодального клиента
	@echo "$(GREEN)Запуск примера мультимодального клиента...$(NC)"
	python3 examples/multimodal_example.py

run-tools: ## Запустить пример функций и инструментов
	@echo "$(GREEN)Запуск примера функций и инструментов...$(NC)"
	python3 examples/function_tool_example.py

run-universal: ## Запустить пример универсального клиента
	@echo "$(GREEN)Запуск примера универсального клиента...$(NC)"
	python3 examples/universal_client_example.py

run-universal-quick: ## Быстрая демонстрация универсального клиента
	@echo "$(GREEN)Быстрая демонстрация универсального клиента...$(NC)"
	python3 examples/quick_universal_example.py

run-universal-comprehensive: ## Полная демонстрация универсального клиента
	@echo "$(GREEN)Полная демонстрация универсального клиента...$(NC)"
	python3 examples/universal_client_comprehensive.py

test-universal-integration: ## Тест интеграции универсального клиента
	@echo "$(GREEN)Тестирование интеграции универсального клиента...$(NC)"
	python3 tests/integration/test_universal_integration.py

capabilities-report: ## Создать отчет о возможностях моделей (legacy)
	@echo "$(GREEN)Создание отчета о возможностях моделей (legacy)...$(NC)"
	python3 model_capabilities_detector.py
	@echo "$(YELLOW)Отчет сохранен в model_capabilities_report.json$(NC)"

capabilities-quick: ## Быстрый тест возможностей моделей
	@echo "$(GREEN)Быстрое тестирование возможностей моделей...$(NC)"
	python3 quick_capabilities_test.py

capabilities-analyze: ## Полный анализ возможностей моделей (новая утилита)
	@echo "$(GREEN)Полный анализ возможностей моделей...$(NC)"
	python3 model_capabilities_analyzer.py
	@echo "$(YELLOW)Отчет сохранен в model_capabilities_report_*.json$(NC)"

capabilities-analyze-quick: ## Быстрый анализ возможностей моделей (новая утилита)
	@echo "$(GREEN)Быстрый анализ возможностей моделей...$(NC)"
	python3 model_capabilities_analyzer.py --quick

capabilities-analyze-md: ## Анализ с Markdown отчетом
	@echo "$(GREEN)Анализ с Markdown отчетом...$(NC)"
	python3 model_capabilities_analyzer.py --output markdown
	@echo "$(YELLOW)Отчет сохранен в model_capabilities_report_*.md$(NC)"



check: format-check lint test ## Полная проверка кода (форматирование + линтинг + тесты)

ci: install-dev check ## Команда для CI/CD

build: clean ## Собрать пакет для распространения
	@echo "$(GREEN)Сборка пакета...$(NC)"
	python3 setup.py sdist bdist_wheel

upload-test: build ## Загрузить пакет в Test PyPI
	@echo "$(GREEN)Загрузка в Test PyPI...$(NC)"
	twine upload --repository testpypi dist/*

upload: build ## Загрузить пакет в PyPI
	@echo "$(GREEN)Загрузка в PyPI...$(NC)"
	twine upload dist/*

venv: ## Создать виртуальное окружение
	@echo "$(GREEN)Создание виртуального окружения...$(NC)"
	python3 -m venv venv
	@echo "$(YELLOW)Активируйте окружение: source venv/bin/activate$(NC)"

requirements: ## Обновить requirements.txt из pyproject.toml
	@echo "$(GREEN)Обновление requirements.txt...$(NC)"
	pip-compile pyproject.toml

dev-setup: venv install-dev ## Полная настройка среды разработки
	@echo "$(GREEN)Среда разработки настроена!$(NC)"
	@echo "$(YELLOW)Не забудьте активировать окружение: source venv/bin/activate$(NC)"

# Алиасы для удобства
i: install
id: install-dev
t: test
tc: test-cov
f: format
l: lint
c: clean