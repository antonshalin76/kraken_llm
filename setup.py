#!/usr/bin/env python3
"""
Setup script for Kraken LLM Framework
"""

from setuptools import setup, find_packages
from pathlib import Path

# Читаем README для long_description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Основные зависимости
install_requires = [
    "openai>=1.0.0,<2.0.0",
    "outlines>=0.0.30,<1.0.0", 
    "pydantic>=2.0.0,<3.0.0",
    "pydantic-settings>=2.0.0,<3.0.0",
    "httpx>=0.25.0,<1.0.0",
    "Pillow>=10.0.0,<11.0.0",
    "numpy>=1.24.0,<2.0.0",
    "python-dotenv>=1.0.0,<2.0.0",
    "typing-extensions>=4.5.0",
]

# Дополнительные зависимости
extras_require = {
    "demo": [
        "fastapi>=0.100.0,<1.0.0",
        "uvicorn>=0.23.0,<1.0.0",
    ],
    "dev": [
        "pytest>=7.0.0,<8.0.0",
        "pytest-asyncio>=0.21.0,<1.0.0",
        "pytest-mock>=3.11.0,<4.0.0",
        "pytest-cov>=4.1.0,<5.0.0",
        "coverage>=7.0.0,<8.0.0",
        "black>=23.0.0,<24.0.0",
        "isort>=5.12.0,<6.0.0",
        "mypy>=1.5.0,<2.0.0",
        "flake8>=6.0.0,<7.0.0",
    ],
    "docs": [
        "sphinx>=7.0.0,<8.0.0",
        "sphinx-rtd-theme>=1.3.0,<2.0.0",
    ],
    "jupyter": [
        "jupyter>=1.0.0,<2.0.0",
        "ipython>=8.0.0,<9.0.0",
    ],
}

# Добавляем 'all' для установки всех дополнительных зависимостей
extras_require["all"] = [
    dep for deps_list in extras_require.values() for dep in deps_list
]

setup(
    name="kraken-llm",
    version="0.1.0",
    description="Universal LLM framework with full OpenAI API support and unified multi-client interface",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Anton Shalin",
    author_email="ashalin@it-one.ru",
    url="https://github.com/example/kraken-llm",
    project_urls={
        "Documentation": "https://kraken-llm.readthedocs.io",
        "Repository": "https://github.com/example/kraken-llm",
        "Issues": "https://github.com/example/kraken-llm/issues",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "llm", "openai", "ai", "machine-learning", "async", "streaming",
        "pydantic", "outlines", "embeddings", "multimodal", "asr", "tts",
        "universal-client", "auto-fallback", "structured-output", "reasoning"
    ],
    include_package_data=True,
    zip_safe=False,
)