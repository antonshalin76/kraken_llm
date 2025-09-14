"""
Embeddings LLM клиент для Kraken фреймворка.

Этот модуль содержит EmbeddingsLLMClient - специализированную реализацию LLM клиента
для получения векторных представлений (embeddings) текста.
"""

from typing import List, Union, Optional
import numpy as np

from .base import BaseLLMClient
from ..exceptions.api import APIError
from ..exceptions.validation import ValidationError
from ..utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingsLLMClient(BaseLLMClient):
    """
    LLM клиент для получения embeddings.
    
    Реализует полную поддержку embeddings через:
    - OpenAI-совместимый API для получения векторных представлений
    - Поддержку как одиночных текстов, так и батчей
    - Автоматическое построение правильных URL'ов через систему API путей
    - Обработку различных форматов входных данных
    """
    
    async def get_embeddings(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> List[List[float]]:
        """
        Получает embeddings для текстов (алиас для create_embeddings).
        
        Args:
            texts: Текст или список текстов
            **kwargs: Дополнительные параметры
            
        Returns:
            Список векторов embeddings
        """
        result = await self.create_embeddings(texts, **kwargs)
        return [embedding["embedding"] for embedding in result["data"]]

    # Реализация абстрактных методов (не используются для embeddings)
    async def chat_completion(self, *args, **kwargs):
        """Не поддерживается для embeddings клиента."""
        raise NotImplementedError("EmbeddingsLLMClient не поддерживает chat_completion. Используйте create_embeddings.")
    
    async def chat_completion_stream(self, *args, **kwargs):
        """Не поддерживается для embeddings клиента."""
        raise NotImplementedError("EmbeddingsLLMClient не поддерживает chat_completion_stream. Используйте create_embeddings.")
    
    async def chat_completion_structured(self, *args, **kwargs):
        """Не поддерживается для embeddings клиента."""
        raise NotImplementedError("EmbeddingsLLMClient не поддерживает chat_completion_structured. Используйте create_embeddings.")
    
    def __init__(self, config):
        """
        Инициализация embeddings LLM клиента.
        
        Args:
            config: Конфигурация клиента
        """
        super().__init__(config)
        
        logger.info(
            f"EmbeddingsLLMClient инициализирован с моделью: {config.model}, "
            f"endpoint: {config.endpoint}, api_mode: {config.api_mode}"
        )
    
    async def create_embeddings(
        self,
        input_texts: Union[str, List[str]],
        model: Optional[str] = None,
        encoding_format: str = "float",
        dimensions: Optional[int] = None,
        user: Optional[str] = None,
        **kwargs
    ) -> dict:
        """
        Создание embeddings для текста или списка текстов.
        
        Args:
            input_texts: Текст или список текстов для получения embeddings
            model: Переопределение модели (опционально)
            encoding_format: Формат кодирования ("float" или "base64")
            dimensions: Количество измерений (для моделей, поддерживающих это)
            user: Идентификатор пользователя (опционально)
            **kwargs: Дополнительные параметры
            
        Returns:
            Словарь с embeddings в формате OpenAI API
            
        Raises:
            ValidationError: При ошибках валидации входных данных
            APIError: При ошибках API
            NetworkError: При сетевых ошибках
        """
        logger.info(
            f"Создание embeddings для {len(input_texts) if isinstance(input_texts, list) else 1} текстов"
        )
        
        # Валидация входных параметров
        if not input_texts:
            raise ValidationError("input_texts не может быть пустым")
        
        # Нормализация входных данных
        if isinstance(input_texts, str):
            texts = [input_texts]
        else:
            texts = input_texts
        
        # Валидация текстов
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValidationError(f"Текст #{i} должен быть строкой, получен: {type(text)}")
            if not text.strip():
                raise ValidationError(f"Текст #{i} не может быть пустым")
        
        try:
            # Подготовка параметров для API
            params = {
                "input": texts,
                "model": model or self.config.model,
                "encoding_format": encoding_format,
            }
            
            # Добавляем опциональные параметры
            if dimensions is not None:
                params["dimensions"] = dimensions
            if user is not None:
                params["user"] = user
            
            # Добавляем дополнительные параметры
            params.update(kwargs)
            
            logger.debug(f"Параметры embeddings API: {list(params.keys())}")
            
            # Выполнение запроса через AsyncOpenAI
            response = await self.openai_client.embeddings.create(**params)
            
            logger.debug(f"Получен ответ embeddings API: {len(response.data)} векторов")
            
            # Преобразуем ответ в словарь для совместимости
            result = {
                "object": response.object,
                "data": [
                    {
                        "object": item.object,
                        "index": item.index,
                        "embedding": item.embedding,
                    }
                    for item in response.data
                ],
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "total_tokens": response.usage.total_tokens,
                } if response.usage else None,
            }
            
            logger.info(f"Embeddings успешно созданы для {len(result['data'])} текстов")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при создании embeddings: {e}")
            # Обрабатываем ошибки OpenAI API
            await self._handle_openai_error(e)
    
    async def get_embeddings_vectors(
        self,
        input_texts: Union[str, List[str]],
        model: Optional[str] = None,
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """
        Получение только векторов embeddings (упрощенный интерфейс).
        
        Args:
            input_texts: Текст или список текстов для получения embeddings
            model: Переопределение модели (опционально)
            **kwargs: Дополнительные параметры
            
        Returns:
            Список чисел (для одного текста) или список списков чисел (для нескольких текстов)
        """
        response = await self.create_embeddings(input_texts, model=model, **kwargs)
        
        vectors = [item["embedding"] for item in response["data"]]
        
        # Если был передан один текст, возвращаем один вектор
        if isinstance(input_texts, str):
            return vectors[0]
        
        return vectors
    
    async def get_embeddings_numpy(
        self,
        input_texts: Union[str, List[str]],
        model: Optional[str] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Получение embeddings в формате numpy array.
        
        Args:
            input_texts: Текст или список текстов для получения embeddings
            model: Переопределение модели (опционально)
            **kwargs: Дополнительные параметры
            
        Returns:
            numpy.ndarray с embeddings
        """
        vectors = await self.get_embeddings_vectors(input_texts, model=model, **kwargs)
        
        # Преобразуем в numpy array
        if isinstance(input_texts, str):
            # Один вектор
            return np.array(vectors)
        else:
            # Матрица векторов
            return np.array(vectors)
    
    def cosine_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """
        Вычисление косинусного сходства между двумя векторами.
        
        Args:
            vector1: Первый вектор
            vector2: Второй вектор
            
        Returns:
            Косинусное сходство (от -1 до 1)
        """
        # Преобразуем в numpy arrays
        v1 = np.array(vector1)
        v2 = np.array(vector2)
        
        # Вычисляем косинусное сходство
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        
        return dot_product / (norm_v1 * norm_v2)
    
    async def similarity_search(
        self,
        query_text: str,
        candidate_texts: List[str],
        model: Optional[str] = None,
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[dict]:
        """
        Поиск наиболее похожих текстов на основе embeddings.
        
        Args:
            query_text: Текст запроса
            candidate_texts: Список текстов-кандидатов
            model: Переопределение модели (опционально)
            top_k: Количество лучших результатов (по умолчанию все)
            **kwargs: Дополнительные параметры
            
        Returns:
            Список словарей с результатами поиска, отсортированный по убыванию сходства
        """
        logger.info(f"Поиск сходства для запроса среди {len(candidate_texts)} кандидатов")
        
        # Получаем embeddings для всех текстов
        all_texts = [query_text] + candidate_texts
        embeddings = await self.get_embeddings_vectors(all_texts, model=model, **kwargs)
        
        query_embedding = embeddings[0]
        candidate_embeddings = embeddings[1:]
        
        # Вычисляем сходство с каждым кандидатом
        results = []
        for i, candidate_embedding in enumerate(candidate_embeddings):
            similarity = self.cosine_similarity(query_embedding, candidate_embedding)
            results.append({
                "text": candidate_texts[i],
                "similarity": similarity,
                "index": i,
            })
        
        # Сортируем по убыванию сходства
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Ограничиваем количество результатов
        if top_k is not None:
            results = results[:top_k]
        
        logger.info(f"Найдено {len(results)} результатов поиска")
        return results