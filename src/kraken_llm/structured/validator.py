"""
Валидатор для structured output в Kraken фреймворке.

Этот модуль содержит утилиты для валидации структурированных ответов
от языковых моделей через Pydantic модели, включая обработку ошибок
валидации, проверку совместимости схем и дополнительные утилиты.
"""

import json
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, ValidationError as PydanticValidationError

from ..exceptions.validation import ValidationError
from ..utils.logging import get_logger

logger = get_logger(__name__)


class StructuredOutputValidator:
    """
    Валидатор для structured output с поддержкой Pydantic моделей.
    
    Предоставляет методы для валидации JSON ответов по Pydantic схемам,
    проверки совместимости моделей, обработки ошибок валидации и
    дополнительные утилиты для работы со структурированными данными.
    """
    
    def __init__(self):
        """Инициализация валидатора."""
        logger.debug("Инициализация StructuredOutputValidator")
    
    async def validate_response(
        self,
        response: Union[str, Dict[str, Any], Any],
        model: Type[BaseModel],
        strict: bool = True
    ) -> BaseModel:
        """
        Валидация ответа по Pydantic модели.
        
        Args:
            response: Ответ для валидации (JSON строка, словарь или объект)
            model: Pydantic модель для валидации
            strict: Строгий режим валидации (по умолчанию True)
            
        Returns:
            Валидированный Pydantic объект
            
        Raises:
            ValidationError: При ошибках валидации с детальной информацией
        """
        logger.debug(f"Валидация ответа по модели {model.__name__}")
        
        try:
            # Если ответ уже является экземпляром нужной модели
            if isinstance(response, model):
                logger.debug("Ответ уже является экземпляром целевой модели")
                return response
            
            # Если ответ - строка JSON
            if isinstance(response, str):
                logger.debug("Валидация JSON строки")
                return self._validate_json_string(response, model, strict)
            
            # Если ответ - словарь
            elif isinstance(response, dict):
                logger.debug("Валидация словаря")
                return self._validate_dict(response, model, strict)
            
            # Если ответ - другой тип объекта
            else:
                logger.debug(f"Валидация объекта типа {type(response)}")
                return self._validate_object(response, model, strict)
                
        except PydanticValidationError as e:
            logger.error(f"Ошибка Pydantic валидации: {e}")
            raise self._convert_pydantic_error(e, model, response)
        
        except Exception as e:
            logger.error(f"Общая ошибка валидации: {e}")
            raise ValidationError(
                f"Не удалось валидировать ответ по модели {model.__name__}: {e}",
                context={
                    "model": model.__name__,
                    "response_type": type(response).__name__,
                    "response_preview": self._get_response_preview(response),
                },
                original_error=e,
            )
    
    def _validate_json_string(
        self,
        json_str: str,
        model: Type[BaseModel],
        strict: bool
    ) -> BaseModel:
        """
        Валидация JSON строки по Pydantic модели.
        
        Args:
            json_str: JSON строка для валидации
            model: Pydantic модель
            strict: Строгий режим валидации
            
        Returns:
            Валидированный Pydantic объект
        """
        try:
            # Предобработка: извлекаем JSON из markdown блоков
            cleaned_json = self._extract_json_from_markdown(json_str)
            
            # Парсинг JSON
            try:
                data = json.loads(cleaned_json)
            except json.JSONDecodeError as e:
                # Пытаемся исправить русские ключи и повторить парсинг
                try:
                    fixed_json = self._fix_russian_keys(cleaned_json, model)
                    data = json.loads(fixed_json)
                    logger.info("JSON успешно исправлен после обработки русских ключей")
                except:
                    # Если исправление не помогло, выбрасываем исходную ошибку
                    raise ValidationError(
                        f"Некорректный JSON в ответе: {e}",
                        context={
                            "json_preview": json_str[:200] + "..." if len(json_str) > 200 else json_str,
                            "error_position": getattr(e, 'pos', None),
                        },
                        original_error=e,
                    )
            
            # Валидация через Pydantic
            try:
                if strict:
                    return model.model_validate(data, strict=True)
                else:
                    return model.model_validate(data)
            except PydanticValidationError as validation_error:
                # Если валидация не удалась, пытаемся исправить русские ключи
                try:
                    fixed_json = self._fix_russian_keys(cleaned_json, model)
                    fixed_data = json.loads(fixed_json)
                    
                    logger.info("Попытка исправления русских ключей после ошибки валидации")
                    
                    if strict:
                        return model.model_validate(fixed_data, strict=True)
                    else:
                        return model.model_validate(fixed_data)
                        
                except Exception as fix_error:
                    logger.warning(f"Не удалось исправить русские ключи: {fix_error}")
                    # Выбрасываем исходную ошибку валидации
                    raise validation_error
                
        except PydanticValidationError as e:
            raise self._convert_pydantic_error(e, model, json_str)
    
    def _extract_json_from_markdown(self, text: str) -> str:
        """
        Извлекает JSON из markdown блоков кода и очищает от лишних символов.
        
        Args:
            text: Текст, который может содержать JSON в markdown блоках
            
        Returns:
            Очищенный JSON текст
        """
        import re
        
        # Удаляем лишние пробелы в начале и конце
        text = text.strip()
        
        # Паттерны для поиска JSON в markdown блоках (в порядке приоритета)
        patterns = [
            r'```json\s*\n(.*?)\n```',  # ```json ... ```
            r'```\s*\n(.*?)\n```',      # ``` ... ```
            r'```(.*?)```',             # ```...``` (без переносов)
            r'`(.*?)`',                 # `...`
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                extracted = match.group(1).strip()
                # Проверяем, что извлеченный текст похож на JSON
                if extracted.startswith(('{', '[')):
                    return extracted
        
        # Дополнительная очистка от markdown символов
        cleaned_text = self._clean_markdown_artifacts(text)
        
        # Если не нашли markdown блоки, возвращаем очищенный текст
        return cleaned_text
    
    def _clean_markdown_artifacts(self, text: str) -> str:
        """
        Очищает текст от артефактов markdown.
        
        Args:
            text: Исходный текст
            
        Returns:
            Очищенный текст
        """
        import re
        
        # Удаляем markdown символы в начале и конце
        text = text.strip()
        
        # Удаляем тройные обратные кавычки
        text = re.sub(r'^```.*?\n', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n```.*?$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^```', '', text)
        text = re.sub(r'```$', '', text)
        
        # Удаляем одинарные обратные кавычки в начале и конце строк
        text = re.sub(r'^`+', '', text, flags=re.MULTILINE)
        text = re.sub(r'`+$', '', text, flags=re.MULTILINE)
        
        # Удаляем лишние переносы строк
        text = re.sub(r'\n\s*\n', '\n', text)
        
        return text.strip()
    
    def _fix_russian_keys(self, json_str: str, model: Type[BaseModel]) -> str:
        """
        Исправляет русские ключи JSON на английские на основе Pydantic модели.
        
        Args:
            json_str: JSON строка с возможными русскими ключами
            model: Pydantic модель для определения правильных ключей
            
        Returns:
            JSON строка с исправленными ключами
        """
        try:
            import json
            
            # Парсим JSON
            data = json.loads(json_str)
            
            if not isinstance(data, dict):
                return json_str
            
            # Получаем схему модели для определения правильных ключей
            schema = model.model_json_schema()
            properties = schema.get("properties", {})
            
            # Создаем маппинг русских ключей на английские
            russian_to_english = self._create_key_mapping(properties)
            
            # Исправляем ключи
            fixed_data = self._translate_keys(data, russian_to_english)
            
            # Возвращаем исправленный JSON
            return json.dumps(fixed_data, ensure_ascii=False, indent=None)
            
        except Exception as e:
            logger.warning(f"Не удалось исправить русские ключи: {e}")
            return json_str
    
    def _create_key_mapping(self, properties: Dict[str, Any]) -> Dict[str, str]:
        """
        Создает маппинг русских ключей на английские.
        
        Args:
            properties: Свойства из JSON схемы Pydantic модели
            
        Returns:
            Словарь маппинга русских ключей на английские
        """
        # Базовый маппинг часто используемых ключей
        base_mapping = {
            # Персональные данные
            "имя": "name",
            "фамилия": "surname", 
            "возраст": "age",
            "город": "city",
            "профессия": "occupation",
            "работа": "occupation",
            "должность": "occupation",
            
            # Продукты и товары
            "название": "name",
            "наименование": "name",
            "цена": "price",
            "стоимость": "price",
            "категория": "category",
            "в_наличии": "in_stock",
            "наличие": "in_stock",
            "доступен": "in_stock",
            "доступность": "in_stock",
            
            # Общие поля
            "описание": "description",
            "тип": "type",
            "статус": "status",
            "дата": "date",
            "время": "time",
            "значение": "value",
            "количество": "quantity",
            "размер": "size",
            "вес": "weight",
            "цвет": "color",
            "материал": "material",
            
            # Технические поля
            "идентификатор": "id",
            "ид": "id",
            "версия": "version",
            "создан": "created",
            "обновлен": "updated",
            "активен": "active",
            "включен": "enabled",
        }
        
        # Добавляем маппинг на основе английских ключей модели
        english_keys = list(properties.keys())
        
        # Создаем расширенный маппинг
        mapping = base_mapping.copy()
        
        # Добавляем прямые соответствия для ключей модели
        for eng_key in english_keys:
            # Если есть прямое соответствие в базовом маппинге
            for rus_key, mapped_eng_key in base_mapping.items():
                if mapped_eng_key == eng_key:
                    mapping[rus_key] = eng_key
        
        return mapping
    
    def _translate_keys(self, data: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        Переводит ключи словаря согласно маппингу.
        
        Args:
            data: Исходный словарь
            mapping: Маппинг ключей
            
        Returns:
            Словарь с переведенными ключами
        """
        if not isinstance(data, dict):
            return data
        
        translated = {}
        
        for key, value in data.items():
            # Ищем перевод ключа
            english_key = mapping.get(key.lower(), key)
            
            # Рекурсивно обрабатываем вложенные объекты
            if isinstance(value, dict):
                translated[english_key] = self._translate_keys(value, mapping)
            elif isinstance(value, list):
                translated[english_key] = [
                    self._translate_keys(item, mapping) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                translated[english_key] = value
        
        return translated
    
    def _validate_dict(
        self,
        data: Dict[str, Any],
        model: Type[BaseModel],
        strict: bool
    ) -> BaseModel:
        """
        Валидация словаря по Pydantic модели.
        
        Args:
            data: Словарь для валидации
            model: Pydantic модель
            strict: Строгий режим валидации
            
        Returns:
            Валидированный Pydantic объект
        """
        try:
            # В Pydantic 2.x строгий режим работает по-другому
            # Используем from_attributes=False для строгого режима
            if strict:
                # Проверяем на лишние поля
                model_fields = set(model.model_fields.keys())
                data_fields = set(data.keys())
                extra_fields = data_fields - model_fields
                
                if extra_fields:
                    raise PydanticValidationError.from_exception_data(
                        "ValidationError",
                        [
                            {
                                "type": "extra_forbidden",
                                "loc": (field,),
                                "msg": f"Extra inputs are not permitted",
                                "input": data[field],
                            }
                            for field in extra_fields
                        ]
                    )
                
                return model.model_validate(data)
            else:
                return model.model_validate(data)
                
        except PydanticValidationError as e:
            raise self._convert_pydantic_error(e, model, data)
    
    def _validate_object(
        self,
        obj: Any,
        model: Type[BaseModel],
        strict: bool
    ) -> BaseModel:
        """
        Валидация произвольного объекта по Pydantic модели.
        
        Args:
            obj: Объект для валидации
            model: Pydantic модель
            strict: Строгий режим валидации
            
        Returns:
            Валидированный Pydantic объект
        """
        try:
            # Попытка прямой валидации
            if strict:
                return model.model_validate(obj, strict=True)
            else:
                return model.model_validate(obj)
                
        except PydanticValidationError:
            # Если прямая валидация не удалась, пытаемся через JSON
            try:
                json_str = json.dumps(obj, default=str)
                return self._validate_json_string(json_str, model, strict)
            except Exception as json_error:
                logger.warning(f"Не удалось сериализовать объект в JSON: {json_error}")
                raise
    
    def _convert_pydantic_error(
        self,
        pydantic_error: PydanticValidationError,
        model: Type[BaseModel],
        response: Any
    ) -> ValidationError:
        """
        Конвертация Pydantic ValidationError в Kraken ValidationError.
        
        Args:
            pydantic_error: Исходная ошибка Pydantic
            model: Pydantic модель
            response: Исходный ответ
            
        Returns:
            Kraken ValidationError с детальной информацией
        """
        # Извлечение детальной информации об ошибках
        error_details = []
        for error in pydantic_error.errors():
            error_info = {
                "field": ".".join(str(loc) for loc in error.get("loc", [])),
                "message": error.get("msg", ""),
                "type": error.get("type", ""),
                "input": error.get("input"),
            }
            error_details.append(error_info)
        
        # Формирование читаемого сообщения об ошибке
        error_messages = []
        for detail in error_details:
            if detail["field"]:
                error_messages.append(f"Поле '{detail['field']}': {detail['message']}")
            else:
                error_messages.append(detail["message"])
        
        main_message = f"Ошибка валидации по модели {model.__name__}:\n" + "\n".join(error_messages)
        
        return ValidationError(
            main_message,
            context={
                "model": model.__name__,
                "error_count": len(error_details),
                "error_details": error_details,
                "response_type": type(response).__name__,
                "response_preview": self._get_response_preview(response),
            },
            original_error=pydantic_error,
        )
    
    def _get_response_preview(self, response: Any, max_length: int = 200) -> str:
        """
        Получение превью ответа для логирования.
        
        Args:
            response: Ответ для превью
            max_length: Максимальная длина превью
            
        Returns:
            Строковое представление ответа (обрезанное)
        """
        try:
            if isinstance(response, str):
                preview = response
            elif isinstance(response, (dict, list)):
                preview = json.dumps(response, ensure_ascii=False, indent=None)
            else:
                preview = str(response)
            
            if len(preview) > max_length:
                return preview[:max_length] + "..."
            return preview
            
        except Exception:
            return f"<{type(response).__name__} object>"
    
    def validate_schema_compatibility(
        self,
        model: Type[BaseModel],
        check_outlines_support: bool = True
    ) -> Dict[str, Any]:
        """
        Проверка совместимости Pydantic модели с structured output.
        
        Args:
            model: Pydantic модель для проверки
            check_outlines_support: Проверять ли совместимость с Outlines
            
        Returns:
            Словарь с результатами проверки совместимости
        """
        logger.debug(f"Проверка совместимости модели {model.__name__}")
        
        result = {
            "model_name": model.__name__,
            "is_compatible": True,
            "issues": [],
            "warnings": [],
            "schema": None,
        }
        
        try:
            # Получение JSON схемы
            schema = model.model_json_schema()
            result["schema"] = schema
            
            # Базовые проверки схемы
            if not schema:
                result["is_compatible"] = False
                result["issues"].append("Модель имеет пустую JSON схему")
                return result
            
            if "type" not in schema:
                result["is_compatible"] = False
                result["issues"].append("Схема не содержит поле 'type'")
            
            # Проверка сложности схемы
            self._check_schema_complexity(schema, result)
            
            # Проверка поддержки Outlines (если требуется)
            if check_outlines_support:
                self._check_outlines_compatibility(schema, result)
            
            logger.debug(f"Проверка совместимости завершена: compatible={result['is_compatible']}")
            
        except Exception as e:
            logger.error(f"Ошибка проверки совместимости модели {model.__name__}: {e}")
            result["is_compatible"] = False
            result["issues"].append(f"Ошибка получения схемы: {e}")
        
        return result
    
    def _check_schema_complexity(self, schema: Dict[str, Any], result: Dict[str, Any]) -> None:
        """
        Проверка сложности JSON схемы.
        
        Args:
            schema: JSON схема для проверки
            result: Результат проверки для обновления
        """
        # Проверка глубины вложенности
        max_depth = self._calculate_schema_depth(schema)
        if max_depth > 5:
            result["warnings"].append(f"Глубокая вложенность схемы: {max_depth} уровней")
        
        # Проверка количества полей
        properties = schema.get("properties", {})
        if len(properties) > 20:
            result["warnings"].append(f"Много полей в схеме: {len(properties)}")
        
        # Проверка сложных типов
        if self._has_complex_types(schema):
            result["warnings"].append("Схема содержит сложные типы (union, anyOf, oneOf)")
    
    def _check_outlines_compatibility(self, schema: Dict[str, Any], result: Dict[str, Any]) -> None:
        """
        Проверка совместимости с Outlines библиотекой.
        
        Args:
            schema: JSON схема для проверки
            result: Результат проверки для обновления
        """
        # Проверка поддерживаемых типов Outlines
        unsupported_features = []
        
        # Рекурсивная проверка схемы на неподдерживаемые возможности
        self._check_schema_features(schema, unsupported_features)
        
        if unsupported_features:
            result["warnings"].extend([
                f"Возможно неподдерживаемая Outlines возможность: {feature}"
                for feature in unsupported_features
            ])
    
    def _calculate_schema_depth(self, schema: Dict[str, Any], current_depth: int = 0) -> int:
        """
        Вычисление максимальной глубины вложенности схемы.
        
        Args:
            schema: JSON схема
            current_depth: Текущая глубина
            
        Returns:
            Максимальная глубина вложенности
        """
        max_depth = current_depth
        
        # Проверка properties
        properties = schema.get("properties", {})
        for prop_schema in properties.values():
            if isinstance(prop_schema, dict):
                depth = self._calculate_schema_depth(prop_schema, current_depth + 1)
                max_depth = max(max_depth, depth)
        
        # Проверка items (для массивов)
        items = schema.get("items")
        if isinstance(items, dict):
            depth = self._calculate_schema_depth(items, current_depth + 1)
            max_depth = max(max_depth, depth)
        
        # Проверка anyOf, oneOf, allOf
        for key in ["anyOf", "oneOf", "allOf"]:
            schemas_list = schema.get(key, [])
            for sub_schema in schemas_list:
                if isinstance(sub_schema, dict):
                    depth = self._calculate_schema_depth(sub_schema, current_depth + 1)
                    max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _has_complex_types(self, schema: Dict[str, Any]) -> bool:
        """
        Проверка наличия сложных типов в схеме.
        
        Args:
            schema: JSON схема
            
        Returns:
            True если схема содержит сложные типы
        """
        # Проверка union типов (но игнорируем простые Optional поля)
        if "anyOf" in schema:
            # Если это просто Optional поле (string | null), не считаем сложным
            any_of = schema["anyOf"]
            if len(any_of) == 2 and any(opt.get("type") == "null" for opt in any_of):
                # Это Optional поле, не сложный тип
                pass
            else:
                return True
        
        if any(key in schema for key in ["oneOf", "allOf"]):
            return True
        
        # Рекурсивная проверка вложенных схем
        properties = schema.get("properties", {})
        for prop_schema in properties.values():
            if isinstance(prop_schema, dict) and self._has_complex_types(prop_schema):
                return True
        
        items = schema.get("items")
        if isinstance(items, dict) and self._has_complex_types(items):
            return True
        
        return False
    
    def _check_schema_features(self, schema: Dict[str, Any], unsupported: List[str]) -> None:
        """
        Рекурсивная проверка возможностей схемы на совместимость.
        
        Args:
            schema: JSON схема для проверки
            unsupported: Список для накопления неподдерживаемых возможностей
        """
        # Проверка специфичных ключевых слов JSON Schema
        potentially_unsupported = [
            "patternProperties", "additionalProperties", "dependencies",
            "const", "enum", "format", "pattern"
        ]
        
        for keyword in potentially_unsupported:
            if keyword in schema:
                unsupported.append(keyword)
        
        # Рекурсивная проверка вложенных схем
        properties = schema.get("properties", {})
        for prop_schema in properties.values():
            if isinstance(prop_schema, dict):
                self._check_schema_features(prop_schema, unsupported)
        
        items = schema.get("items")
        if isinstance(items, dict):
            self._check_schema_features(items, unsupported)
        
        for key in ["anyOf", "oneOf", "allOf"]:
            schemas_list = schema.get(key, [])
            for sub_schema in schemas_list:
                if isinstance(sub_schema, dict):
                    self._check_schema_features(sub_schema, unsupported)
    
    def create_example_instance(self, model: Type[BaseModel]) -> BaseModel:
        """
        Создание примера экземпляра Pydantic модели с заполненными полями.
        
        Args:
            model: Pydantic модель
            
        Returns:
            Экземпляр модели с примерными данными
            
        Raises:
            ValidationError: При ошибке создания примера
        """
        logger.debug(f"Создание примера экземпляра для модели {model.__name__}")
        
        try:
            # Получение схемы модели
            schema = model.model_json_schema()
            
            # Генерация примерных данных на основе схемы
            example_data = self._generate_example_data(schema)
            
            # Создание и валидация экземпляра
            instance = model.model_validate(example_data)
            
            logger.debug(f"Пример экземпляра создан успешно")
            return instance
            
        except Exception as e:
            logger.error(f"Ошибка создания примера для модели {model.__name__}: {e}")
            raise ValidationError(
                f"Не удалось создать пример экземпляра для модели {model.__name__}: {e}",
                context={"model": model.__name__},
                original_error=e,
            )
    
    def _generate_example_data(self, schema: Dict[str, Any], root_schema: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Генерация примерных данных на основе JSON схемы.
        
        Args:
            schema: JSON схема
            root_schema: Корневая схема для разрешения $ref
            
        Returns:
            Словарь с примерными данными
        """
        if schema.get("type") != "object":
            return {}
        
        # Если root_schema не передана, используем текущую схему как корневую
        if root_schema is None:
            root_schema = schema
        
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        example_data = {}
        
        for field_name, field_schema in properties.items():
            # Генерируем данные для всех обязательных полей
            if field_name in required:
                example_data[field_name] = self._generate_field_example(field_schema, root_schema)
            # Также генерируем для полей с default значениями
            elif "default" in field_schema:
                example_data[field_name] = self._generate_field_example(field_schema, root_schema)
        
        return example_data
    
    def _generate_field_example(self, field_schema: Dict[str, Any], root_schema: Dict[str, Any] = None) -> Any:
        """
        Генерация примерного значения для поля на основе его схемы.
        
        Args:
            field_schema: Схема поля
            root_schema: Корневая схема для разрешения $ref
            
        Returns:
            Примерное значение для поля
        """
        # Если есть default значение
        if "default" in field_schema:
            return field_schema["default"]
        
        # Если есть example
        if "example" in field_schema:
            return field_schema["example"]
        
        # Обработка $ref (ссылки на другие модели)
        if "$ref" in field_schema and root_schema:
            ref_path = field_schema["$ref"]
            if ref_path.startswith("#/$defs/"):
                def_name = ref_path.replace("#/$defs/", "")
                if "$defs" in root_schema and def_name in root_schema["$defs"]:
                    ref_schema = root_schema["$defs"][def_name]
                    return self._generate_example_data(ref_schema, root_schema)
        
        # Обработка anyOf (union типов)
        if "anyOf" in field_schema:
            # Берем первый не-null тип
            for option in field_schema["anyOf"]:
                if option.get("type") != "null":
                    return self._generate_field_example(option, root_schema)
            return None
        
        # Генерация на основе типа
        field_type = field_schema.get("type")
        
        if field_type == "string":
            return "example_string"
        elif field_type == "integer":
            return 42
        elif field_type == "number":
            return 3.14
        elif field_type == "boolean":
            return True
        elif field_type == "array":
            items_schema = field_schema.get("items", {})
            return [self._generate_field_example(items_schema, root_schema)]
        elif field_type == "object":
            return self._generate_example_data(field_schema, root_schema)
        else:
            return None


# Глобальный экземпляр валидатора для удобства использования
validator = StructuredOutputValidator()


# Удобные функции для быстрого использования
async def validate_structured_response(
    response: Union[str, Dict[str, Any], Any],
    model: Type[BaseModel],
    strict: bool = True
) -> BaseModel:
    """
    Быстрая валидация structured output ответа.
    
    Args:
        response: Ответ для валидации
        model: Pydantic модель
        strict: Строгий режим валидации
        
    Returns:
        Валидированный Pydantic объект
    """
    return await validator.validate_response(response, model, strict)


def check_model_compatibility(
    model: Type[BaseModel],
    check_outlines_support: bool = True
) -> Dict[str, Any]:
    """
    Быстрая проверка совместимости модели.
    
    Args:
        model: Pydantic модель
        check_outlines_support: Проверять ли совместимость с Outlines
        
    Returns:
        Результат проверки совместимости
    """
    return validator.validate_schema_compatibility(model, check_outlines_support)


def create_model_example(model: Type[BaseModel]) -> BaseModel:
    """
    Быстрое создание примера экземпляра модели.
    
    Args:
        model: Pydantic модель
        
    Returns:
        Пример экземпляра модели
    """
    return validator.create_example_instance(model)