#!/usr/bin/env python3
"""
Тест поддержки всех возможных thinking токенов.

Проверяет работу с различными вариантами токенов рассуждения,
которые могут использоваться разными моделями и провайдерами.
"""

import asyncio
from kraken_llm.client.reasoning import (
    ReasoningLLMClient, 
    ReasoningConfig, 
    ReasoningModelType
)
from kraken_llm.config.settings import LLMConfig


async def test_all_thinking_tokens_detection():
    """Тест определения всех возможных thinking токенов"""
    print("=== Тест определения всех thinking токенов ===")
    
    config = LLMConfig()
    client = ReasoningLLMClient(config)
    
    # Получаем все возможные токены
    all_tokens = client._get_all_possible_thinking_tokens()
    
    print(f"Поддерживается {len(all_tokens)} вариантов thinking токенов:")
    for i, (start, end) in enumerate(all_tokens, 1):
        print(f"  {i:2d}. {start} ... {end}")
    
    # Проверяем, что есть основные варианты
    token_strings = [f"{start}{end}" for start, end in all_tokens]
    
    expected_tokens = [
        "<thinking></thinking>",
        "<think></think>", 
        "<reasoning></reasoning>",
        "<reason></reason>",
        "<thought></thought>",
        "<analysis></analysis>",
        "<reflection></reflection>"
    ]
    
    for expected in expected_tokens:
        if expected in token_strings:
            print(f"✓ {expected} поддерживается")
        else:
            print(f"✗ {expected} НЕ поддерживается")
            return False
    
    print("✓ Все основные thinking токены поддерживаются")
    return True


async def test_universal_token_extraction():
    """Тест универсального извлечения токенов"""
    print("=== Тест универсального извлечения токенов ===")
    
    config = LLMConfig()
    client = ReasoningLLMClient(config)
    
    # Тестируем различные форматы ответов с разными токенами
    test_cases = [
        # Qwen стиль
        ("Рассуждаю над задачей...</think>Ответ готов", "<think>", "Рассуждаю над задачей...", "Ответ готов"),
        
        # OpenAI стиль
        ("<thinking>Анализирую проблему...</thinking>Решение найдено", "<thinking>", "Анализирую проблему...", "Решение найдено"),
        
        # Reasoning токены
        ("<reasoning>Логически размышляю...</reasoning>Вывод сделан", "<reasoning>", "Логически размышляю...", "Вывод сделан"),
        
        # Альтернативные токены
        ("<thought>Обдумываю варианты...</thought>Вариант выбран", "<thought>", "Обдумываю варианты...", "Вариант выбран"),
        ("<analysis>Провожу анализ...</analysis>Анализ завершен", "<analysis>", "Провожу анализ...", "Анализ завершен"),
        
        # Экзотические токены
        ("<scratchpad>Черновые заметки...</scratchpad>Заметки готовы", "<scratchpad>", "Черновые заметки...", "Заметки готовы"),
        ("<deliberation>Взвешиваю решения...</deliberation>Решение принято", "<deliberation>", "Взвешиваю решения...", "Решение принято"),
    ]
    
    for response_text, expected_token, expected_thinking, expected_answer in test_cases:
        thinking_content, final_answer = client._extract_thinking_from_content(response_text)
        
        print(f"Токен {expected_token}:")
        print(f"  Thinking: {thinking_content[:30]}...")
        print(f"  Ответ: {final_answer[:30]}...")
        
        # Проверяем корректность извлечения
        if thinking_content.strip() == expected_thinking.strip() and final_answer.strip() == expected_answer.strip():
            print("  ✓ Извлечение корректно")
        else:
            print("  ✗ Ошибка извлечения")
            print(f"    Ожидалось thinking: {expected_thinking}")
            print(f"    Получено thinking: {thinking_content}")
            print(f"    Ожидался ответ: {expected_answer}")
            print(f"    Получен ответ: {final_answer}")
            return False
        
        print()
    
    print("✓ Универсальное извлечение токенов работает корректно")
    return True


async def test_fallback_token_parsing():
    """Тест fallback парсинга различных токенов"""
    print("=== Тест fallback парсинга токенов ===")
    
    config = LLMConfig()
    client = ReasoningLLMClient(config)
    
    # Тестируем различные форматы ответов
    test_responses = [
        # Стандартные токены
        ("<thinking>Анализирую проблему...</thinking>Ответ готов", "thinking"),
        ("<think>Размышляю над задачей...</think>Решение найдено", "think"),
        ("<reasoning>Логически рассуждаю...</reasoning>Вывод сделан", "reasoning"),
        ("<reason>Ищу причины...</reason>Причина найдена", "reason"),
        
        # Альтернативные токены
        ("<thought>Обдумываю варианты...</thought>Вариант выбран", "thought"),
        ("<analysis>Провожу анализ...</analysis>Анализ завершен", "analysis"),
        ("<reflection>Размышляю глубоко...</reflection>Понимание достигнуто", "reflection"),
        ("<internal>Внутренний диалог...</internal>Решение принято", "internal"),
        
        # Специальные токены
        ("<cot>Chain of thought процесс...</cot>Цепочка завершена", "cot"),
        ("<scratchpad>Черновые заметки...</scratchpad>Заметки готовы", "scratchpad"),
        ("<process>Обрабатываю информацию...</process>Обработка завершена", "process"),
        
        # Экзотические токены
        ("<deliberation>Взвешиваю решения...</deliberation>Решение взвешено", "deliberation"),
        ("<brainstorm>Генерирую идеи...</brainstorm>Идеи сгенерированы", "brainstorm")
    ]
    
    for response_text, token_type in test_responses:
        chain = client._parse_thinking_style_response(response_text)
        
        print(f"Токен {token_type}:")
        print(f"  Thinking блоков: {len(chain.thinking_blocks) if chain.thinking_blocks else 0}")
        print(f"  Финальный ответ: {chain.final_answer[:30]}...")
        
        # Проверяем успешность парсинга
        if chain.thinking_blocks and len(chain.thinking_blocks) > 0:
            thinking_content = chain.thinking_blocks[0].content
            print(f"  Thinking содержимое: {thinking_content[:40]}...")
            print("  ✓ Парсинг успешен")
        else:
            print("  ✗ Парсинг не удался")
            return False
        
        print()
    
    print("✓ Все токены успешно парсятся")
    return True


async def test_mixed_tokens_in_response():
    """Тест ответов с несколькими типами токенов"""
    print("=== Тест смешанных токенов в ответе ===")
    
    config = LLMConfig()
    client = ReasoningLLMClient(config)
    
    # Ответ с несколькими типами токенов (должен найти первый)
    mixed_response = """
    Начинаю анализ задачи.
    
    <thinking>
    Сначала нужно понять суть проблемы.
    Проблема заключается в том, что...
    </thinking>
    
    Промежуточный текст.
    
    <reasoning>
    Теперь применю логическое рассуждение.
    Если A, то B. A истинно, значит B тоже истинно.
    </reasoning>
    
    Финальный ответ: Задача решена успешно.
    """
    
    chain = client._parse_thinking_style_response(mixed_response)
    
    print(f"Thinking блоков найдено: {len(chain.thinking_blocks) if chain.thinking_blocks else 0}")
    print(f"Шагов рассуждения: {len(chain.steps)}")
    print(f"Финальный ответ: {chain.final_answer[:50]}...")
    
    if chain.thinking_blocks:
        print(f"Первый thinking блок: {chain.thinking_blocks[0].content[:60]}...")
    
    # Должен найти хотя бы один thinking блок
    assert chain.thinking_blocks and len(chain.thinking_blocks) > 0
    assert len(chain.steps) > 0
    assert "Финальный ответ: Задача решена успешно." in chain.final_answer
    
    print("✓ Смешанные токены обрабатываются корректно")
    return True


async def test_edge_cases():
    """Тест граничных случаев с токенами"""
    print("=== Тест граничных случаев ===")
    
    config = LLMConfig()
    client = ReasoningLLMClient(config)
    
    # Тестовые случаи
    edge_cases = [
        # Пустой thinking блок
        ("<thinking></thinking>Только ответ", "Пустой thinking"),
        
        # Thinking без закрывающего тега
        ("<thinking>Незавершенное рассуждение... Ответ", "Незавершенный thinking"),
        
        # Вложенные токены (некорректный случай)
        ("<thinking>Внешний <think>внутренний</think> блок</thinking>Ответ", "Вложенные токены"),
        
        # Токены в неправильном порядке
        ("</thinking>Обратный порядок<thinking>Ответ", "Обратный порядок"),
        
        # Множественные одинаковые токены
        ("<thinking>Первый</thinking> текст <thinking>Второй</thinking>Ответ", "Множественные токены"),
        
        # Токены с пробелами
        ("< thinking >С пробелами</ thinking >Ответ", "Токены с пробелами"),
        
        # Без токенов вообще
        ("Просто текст без thinking токенов", "Без токенов")
    ]
    
    for response_text, case_name in edge_cases:
        print(f"\nТест: {case_name}")
        print(f"Входной текст: {response_text[:50]}...")
        
        try:
            chain = client._parse_thinking_style_response(response_text)
            
            print(f"  Thinking блоков: {len(chain.thinking_blocks) if chain.thinking_blocks else 0}")
            print(f"  Шагов: {len(chain.steps)}")
            print(f"  Финальный ответ: {chain.final_answer[:30]}...")
            
            # Проверяем, что парсинг не упал
            assert len(chain.steps) > 0, "Должен быть хотя бы один шаг"
            assert chain.final_answer, "Должен быть финальный ответ"
            
            print("  ✓ Обработано корректно")
            
        except Exception as e:
            print(f"  ✗ Ошибка: {e}")
            return False
    
    print("\n✓ Все граничные случаи обработаны корректно")
    return True


async def test_performance_with_many_tokens():
    """Тест производительности с большим количеством токенов"""
    print("=== Тест производительности ===")
    
    config = LLMConfig()
    client = ReasoningLLMClient(config)
    
    # Создаем ответ с множеством различных токенов
    all_tokens = client._get_all_possible_thinking_tokens()
    
    # Берем первые 5 типов токенов для теста
    test_tokens = all_tokens[:5]
    
    large_response = "Начало ответа.\n\n"
    
    for i, (start, end) in enumerate(test_tokens):
        large_response += f"{start}\nРассуждение блока {i+1}.\nДетальный анализ проблемы.\n{end}\n\n"
    
    large_response += "Финальный ответ после всех рассуждений."
    
    print(f"Тестовый ответ содержит {len(test_tokens)} thinking блоков")
    print(f"Общая длина: {len(large_response)} символов")
    
    import time
    start_time = time.time()
    
    chain = client._parse_thinking_style_response(large_response)
    
    end_time = time.time()
    parse_time = end_time - start_time
    
    print(f"Время парсинга: {parse_time:.4f}s")
    print(f"Найдено thinking блоков: {len(chain.thinking_blocks) if chain.thinking_blocks else 0}")
    print(f"Создано шагов: {len(chain.steps)}")
    
    # Проверяем результат
    assert chain.thinking_blocks and len(chain.thinking_blocks) > 0
    assert len(chain.steps) > 0
    assert "Финальный ответ после всех рассуждений." in chain.final_answer
    
    print("✓ Производительность приемлемая")
    return True


async def run_all_thinking_tokens_tests():
    """Запуск всех тестов thinking токенов"""
    print("Тестирование всех возможных Thinking токенов")
    print("=" * 60)
    print()
    
    tests = [
        ("Определение всех токенов", test_all_thinking_tokens_detection),
        ("Универсальное извлечение токенов", test_universal_token_extraction),
        ("Fallback парсинг токенов", test_fallback_token_parsing),
        ("Смешанные токены", test_mixed_tokens_in_response),
        ("Граничные случаи", test_edge_cases),
        ("Тест производительности", test_performance_with_many_tokens)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"Запуск теста: {test_name}")
        try:
            result = await test_func()
            results.append((test_name, result))
            status = "✓ ПРОЙДЕН" if result else "✗ ПРОВАЛЕН"
            print(f"Результат: {status}")
        except Exception as e:
            results.append((test_name, False))
            print(f"✗ ОШИБКА: {e}")
        
        print("-" * 50)
        print()
    
    # Итоговый отчет
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"Пройдено тестов: {passed}/{total}")
    
    if passed == total:
        print("ВСЕ ТЕСТЫ THINKING ТОКЕНОВ ПРОЙДЕНЫ!")
        print("\nПоддерживаемые токены:")
        
        config = LLMConfig()
        client = ReasoningLLMClient(config)
        all_tokens = client._get_all_possible_thinking_tokens()
        
        for start, end in all_tokens:
            print(f"  • {start} ... {end}")
            
    else:
        print(f"⚠ {total - passed} тестов провалено")
    
    return passed == total


if __name__ == "__main__":
    asyncio.run(run_all_thinking_tokens_tests())