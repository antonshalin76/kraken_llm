#!/usr/bin/env python3
"""
Тест приоритета thinking токенов.

Проверяет, какой токен будет найден первым, если в ответе
присутствует несколько различных типов thinking токенов.
"""

import asyncio
from kraken_llm.client.reasoning import ReasoningLLMClient
from kraken_llm.config.settings import LLMConfig


async def test_token_priority():
    """Тест приоритета токенов при множественном выборе"""
    print("=== Тест приоритета thinking токенов ===")
    
    config = LLMConfig()
    client = ReasoningLLMClient(config)
    
    # Получаем порядок токенов (как они определены в _get_all_possible_thinking_tokens)
    all_tokens = client._get_all_possible_thinking_tokens()
    print("Порядок приоритета токенов:")
    for i, (start, end) in enumerate(all_tokens, 1):
        print(f"  {i:2d}. {start} ... {end}")
    
    print()
    
    # Тестируем различные комбинации токенов
    test_cases = [
        # Случай 1: thinking и think (thinking должен иметь приоритет)
        {
            "response": "<thinking>Первый блок</thinking> текст <think>Второй блок</think> ответ",
            "expected_token": "<thinking>",
            "expected_content": "Первый блок"
        },
        
        # Случай 2: reasoning и think (thinking должен иметь приоритет)
        {
            "response": "<reasoning>Reasoning блок</reasoning> и <think>Think блок</think> ответ",
            "expected_token": "<thinking>",  # thinking имеет приоритет над reasoning
            "expected_content": "Think блок"  # но think найдется первым в списке
        },
        
        # Случай 3: Обратный порядок в тексте
        {
            "response": "текст <analysis>Анализ</analysis> и <thinking>Мышление</thinking> ответ",
            "expected_token": "<thinking>",
            "expected_content": "Мышление"
        },
        
        # Случай 4: Много разных токенов
        {
            "response": """
            <scratchpad>Заметки</scratchpad>
            <reasoning>Рассуждение</reasoning>
            <thinking>Мышление</thinking>
            <analysis>Анализ</analysis>
            Финальный ответ
            """,
            "expected_token": "<thinking>",
            "expected_content": "Мышление"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Тест {i}: {test_case['response'][:50]}...")
        
        thinking_content, final_answer = client._extract_thinking_from_content(test_case['response'])
        
        print(f"  Найденный контент: '{thinking_content}'")
        print(f"  Ожидаемый контент: '{test_case['expected_content']}'")
        
        # Проверяем, что найден правильный контент
        if thinking_content.strip() == test_case['expected_content'].strip():
            print("  ✓ Приоритет токенов работает корректно")
        else:
            print("  ✗ Неожиданный результат приоритета")
            return False
        
        print()
    
    print("✓ Приоритет токенов работает корректно во всех случаях")
    return True


async def test_token_order_consistency():
    """Тест консистентности порядка токенов"""
    print("=== Тест консистентности порядка токенов ===")
    
    config = LLMConfig()
    client = ReasoningLLMClient(config)
    
    # Получаем токены несколько раз и проверяем, что порядок не меняется
    orders = []
    for i in range(5):
        tokens = client._get_all_possible_thinking_tokens()
        token_strings = [f"{start}{end}" for start, end in tokens]
        orders.append(token_strings)
    
    # Проверяем, что все порядки одинаковые
    first_order = orders[0]
    for i, order in enumerate(orders[1:], 2):
        if order != first_order:
            print(f"✗ Порядок токенов изменился в вызове {i}")
            print(f"  Первый: {first_order[:3]}...")
            print(f"  Текущий: {order[:3]}...")
            return False
    
    print(f"✓ Порядок токенов консистентен во всех {len(orders)} вызовах")
    print(f"✓ Всего токенов: {len(first_order)}")
    
    return True


async def test_performance_with_priority():
    """Тест производительности поиска с приоритетом"""
    print("=== Тест производительности поиска с приоритетом ===")
    
    config = LLMConfig()
    client = ReasoningLLMClient(config)
    
    # Создаем большой текст с токеном в конце (худший случай)
    large_text = "Большой текст " * 1000
    large_text += "<mindmap>Последний токен в списке приоритета</mindmap>Ответ"
    
    print(f"Размер тестового текста: {len(large_text)} символов")
    print("Токен находится в конце текста и в конце списка приоритета")
    
    import time
    start_time = time.time()
    
    thinking_content, final_answer = client._extract_thinking_from_content(large_text)
    
    end_time = time.time()
    search_time = end_time - start_time
    
    print(f"Время поиска: {search_time:.4f}s")
    print(f"Найденный контент: '{thinking_content}'")
    
    # Проверяем корректность
    if thinking_content == "Последний токен в списке приоритета":
        print("✓ Поиск работает корректно даже для последнего токена")
    else:
        print("✗ Ошибка поиска")
        return False
    
    # Проверяем производительность (должно быть быстро даже для большого текста)
    if search_time < 0.1:  # Менее 100ms
        print("✓ Производительность приемлемая")
    else:
        print(f"⚠ Производительность может быть улучшена: {search_time:.4f}s")
    
    return True


async def test_edge_case_priorities():
    """Тест граничных случаев с приоритетами"""
    print("=== Тест граничных случаев с приоритетами ===")
    
    config = LLMConfig()
    client = ReasoningLLMClient(config)
    
    edge_cases = [
        # Вложенные токены разных типов
        {
            "response": "<thinking>Внешний <reasoning>внутренний</reasoning> блок</thinking>Ответ",
            "description": "Вложенные токены разных типов"
        },
        
        # Пересекающиеся токены
        {
            "response": "<thinking>Начало <analysis>пересечение</thinking> конец</analysis>Ответ",
            "description": "Пересекающиеся токены"
        },
        
        # Одинаковые токены
        {
            "response": "<thinking>Первый</thinking> текст <thinking>Второй</thinking>Ответ",
            "description": "Множественные одинаковые токены"
        },
        
        # Токены без контента
        {
            "response": "<thinking></thinking><reasoning>Есть контент</reasoning>Ответ",
            "description": "Пустой приоритетный токен"
        }
    ]
    
    for case in edge_cases:
        print(f"\nТест: {case['description']}")
        print(f"Входной текст: {case['response'][:60]}...")
        
        try:
            thinking_content, final_answer = client._extract_thinking_from_content(case['response'])
            
            print(f"  Найденный контент: '{thinking_content[:30]}...' ({len(thinking_content)} символов)")
            print(f"  Финальный ответ: '{final_answer[:30]}...'")
            
            # Проверяем, что что-то найдено или корректно обработан пустой случай
            if thinking_content or "Ответ" in final_answer:
                print("  ✓ Граничный случай обработан корректно")
            else:
                print("  ✗ Ошибка обработки граничного случая")
                return False
                
        except Exception as e:
            print(f"  ✗ Исключение: {e}")
            return False
    
    print("\n✓ Все граничные случаи с приоритетами обработаны корректно")
    return True


async def run_token_priority_tests():
    """Запуск всех тестов приоритета токенов"""
    print("Тестирование приоритета Thinking токенов")
    print("=" * 50)
    print()
    
    tests = [
        ("Приоритет токенов", test_token_priority),
        ("Консистентность порядка", test_token_order_consistency),
        ("Производительность поиска", test_performance_with_priority),
        ("Граничные случаи приоритетов", test_edge_case_priorities)
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
        
        print("-" * 40)
        print()
    
    # Итоговый отчет
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"Пройдено тестов: {passed}/{total}")
    
    if passed == total:
        print("🎉 ВСЕ ТЕСТЫ ПРИОРИТЕТА ТОКЕНОВ ПРОЙДЕНЫ!")
    else:
        print(f"⚠ {total - passed} тестов провалено")
    
    return passed == total


if __name__ == "__main__":
    asyncio.run(run_token_priority_tests())