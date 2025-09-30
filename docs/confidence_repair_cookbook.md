# Confidence & Repair Cookbook

Этот документ описывает практики работы с метриками уверенности (confidence, logprobs) и режимами «ремонта» низкоуверенных токенов (shadow/server), а также порядок интеграции во все режимы генерации.

## 1. Термины
- Уверенность токена: p = exp(logprob). Диапазон 0..1.
- Порог токена (per_token_threshold): если токен ниже порога — кандидат на ремонт.
- Режимы ремонта:
  - off — ремонт выключен
  - shadow — клиентский «теневой» ремонт (требуются logprobs в стриме)
  - server — серверный ремонт через порог min_p (если поддерживается, напр. vLLM)

## 2. Shadow‑ремонт: поток событий

Сценарий (упрощённо):

```text
Пользователь → Сообщения → [StreamingLLMClient]
  ↓
(1) Основной поток chat.completions (stream, logprobs)
  ↓
(2) Приходит токен с logprob → confidence = exp(logprob)
  ↓ если confidence < per_token_threshold
(3) Эмит «token» (для визуализации), затем «rollback» (мягкое стирание в UI)
  ↓
(4) Форк ремонтного потока от принятого префикса (более «жёсткие» параметры: ниже T, top_p и т.д.)
  ↓
(5) Приём токенов из ремонтного потока; если ремонт исчерпал лимит попыток — выбирается лучший токен (наиболее уверенный) из наблюдавшихся
  ↓
(6) Продолжение генерации до done (включая fallback‑продолжение при обрывах)
```

Анти‑loop:
- детектор повторов (длинный хвост, n‑граммы, последнее «предложение»),
- динамич. повышение frequency/presence penalty на ветках ремонта/продолжения,
- принудительное завершение done при множественных срабатываниях (repeat_hits_stop_threshold).

## 3. Server‑ремонт (min_p)

- Включается через repair_mode=server.
- min_p пробрасывается в extra_body.min_p (если не задан, берётся per_token_repair_threshold).
- Поддерживает не все бэкенды, ориентирован на совместимые (например, vLLM).

## 4. Параметры конфигурации

```env
LLM_REPAIR_MODE=shadow            # off|shadow|server
LLM_PER_TOKEN_REPAIR_THRESHOLD=0.5
LLM_MAX_ATTEMPTS_PER_TOKEN=8
LLM_MAX_LIVE_REPAIRS=8
LLM_SERVER_MIN_P=0.4              # для server‑режима

# Рекомендации для shadow
LLM_FORCE_OPENAI_STREAMING=true
LLM_LOGPROBS=true
LLM_TOP_LOGPROBS=5
```

Через LLMConfig:

```python
from kraken_llm import LLMConfig
cfg = LLMConfig(
    repair_mode="shadow",
    per_token_repair_threshold=0.5,
    max_attempts_per_token=8,
    max_live_repairs=8,
    server_min_p=0.4,
    force_openai_streaming=True,
    logprobs=True,
    top_logprobs=5,
)
```

## 5. Интеграция по режимам

### 5.1 Streaming (Realtime/Aggregated)
- Реалтайм поток с теневым ремонтом: token_stream_with_shadow_repair(client, messages, per_token_threshold, ...)
- UI может мягко стирать символы при rollback, чтобы избежать дублей.
- Глобальный бюджет ремонтов и «окна без ремонта» предотвращают зацикливание.

CLI:
```bash
python3 examples/chatbot_streaming_colors.py \
  --mode realtime \
  --repair-mode shadow \
  --per-token-threshold 0.5 \
  --max-attempts-per-token 8 \
  --no-rollback-marker
```

### 5.2 Structured Output (SO)
- Native (response_format): включены консервативные повторные попытки для получения валидного JSON; при server‑режиме пробрасывается min_p.
- Outlines: пробрасывание min_p через extra_body при server‑режиме; инкрементальный парсер в streaming Outlines варианте.

### 5.3 Reasoning (native thinking / CoT)
- При включённом repair_mode в потоковом reasoning используется тот же устойчивый поток с ремонтом.
- Rollback уменьшает текущий буфер шага; шаги продолжают парситься; анти‑loop защита активна.

## 6. Рекомендации и настройки по умолчанию
- per_token_threshold: 0.4–0.6 в зависимости от модели и шума в logprobs.
- max_attempts_per_token: 3–8, max_live_repairs: 4–12 (баланс задержки и устойчивости).
- shadow‑режим: включите FORCE_OPENAI_STREAMING, LOGPROBS, TOP_LOGPROBS.
- Увеличьте таймауты для длинных ответов (LLM_READ_TIMEOUT/WRITE_TIMEOUT).
- Включите подавление предупреждений об обрывах (LLM_SUPPRESS_STREAM_WARNINGS=true), если используете нестабильные каналы SSE.

## 7. Отладка и диагностика
- Логи на уровне DEBUG показывают переключения потоков, события rollback, эскалацию параметров, повторные попытки и принудительные завершения.
- При зацикливании проверяйте: слишком высокий per_token_threshold, непредсказуемая модель при низких температурах, агрессивные stop‑последовательности.

## 8. Частые вопросы (FAQ)
- «Почему я вижу дубли текста в realtime?» — в примере включите `--no-rollback-marker` (мягкое стирание уже реализовано), избегайте печати маркеров отката.
- «Почему ответ обрывается?» — включите SUPPRESS_STREAM_WARNINGS и/или используйте fallback‑продолжение (реализовано внутри ремонта), увеличьте таймауты.
- «Можно ли управлять порогом принудительной остановки?» — да, параметр repeat_hits_stop_threshold зашит в состояние ремонта; при необходимости можно вынести его в конфигурацию.
