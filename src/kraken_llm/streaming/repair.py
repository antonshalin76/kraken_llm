"""
Потоковые утилиты для live-ремонта токенов (теневой форк) в режиме streaming.

Предупреждение: «теневой ремонт» выполняется на стороне клиента и требует
поддержки logprobs/top_logprobs в потоковых чанках. Он останавливает основной
поток при срабатывании порога и переключается на альтернативный поток,
стартующий от префикса уже принятых токенов.

Это наилучшее приближение к точечной замене токена в реальном времени без
специальной поддержки на стороне бэкенда.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any, AsyncGenerator, Dict, List, Optional

from ..client.streaming import StreamingLLMClient
from ..config.settings import LLMConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


def _should_penalize_repeats(text: str) -> bool:
    """Эвристика: обнаружение повторов на уровне n-грамм и предложений.
    Возвращает True, если стоит усилить frequency/presence penalty.
    """
    try:
        t = (text or "").strip()
        if not t:
            return False
        # Проверка длинного хвоста-символьного повтора
        tail = t[-160:]
        if tail and t.count(tail) > 1:
            return True
        # N-граммы по словам (12-словный хвост)
        words = t.split()
        if len(words) >= 12:
            ngram = " ".join(words[-12:])
            if ngram and t.count(ngram) > 1:
                return True
        # Повтор последнего предложения
        import re
        sentences = re.split(r"[\.!?\n]+", t)
        last_sent = sentences[-1].strip() if sentences else ""
        if len(last_sent) >= 30 and t.count(last_sent) > 1:
            return True
    except Exception:
        return False
    return False


async def token_stream_with_shadow_repair(
    client: StreamingLLMClient,
    messages: List[Dict[str, str]],
    *,
    per_token_threshold: float = 0.4,
    max_tokens: int = 512,
    repair_temperature_factor: float = 0.6,
    repair_top_p_cap: float = 0.6,
    enable_cutover: bool = True,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Потоковая генерация с live-мониторингом уверенности токенов и «теневым ремонтом».

    Поведение:
    - Открывает поток chat.completions с logprobs/top_logprobs.
    - На каждом токене вычисляет confidence = exp(logprob).
    - При падении confidence ниже per_token_threshold — немедленно сворачивает основной
      поток, выдаёт событие rollback последнего токена и стартует «ремонтный» поток
      (сниженная температура/ограниченный top_p) от префикса уже принятых токенов.
    - Далее отдаёт токены из ремонтного потока до завершения.

    Yields события словарями:
      - {"type": "token", "token": str, "confidence": float}
      - {"type": "rollback", "count": int}  # удалить N последних символов (наивно)
      - {"type": "done"}

    Важно: «теневой ремонт» требует, чтобы сервер отдавал logprobs в chunks.
    """
    # Подготовка параметров для основного потока
    params = client._prepare_openai_params(
        messages=messages,
        stream=True,
        logprobs=True,
        top_logprobs=5,
        max_tokens=max_tokens,
    )

    # Основной поток через AsyncOpenAI
    stream_gen = await client.openai_client.chat.completions.create(**params)

    accepted_tokens: List[str] = []
    accepted_text: str = ""

    # Глобальное состояние ремонта для защиты от зацикливания
    repair_state: Dict[str, Any] = {
        "repairs_done": 0,
        "max_repairs": int(getattr(client.config, "max_live_repairs", 8) or 8),
        "no_repair_tokens_left": 0,
        "repeat_hits": 0,
        "repeat_hits_stop_threshold": 6,
        "force_stop": False,
    }

    try:
        async for chunk in stream_gen:
            # Принудительное завершение при детекции зацикливания
            if repair_state.get("force_stop"):
                yield {"type": "done"}
                return
            if not getattr(chunk, "choices", None):
                continue
            ch0 = chunk.choices[0]

            # Если пришли токенные logprobs — используем их как источник истины
            had_logprobs = False
            logprobs = getattr(ch0, "logprobs", None)
            if logprobs and getattr(logprobs, "content", None):
                had_logprobs = True
                for item in logprobs.content:
                    tok = getattr(item, "token", "") or ""
                    lp = getattr(item, "logprob", None)
                    if lp is None:
                        continue
                    conf = float(math.exp(lp))

                    # Перед тем как принять токен, проверим порог
                    clean_tok = tok.replace("Ġ", " ").replace("▁", " ")
                    should_cutover = (
                        enable_cutover
                        and conf < per_token_threshold
                        and repair_state.get("no_repair_tokens_left", 0) <= 0
                        and repair_state.get("repairs_done", 0) < repair_state.get("max_repairs", 8)
                    )
                    if should_cutover:
                        # Увеличиваем счётчик ремонтов
                        repair_state["repairs_done"] = repair_state.get("repairs_done", 0) + 1
                        # Показать проблемный токен и откатить его визуально, НЕ добавляя в контекст
                        yield {"type": "token", "token": tok, "confidence": conf}
                        yield {"type": "rollback", "count": len(clean_tok)}

                        # Запустить ремонтный поток от префикса, БЕЗ проблемного токена
                        # Эскалация параметров для попытки №1 (первый ретрай текущего токена)
                        attempt_idx = 1
                        base_top_p = min(repair_top_p_cap, getattr(client.config, "top_p", 1.0))
                        escalated_temp = max(0.0, client.config.temperature * (repair_temperature_factor ** attempt_idx))
                        escalated_top_p = max(0.05, base_top_p * (0.7 ** (attempt_idx - 1)))

                        # Динамические штрафы при повторениях
                        freq_pen = getattr(client.config, "frequency_penalty", 0.0)
                        pres_pen = getattr(client.config, "presence_penalty", 0.0)
                        if _should_penalize_repeats(accepted_text):
                            freq_pen = min(2.0, (freq_pen or 0.0) + 0.6)
                            pres_pen = min(2.0, (pres_pen or 0.0) + 0.3)

                        repair_cfg = LLMConfig(
                            endpoint=client.config.endpoint,
                            api_key=client.config.api_key,
                            model=client.config.model,
                            temperature=escalated_temp,
                            top_p=escalated_top_p,
                            frequency_penalty=freq_pen,
                            presence_penalty=pres_pen,
                            stream=True,
                            logprobs=True,
                            top_logprobs=5,
                            force_openai_streaming=True,
                            suppress_stream_warnings=True,
                        )
                        repair_client = StreamingLLMClient(repair_cfg)
                        try:
                            # Форк: добавляем ассистентский префикс, чтобы продолжить с корректного места
                            # ВАЖНО: сохраняем последнее пользовательское сообщение и не добавляем проблемный токен
                            fork_messages = messages + [
                                {"role": "assistant", "content": accepted_text}
                            ]

                            # Перегенерация для ЭТОГО токена с лимитом попыток
                            max_attempts = getattr(client.config, "max_attempts_per_token", None) or 3

                            nested_yielded = False
                            async for ev in _emit_simple_token_stream(
                                repair_client,
                                fork_messages,
                                max_tokens=max_tokens,
                                per_token_threshold=per_token_threshold,
                                enable_cutover=True,
                                repair_temperature_factor=repair_temperature_factor,
                                repair_top_p_cap=repair_top_p_cap,
                                max_attempts_per_token=max_attempts,
                                slot_retry_active=True,
                                slot_retry_count=1,
                                awaiting_slot_token=True,
                                repair_state=repair_state,
                            ):
                                if ev.get("type") != "done":
                                    if ev.get("type") == "token":
                                        nested_yielded = True
                                    yield ev
                            # Если вложенный поток ничего не отдал, принудительно дотянем агрегированно до конца
                            if not nested_yielded:
                                try:
                                    fallback_text = await repair_client.chat_completion(
                                        fork_messages,
                                        max_tokens=max_tokens,
                                    )
                                    if isinstance(fallback_text, dict):
                                        fallback_text = fallback_text.get("text", "")
                                    if fallback_text:
                                        # Выводим fallback фрагмент и пробуем продолжить до конца
                                        yield {"type": "token", "token": fallback_text, "confidence": 0.5}
                                        # Попытка продолжить оставшуюся генерацию до конца
                                        # Динамические штрафы при повторениях
                                        freq_pen_c = getattr(client.config, "frequency_penalty", 0.0)
                                        pres_pen_c = getattr(client.config, "presence_penalty", 0.0)
                                        if _should_penalize_repeats(accepted_text + (fallback_text or "")):
                                            freq_pen_c = min(2.0, (freq_pen_c or 0.0) + 0.6)
                                            pres_pen_c = min(2.0, (pres_pen_c or 0.0) + 0.3)

                                        cont_cfg = LLMConfig(
                                            endpoint=client.config.endpoint,
                                            api_key=client.config.api_key,
                                            model=client.config.model,
                                            temperature=escalated_temp,
                                            top_p=escalated_top_p,
                                            top_k=100,
                                            frequency_penalty=freq_pen_c,
                                            presence_penalty=pres_pen_c,
                                            stream=True,
                                            logprobs=True,
                                            top_logprobs=5,
                                            force_openai_streaming=True,
                                            suppress_stream_warnings=True,
                                        )
                                        cont_client = StreamingLLMClient(cont_cfg)
                                        try:
                                            cont_messages = messages + [
                                                {"role": "assistant", "content": accepted_text + fallback_text}
                                            ]
                                            async for ev in _emit_simple_token_stream(
                                                cont_client,
                                                cont_messages,
                                                max_tokens=max_tokens,
                                                per_token_threshold=per_token_threshold,
                                                enable_cutover=True,
                                                repair_temperature_factor=repair_temperature_factor,
                                                repair_top_p_cap=repair_top_p_cap,
                                                max_attempts_per_token=max_attempts,
                                                slot_retry_active=False,
                                                slot_retry_count=0,
                                                awaiting_slot_token=False,
                                                repair_state=repair_state,
                                            ):
                                                if ev.get("type") != "done":
                                                    yield ev
                                        finally:
                                            await cont_client.close()
                                except Exception:
                                    pass
                            # Финализируем верхним уровнем
                            yield {"type": "done"}
                            return
                        finally:
                            await repair_client.close()

                    # Принимаем токен в обычном порядке
                    yield {"type": "token", "token": tok, "confidence": conf}
                    accepted_tokens.append(tok)
                    accepted_text += clean_tok
                    # Снижаем окно запрета ремонта, если активно
                    if repair_state.get("no_repair_tokens_left", 0) > 0:
                        repair_state["no_repair_tokens_left"] -= 1
                    # Анти‑loop: детекция повторов с эскалацией до принудительного завершения
                    if _should_penalize_repeats(accepted_text):
                        repair_state["repeat_hits"] = repair_state.get("repeat_hits", 0) + 1
                        if repair_state["repeat_hits"] >= int(repair_state.get("repeat_hits_stop_threshold", 6)):
                            repair_state["force_stop"] = True
                            yield {"type": "done"}
                            return
                        # На ранних стадиях только отключаем ремонт на небольшое окно
                        repair_state["no_repair_tokens_left"] = max(
                            repair_state.get("no_repair_tokens_left", 0), 20
                        )
                    else:
                        # Сбрасываем счётчик при нормальном прогрессе
                        repair_state["repeat_hits"] = 0

            # Если logprobs нет — отдаём обычный контент из delta
            if not had_logprobs and getattr(ch0, "delta", None) and getattr(ch0.delta, "content", None):
                yield {"type": "token", "token": ch0.delta.content, "confidence": 0.5}
                accepted_tokens.append(ch0.delta.content)
                accepted_text += ch0.delta.content
                if repair_state.get("no_repair_tokens_left", 0) > 0:
                    repair_state["no_repair_tokens_left"] -= 1
                # Повторная проверка анти‑loop после присоединения chunk
                if _should_penalize_repeats(accepted_text):
                    repair_state["repeat_hits"] = repair_state.get("repeat_hits", 0) + 1
                    if repair_state["repeat_hits"] >= int(repair_state.get("repeat_hits_stop_threshold", 6)):
                        repair_state["force_stop"] = True
                        yield {"type": "done"}
                        return

        # основной поток завершился без ремонта
        yield {"type": "done"}
        return
    except Exception as stream_err:
        # Сервер мог закрыть поток раньше (incomplete chunked read и т.п.) —
        # при suppress_stream_warnings пытаемся продолжить генерацию до конца от текущего префикса
        if getattr(client.config, "suppress_stream_warnings", False):
            logger.debug(f"Поток OpenAI прерван в shadow repair: {stream_err}; пробуем продолжить с накопленного префикса")
            try:
                # Продолжение после разрыва: используем более «консервативные» параметры для устойчивости
                attempt_idx = 1
                base_top_p = min(repair_top_p_cap, getattr(client.config, "top_p", 1.0))
                escalated_temp = max(0.0, client.config.temperature * (repair_temperature_factor ** attempt_idx))
                escalated_top_p = max(0.05, base_top_p * (0.7 ** (attempt_idx - 1)))
                # Динамические штрафы при повторениях
                freq_pen = getattr(client.config, "frequency_penalty", 0.0)
                pres_pen = getattr(client.config, "presence_penalty", 0.0)
                if _should_penalize_repeats(accepted_text):
                    freq_pen = min(2.0, (freq_pen or 0.0) + 0.6)
                    pres_pen = min(2.0, (pres_pen or 0.0) + 0.3)

                repair_cfg = LLMConfig(
                    endpoint=client.config.endpoint,
                    api_key=client.config.api_key,
                    model=client.config.model,
                    temperature=escalated_temp,
                    top_p=escalated_top_p,
                    top_k=100,
                    frequency_penalty=freq_pen,
                    presence_penalty=pres_pen,
                    stream=True,
                    logprobs=True,
                    top_logprobs=5,
                    force_openai_streaming=True,
                    suppress_stream_warnings=True,
                )
                cont_client = StreamingLLMClient(repair_cfg)
                try:
                    fork_messages = messages + [
                        {"role": "assistant", "content": accepted_text}
                    ]
                    max_attempts = getattr(client.config, "max_attempts_per_token", None) or 3
                    async for ev in _emit_simple_token_stream(
                        cont_client,
                        fork_messages,
                        max_tokens=max_tokens,
                        per_token_threshold=per_token_threshold,
                        enable_cutover=True,
                        repair_temperature_factor=repair_temperature_factor,
                        repair_top_p_cap=repair_top_p_cap,
                        max_attempts_per_token=max_attempts,
                        slot_retry_active=False,
                        slot_retry_count=0,
                        awaiting_slot_token=False,
                    ):
                        if ev.get("type") != "done":
                            yield ev
                    return
                finally:
                    await cont_client.close()
            except Exception as cont_err:
                logger.debug(f"Не удалось продолжить после разрыва: {cont_err}; завершаем текущий ответ")
                yield {"type": "done"}
                return
        else:
            raise
    finally:
        # Закроем основной клиентский поток
        try:
            await client.close()
        except Exception:
            pass


async def _emit_simple_token_stream(
    client: StreamingLLMClient,
    messages: List[Dict[str, str]],
    *,
    max_tokens: int = 512,
    per_token_threshold: Optional[float] = None,
    enable_cutover: bool = False,
    repair_temperature_factor: float = 0.6,
    repair_top_p_cap: float = 0.6,
    max_attempts_per_token: Optional[int] = None,
    slot_retry_active: bool = False,
    slot_retry_count: int = 0,
    awaiting_slot_token: bool = False,
    slot_best_token: Optional[str] = None,
    slot_best_conf: float = -1.0,
    repair_state: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Простой генератор токенов (с logprobs, если доступны), отдаёт {type: token, token, confidence}.

    Важно:
    - Не смешиваем delta.content и logprobs в одном чанке, чтобы избежать дублирования ("заикания").
    - Перегенерация выполняется для КАЖДОГО низкоуверенного токена отдельно, с лимитом попыток
      max_attempts_per_token для одного и того же позиционного слота.
    """
    # Эффективный лимит попыток на один токен
    effective_max_attempts = max_attempts_per_token
    if effective_max_attempts is None:
        effective_max_attempts = getattr(client.config, "max_attempts_per_token", None) or 3

    params = client._prepare_openai_params(
        messages=messages,
        stream=True,
        logprobs=True,
        top_logprobs=5,
        max_tokens=max_tokens,
    )
    stream_gen = await client.openai_client.chat.completions.create(**params)

    accepted_text_local = ""
    # Инициализация состояния ремонта
    if repair_state is None:
        repair_state = {
            "repairs_done": 0,
            "max_repairs": int(getattr(client.config, "max_live_repairs", 8) or 8),
            "no_repair_tokens_left": 0,
            "repeat_hits": 0,
            "repeat_hits_stop_threshold": 6,
            "force_stop": False,
        }

    try:
        async for chunk in stream_gen:
            # Принудительное завершение при детекции зацикливания
            if repair_state.get("force_stop"):
                yield {"type": "done"}
                return
            if not getattr(chunk, "choices", None):
                continue
            ch0 = chunk.choices[0]

            had_logprobs = False
            # Токенные logprobs (предпочтительны)
            logprobs = getattr(ch0, "logprobs", None)
            if logprobs and getattr(logprobs, "content", None):
                had_logprobs = True
                for item in logprobs.content:
                    tok = getattr(item, "token", "") or ""
                    lp = getattr(item, "logprob", None)
                    if lp is None:
                        continue
                    conf = float(math.exp(lp))
                    # Очистка символов BPE для визуализации/контекста
                    clean_tok = tok.replace("Ġ", " ").replace("▁", " ")

                    # Если требуется ремонт и достигнут порог — делаем cutover
                    if enable_cutover and per_token_threshold is not None and conf < per_token_threshold and (repair_state.get("no_repair_tokens_left", 0) <= 0) and (repair_state.get("repairs_done", 0) < repair_state.get("max_repairs", 8)):
                        # Увеличиваем глобальный счетчик ремонтов
                        repair_state["repairs_done"] = repair_state.get("repairs_done", 0) + 1
                        # Определяем число попыток для ЭТОГО токена (слота)
                        current_attempt = slot_retry_count if slot_retry_active else 0
                        # Обновим лучшее предложение для этого слота (если это первый токен слота)
                        if awaiting_slot_token:
                            if conf > slot_best_conf:
                                slot_best_token = tok
                                slot_best_conf = conf
                        if current_attempt < effective_max_attempts:
                            # Покажем проблемный токен, затем откатим его визуально
                            yield {"type": "token", "token": tok, "confidence": conf}
                            yield {"type": "rollback", "count": len(clean_tok)}

                            # Стартуем ремонтный поток от префикса принятых токенов с увеличенным счётчиком
                            # Эскалация параметров по номеру попытки
                            attempt_idx = current_attempt + 1  # 1..N
                            base_top_p = min(repair_top_p_cap, getattr(client.config, "top_p", 1.0))
                            escalated_temp = max(0.0, client.config.temperature * (repair_temperature_factor ** attempt_idx))
                            escalated_top_p = max(0.05, base_top_p * (0.7 ** (attempt_idx - 1)))

                            # Динамические штрафы при повторениях
                            freq_pen = getattr(client.config, "frequency_penalty", 0.0)
                            pres_pen = getattr(client.config, "presence_penalty", 0.0)
                            if _should_penalize_repeats(accepted_text_local):
                                freq_pen = min(2.0, (freq_pen or 0.0) + 0.6)
                                pres_pen = min(2.0, (pres_pen or 0.0) + 0.3)

                            repair_cfg = LLMConfig(
                                endpoint=client.config.endpoint,
                                api_key=client.config.api_key,
                                model=client.config.model,
                                temperature=escalated_temp,
                                top_p=escalated_top_p,
                                top_k=150 if attempt_idx % 2 == 0 else 20,
                                frequency_penalty=freq_pen,
                                presence_penalty=pres_pen,
                                stream=True,
                                logprobs=True,
                                top_logprobs=5,
                                force_openai_streaming=True,
                                suppress_stream_warnings=True,
                            )
                            repair_client = StreamingLLMClient(repair_cfg)
                            try:
                                fork_messages = messages + [
                                    {"role": "assistant", "content": accepted_text_local}
                                ]
                                async for ev in _emit_simple_token_stream(
                                    repair_client,
                                    fork_messages,
                                    max_tokens=max_tokens,
                                    per_token_threshold=per_token_threshold,
                                    enable_cutover=enable_cutover,
                                    repair_temperature_factor=repair_temperature_factor,
                                    repair_top_p_cap=repair_top_p_cap,
                                    max_attempts_per_token=effective_max_attempts,
                                    slot_retry_active=True,
                                    slot_retry_count=current_attempt + 1,
                                    awaiting_slot_token=True,
                                    slot_best_token=slot_best_token,
                                    slot_best_conf=slot_best_conf,
                                    repair_state=repair_state,
                                ):
                                    if ev.get("type") != "done":
                                        yield ev
                                # Возвращаемся во внешний цикл; done эмитится верхним уровнем
                                return
                            finally:
                                await repair_client.close()
                        else:
                            # Лимит попыток исчерпан: выбираем лучший токен среди (этого и ранее наблюдавшихся)
                            best_tok = tok
                            best_conf = conf
                            if slot_best_token is not None and slot_best_conf >= best_conf:
                                best_tok = slot_best_token
                                best_conf = slot_best_conf
                            best_clean = best_tok.replace("Ġ", " ").replace("▁", " ")
                            # Эмитим выбранный токен и продолжаем генерацию из нового префикса
                            yield {"type": "token", "token": best_tok, "confidence": best_conf}
                            accepted_text_local += best_clean
                            # После принудительного принятия — отключаем ремонт на короткое окно, чтобы избежать зацикливания
                            repair_state["no_repair_tokens_left"] = max(repair_state.get("no_repair_tokens_left", 0), 20)
                            # Сброс счетчиков слота
                            slot_retry_active = False
                            slot_retry_count = 0
                            awaiting_slot_token = False
                            slot_best_token = None
                            slot_best_conf = -1.0
                            # Продолжим оставшийся ответ новым устойчивым потоком
                            attempt_idx = current_attempt + 1
                            base_top_p = min(repair_top_p_cap, getattr(client.config, "top_p", 1.0))
                            escalated_temp = max(0.0, client.config.temperature * (repair_temperature_factor ** attempt_idx))
                            escalated_top_p = max(0.05, base_top_p * (0.7 ** (attempt_idx - 1)))
                            # Динамические штрафы при повторениях
                            freq_pen_c = getattr(client.config, "frequency_penalty", 0.0)
                            pres_pen_c = getattr(client.config, "presence_penalty", 0.0)
                            if _should_penalize_repeats(accepted_text_local):
                                freq_pen_c = min(2.0, (freq_pen_c or 0.0) + 0.6)
                                pres_pen_c = min(2.0, (pres_pen_c or 0.0) + 0.3)

                            cont_cfg = LLMConfig(
                                endpoint=client.config.endpoint,
                                api_key=client.config.api_key,
                                model=client.config.model,
                                temperature=escalated_temp,
                                top_p=escalated_top_p,
                                top_k=100,
                                frequency_penalty=freq_pen_c,
                                presence_penalty=pres_pen_c,
                                stream=True,
                                logprobs=True,
                                top_logprobs=5,
                                force_openai_streaming=True,
                                suppress_stream_warnings=True,
                            )
                            cont_client = StreamingLLMClient(cont_cfg)
                            try:
                                cont_messages = messages + [
                                    {"role": "assistant", "content": accepted_text_local}
                                ]
                                async for ev in _emit_simple_token_stream(
                                    cont_client,
                                    cont_messages,
                                    max_tokens=max_tokens,
                                    per_token_threshold=per_token_threshold,
                                    enable_cutover=enable_cutover,
                                    repair_temperature_factor=repair_temperature_factor,
                                    repair_top_p_cap=repair_top_p_cap,
                                    max_attempts_per_token=effective_max_attempts,
                                    slot_retry_active=False,
                                    slot_retry_count=0,
                                    awaiting_slot_token=False,
                                    repair_state=repair_state,
                                ):
                                    if ev.get("type") != "done":
                                        yield ev
                                return
                            finally:
                                await cont_client.close()

                    # Иначе принимаем токен как есть (или лимит попыток/бюджет исчерпан)
                    yield {"type": "token", "token": tok, "confidence": conf}
                    accepted_text_local += clean_tok
                    # Слот заполнен — сбрасываем счетчик
                    slot_retry_active = False
                    slot_retry_count = 0
                    awaiting_slot_token = False
                    slot_best_token = None
                    slot_best_conf = -1.0
                    # Декоментируем окно запрета ремонта
                    if repair_state.get("no_repair_tokens_left", 0) > 0:
                        repair_state["no_repair_tokens_left"] -= 1
                    # Эвристика на повторы
                    if _should_penalize_repeats(accepted_text_local):
                        repair_state["repeat_hits"] = repair_state.get("repeat_hits", 0) + 1
                        if repair_state["repeat_hits"] >= int(repair_state.get("repeat_hits_stop_threshold", 6)):
                            repair_state["force_stop"] = True
                            yield {"type": "done"}
                            return
                        repair_state["no_repair_tokens_left"] = max(
                            repair_state.get("no_repair_tokens_left", 0), 20
                        )
                    else:
                        repair_state["repeat_hits"] = 0

            # Если logprobs нет — отдаём обычный контент
            if not had_logprobs and getattr(ch0, "delta", None) and getattr(ch0.delta, "content", None):
                content = ch0.delta.content
                # Для delta.content нет per-token метрики, считаем его принятым без попыток
                yield {"type": "token", "token": content, "confidence": 0.5}
                accepted_text_local += content
                # Проверка принудительной остановки
                if _should_penalize_repeats(accepted_text_local):
                    repair_state["repeat_hits"] = repair_state.get("repeat_hits", 0) + 1
                    if repair_state["repeat_hits"] >= int(repair_state.get("repeat_hits_stop_threshold", 6)):
                        repair_state["force_stop"] = True
                        yield {"type": "done"}
                        return
                slot_retry_active = False
                slot_retry_count = 0
                awaiting_slot_token = False
        # Попытка завершить ответ до конца при раннем разрыве потока
        if getattr(client.config, "suppress_stream_warnings", False):
            logger.debug(f"Ремонтный поток прерван: {stream_err}; пробуем продолжить с накопленного префикса")
            try:
                # Продолжим генерацию от уже набранного текста
                attempt_idx = 1
                base_top_p = min(repair_top_p_cap, getattr(client.config, "top_p", 1.0))
                escalated_temp = max(0.0, client.config.temperature * (repair_temperature_factor ** attempt_idx))
                escalated_top_p = max(0.05, base_top_p * (0.7 ** (attempt_idx - 1)))
                # Динамические штрафы при повторениях
                freq_pen = getattr(client.config, "frequency_penalty", 0.0)
                pres_pen = getattr(client.config, "presence_penalty", 0.0)
                if _should_penalize_repeats(accepted_text_local):
                    freq_pen = min(2.0, (freq_pen or 0.0) + 0.6)
                    pres_pen = min(2.0, (pres_pen or 0.0) + 0.3)

                repair_cfg = LLMConfig(
                    endpoint=client.config.endpoint,
                    api_key=client.config.api_key,
                    model=client.config.model,
                    temperature=escalated_temp,
                    top_p=escalated_top_p,
                    top_k=100,
                    frequency_penalty=freq_pen,
                    presence_penalty=pres_pen,
                    stream=True,
                    logprobs=True,
                    top_logprobs=5,
                    force_openai_streaming=True,
                    suppress_stream_warnings=True,
                )
                cont_client = StreamingLLMClient(repair_cfg)
                try:
                    fork_messages = messages + [
                        {"role": "assistant", "content": accepted_text_local}
                    ]
                    async for ev in _emit_simple_token_stream(
                        cont_client,
                        fork_messages,
                        max_tokens=max_tokens,
                        per_token_threshold=per_token_threshold,
                        enable_cutover=enable_cutover,
                        repair_temperature_factor=repair_temperature_factor,
                        repair_top_p_cap=repair_top_p_cap,
                        max_attempts_per_token=effective_max_attempts,
                        slot_retry_active=False,
                        slot_retry_count=0,
                        awaiting_slot_token=False,
                        repair_state=repair_state,
                    ):
                        if ev.get("type") != "done":
                            yield ev
                    # По окончании дочернего потока — сигнализируем завершение
                    yield {"type": "done"}
                    return
                finally:
                    await cont_client.close()
            except Exception as cont_err:
                logger.debug(f"Не удалось продолжить после разрыва: {cont_err}; завершаем текущий ответ")
        else:
            raise
    finally:
        try:
            await client.close()
        except Exception:
            pass
