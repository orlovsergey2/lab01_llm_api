
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import os

from test_data import SUMMARIZE_TEXT, STRUCTURE_TEXT


# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------
@dataclass
class Config:
    api_key: str
    base_url: str
    models: list[str]
    out_dir: Path = Path("results")
    system_prompt: str = (
        "Отвечай сразу и по существу на русском языке. "
        "Не показывай рассуждения, сразу пиши финальный ответ."
    )
    temperature: float = 0.7
    max_retries: int = 3
    retry_delay: float = 1.5
    warmup_tokens: int = 1


def load_config(cli_models: list[str] | None = None) -> Config:
    """Собирает конфигурацию из .env + аргументов CLI."""
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(env_path)

    api_key = os.getenv("LMSTUDIO_API_KEY", "")
    base_url = os.getenv("LMSTUDIO_BASE_URL", "")
    models_env = os.getenv("LMSTUDIO_MODELS", "")

    if cli_models:
        models = cli_models
    else:
        models = [m.strip() for m in models_env.split(",") if m.strip()]

    missing = []
    if not api_key:
        missing.append("LMSTUDIO_API_KEY")
    if not base_url:
        missing.append("LMSTUDIO_BASE_URL")
    if not models:
        missing.append("LMSTUDIO_MODELS (или --models)")

    if missing:
        raise SystemExit(
            "Не заданы параметры:\n  " + "\n  ".join(missing) +
            f"\nПроверьте файл: {env_path}"
        )

    return Config(api_key=api_key, base_url=base_url, models=models)


# ---------------------------------------------------------------------------
# Тестовый набор (10 типов задач)
# ---------------------------------------------------------------------------
PROMPTS: list[str] = [
    # 1. Объяснение
    "Объясни студенту 4 курса разницу между хешированием и симметричным "
    "шифрованием. Не более 120 слов.",

    # 2. Классификация
    "Определи тип события («После 15 неудачных попыток входа с одного IP "
    "выполнен успешный вход администратора»): норма / подозрительно / "
    "критично и объясни.",

    # 3. Суммаризация (текст из test_data.py)
    (
        "Сожми приведённый ниже текст об информационной безопасности "
        "ровно в 5 тезисов. Каждый тезис — одно предложение, начиная "
        "с номера и точки. Никаких вступлений и заключений.\n\n"
        f"Текст:\n\"{SUMMARIZE_TEXT}\""
    ),

    # 4. Структурирование (текст из test_data.py)
    (
        "Извлеки из приведённого ниже текста сущности и верни ТОЛЬКО "
        "валидный JSON без markdown-обёрток и без пояснений. Схема:\n"
        "{\n"
        '  "ip": "string",\n'
        '  "user": "string",\n'
        '  "timestamp": "string (ISO 8601)",\n'
        '  "event_type": "string"\n'
        "}\n\n"
        f"Текст:\n\"{STRUCTURE_TEXT}\""
    ),

    # 5. Генерация кода
    "Напиши Python-функцию проверки SHA-256 хеша строки (на входе строка и "
    "ожидаемый хеш, на выходе True/False).",

    # 6. Отладка
    "Найди ошибку в Python-функции find_duplicates (вызов dups.add(item) "
    "для списка) и исправь её.",

    # 7. Аналитика
    "Предложи план первичного анализа подозрительного входа в "
    "корпоративную систему. 5 пунктов.",

    # 8. Информационная безопасность
    "Назови 5 рисков использования внешней LLM для анализа внутренних "
    "документов организации.",

    # 9. Деловой текст
    "Сформулируй краткое уведомление сотрудникам о запрете передачи "
    "паролей в AI-сервисах.",

    # 10. Жёсткий формат
    "Ответь только таблицей Markdown с колонками риск | вероятность | "
    "ущерб | мера защиты. Оцени 4 риска AI-агента с доступом к почте.",
]


# ---------------------------------------------------------------------------
# Результат одного вызова
# ---------------------------------------------------------------------------
@dataclass
class CallResult:
    model: str
    prompt_id: int
    run_id: int
    latency: float | None
    answer: str | None
    error: str | None
    answer_len: int = 0
    attempts: int = 1
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


# ---------------------------------------------------------------------------
# Проверка JSON-ответа (для промпта 4)
# ---------------------------------------------------------------------------
REQUIRED_JSON_FIELDS = {"ip", "user", "timestamp", "event_type"}


def validate_json_answer(answer: str) -> tuple[bool, str]:
    """
    Пытается распарсить ответ как JSON и проверить обязательные поля.
    Возвращает (валидность, сообщение).
    """
    if not answer:
        return False, "пустой ответ"

    cleaned = answer.strip()
    # убираем markdown-обёртку ```json ... ```
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return False, f"невалидный JSON: {exc}"

    if not isinstance(parsed, dict):
        return False, "JSON не является объектом"

    missing = REQUIRED_JSON_FIELDS - set(parsed.keys())
    if missing:
        return False, f"нет полей: {', '.join(sorted(missing))}"
    return True, "валидный JSON со всеми полями"


# ---------------------------------------------------------------------------
# Клиент LLM
# ---------------------------------------------------------------------------
class LLMClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

    def warmup(self, model: str) -> None:
        try:
            self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=self.cfg.warmup_tokens,
            )
        except Exception as exc:
            print(f"  [warmup] {model}: {exc}", file=sys.stderr)

    def ask(self, model: str, prompt: str) -> tuple[str, float, int]:
        """
        Возвращает (ответ, latency, число попыток).
        Повторяет запрос при ошибке до cfg.max_retries раз.
        """
        last_exc: Exception | None = None
        for attempt in range(1, self.cfg.max_retries + 1):
            t0 = time.perf_counter()
            try:
                resp = self._client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": self.cfg.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.cfg.temperature,
                )
                dt = time.perf_counter() - t0
                text = resp.choices[0].message.content or ""
                return text, dt, attempt
            except Exception as exc:
                last_exc = exc
                if attempt < self.cfg.max_retries:
                    time.sleep(self.cfg.retry_delay * attempt)
        raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Прогон одной модели
# ---------------------------------------------------------------------------
def run_one_model(client: LLMClient, cfg: Config, model: str) -> list[CallResult]:
    print(f"\n=== Модель: {model} ===")
    client.warmup(model)

    results: list[CallResult] = []
    for pid, prompt in enumerate(PROMPTS, start=1):
        try:
            answer, latency, attempts = client.ask(model, prompt)

            # Дополнительная проверка для 4-го промпта
            extra = ""
            if pid == 4:
                valid, msg = validate_json_answer(answer)
                extra = f" [JSON: {'OK' if valid else 'ОШИБКА — ' + msg}]"

            results.append(CallResult(
                model=model, prompt_id=pid, run_id=1,
                latency=round(latency, 3), answer=answer, error=None,
                answer_len=len(answer), attempts=attempts,
            ))
            print(f"  [{pid:>2}/{len(PROMPTS)}] {latency:7.2f} с  OK "
                  f"(попыток: {attempts}, длина: {len(answer)}){extra}")
        except Exception as exc:
            results.append(CallResult(
                model=model, prompt_id=pid, run_id=1,
                latency=None, answer=None, error=str(exc),
            ))
            print(f"  [{pid:>2}/{len(PROMPTS)}]   ---   ОШИБКА: {exc}")

    ok = [r for r in results if r.error is None and r.latency is not None]
    if ok:
        lats = [r.latency for r in ok]  # type: ignore[misc]
        print(f"  Итого: успешно {len(ok)}/{len(PROMPTS)}, "
              f"avg={statistics.mean(lats):.2f} с, "
              f"min={min(lats):.2f} с, max={max(lats):.2f} с")
    else:
        print(f"  Итого: все {len(PROMPTS)} запросов завершились ошибкой")
    return results


# ---------------------------------------------------------------------------
# Сохранение
# ---------------------------------------------------------------------------
def _safe(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").replace(":", "_")


def save_per_model(rows: list[CallResult], out_dir: Path, model: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    base = out_dir / f"{_safe(model)}_{stamp}"
    fields = ["model", "prompt_id", "run_id", "latency",
              "answer_len", "attempts", "answer", "error", "timestamp"]

    with base.with_suffix(".csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))

    with base.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in rows], f, ensure_ascii=False, indent=2)

    print(f"  Сохранено: {base}.csv / .json")
    return base


def save_summary(all_rows: list[CallResult], out_dir: Path) -> None:
    """Сводная таблица по всем моделям + общий CSV/JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")

    fields = ["model", "prompt_id", "run_id", "latency",
              "answer_len", "attempts", "answer", "error", "timestamp"]
    combined = out_dir / f"summary_all_{stamp}"
    with combined.with_suffix(".csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow(asdict(r))
    with combined.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in all_rows], f, ensure_ascii=False, indent=2)

    # Агрегаты по моделям
    per_model: dict[str, dict] = {}
    for m in {r.model for r in all_rows}:
        rows = [r for r in all_rows if r.model == m]
        ok = [r for r in rows if r.error is None and r.latency is not None]
        lats = [r.latency for r in ok]  # type: ignore[misc]
        per_model[m] = {
            "total": len(rows),
            "ok": len(ok),
            "errors": len(rows) - len(ok),
            "avg_latency": round(statistics.mean(lats), 3) if lats else None,
            "min_latency": round(min(lats), 3) if lats else None,
            "max_latency": round(max(lats), 3) if lats else None,
            "avg_answer_len": round(statistics.mean(r.answer_len for r in ok), 1) if ok else None,
        }

    agg_path = out_dir / f"summary_by_model_{stamp}.json"
    with agg_path.open("w", encoding="utf-8") as f:
        json.dump(per_model, f, ensure_ascii=False, indent=2)

    # Красивый вывод в консоль
    print("\n" + "=" * 72)
    print("Сводка по моделям")
    print("=" * 72)
    header = f"{'model':<38}{'ok':>4}{'err':>5}{'avg,s':>9}{'min,s':>8}{'max,s':>8}"
    print(header)
    print("-" * len(header))
    for m, s in per_model.items():
        print(f"{m:<38}{s['ok']:>4}{s['errors']:>5}"
              f"{(s['avg_latency'] or 0):>9.2f}"
              f"{(s['min_latency'] or 0):>8.2f}"
              f"{(s['max_latency'] or 0):>8.2f}")
    print(f"\nОбщий CSV: {combined}.csv")
    print(f"Агрегаты:  {agg_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ЛР1: сравнение LLM через API")
    p.add_argument("--models", nargs="+", default=None,
                   help="Список моделей (переопределяет LMSTUDIO_MODELS)")
    p.add_argument("--out", default="results", type=Path,
                   help="Папка для результатов (по умолчанию results/)")
    p.add_argument("--runs", default=1, type=int,
                   help="Число повторов всего набора (для оценки стабильности)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    cfg = load_config(cli_models=args.models)
    cfg.out_dir = args.out

    print(f"Base URL:  {cfg.base_url}")
    print(f"Модели:    {', '.join(cfg.models)}")
    print(f"Запросов:  {len(PROMPTS)} × {args.runs} прогон(ов)")
    print(f"Всего:     {len(cfg.models) * len(PROMPTS) * args.runs} вызовов")

    client = LLMClient(cfg)
    all_rows: list[CallResult] = []

    for run_id in range(1, args.runs + 1):
        if args.runs > 1:
            print(f"\n########## Прогон {run_id}/{args.runs} ##########")
        for model in cfg.models:
            rows = run_one_model(client, cfg, model)
            for r in rows:
                r.run_id = run_id
            all_rows.extend(rows)
            save_per_model(rows, cfg.out_dir, model)

    save_summary(all_rows, cfg.out_dir)
    print("\nЭксперимент завершён.")


if __name__ == "__main__":
    main()