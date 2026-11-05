# Football AI Coach (P01)

Заглушка. Football AI Coach превращает запись матча КПЛ в каталог событий с таймкодами. Полное описание: `CLAUDE.md`, ТЗ: `docs/TZ.md`.

## Быстрый старт (Windows, PowerShell)

1. Создать окружение: `python -m venv .venv` и активировать `.venv\Scripts\Activate.ps1`.
2. Установить пакет с dev-зависимостями: `pip install -e ".[dev]"`.
3. Проверить качество: `python scripts/check.py` (ruff, ruff format, mypy, pytest).

> Хук `mypy` в `pre-commit` запускает `.venv\Scripts\python.exe`, поэтому окружение должно лежать именно в папке `.venv` в корне репозитория (Windows). Установка хуков: `pre-commit install` (ставит и хук проверки сообщения коммита).
