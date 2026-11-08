# SKILLS.md — процессы и автоматизация

Команды и порядок работы для всех агентов. Подключено из `CLAUDE.md`.

> **Статус команд:** репозиторий создаётся с нуля, поэтому команды ниже описывают целевой вид. Среда разработки Windows (PowerShell) без `make`, `uv`, WSL и Docker (см. D-016 в `MEMORY.md`); общий контроль качества это `python scripts/check.py`. Если команда изменилась или добавилась, обнови этот файл в том же коммите.

## 1. Окружения

Нужны два окружения: основное (PyTorch, Whisper, Streamlit) и отдельное для извлечения признаков ResNet-TF2 (TensorFlow 2). Они конфликтуют по зависимостям, смешивать их нельзя.

```bash
# основное окружение
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
cp .env.example .env        # затем заполнить значения вручную; секреты не коммитить

# окружение признаков (TensorFlow), только для стадии features-tf
python3.11 -m venv .venv-tf && source .venv-tf/bin/activate
pip install -e ".[features-tf]"
```

На Windows (PowerShell) основное окружение создаётся так: `python -m venv .venv`, затем `.venv\Scripts\Activate.ps1` и `pip install -e ".[dev]"`.

## 2. Основные команды

| Команда | Что делает |
| --- | --- |
| `python scripts/check.py` | полный контроль качества: `ruff check`, `ruff format --check`, `mypy`, `pytest` (запускаются все четыре) |
| `ruff check .` и `ruff format --check .` | линтер и проверка формата |
| `ruff format .` и `ruff check --fix .` | автоформатирование и автоисправления |
| `mypy` | типы (`files = ["src", "tests"]` из `pyproject.toml`) |
| `pytest` | тесты с отчётом о покрытии (порог `fail_under = 80`) |
| `pytest <путь> --no-cov` | подмножество тестов: без `--no-cov` покрытие считается по неполному запуску и `fail_under` роняет его, хотя тесты зелёные |
| `pytest -m integration` | интеграционные тесты (нужен `ffmpeg`, сети нет); маркеры появятся вместе с тестами |
| `mlflow ui --backend-store-uri file:./mlruns` | просмотр экспериментов |

## 3. Запуск пайплайна

Каждая стадия отдельная команда, пропускает готовое и пишет файл в каталог матча (`data/<match_id>/`; для стадии `scoreboard` это каталог видео). Пути и пороги берутся из `.env` и `configs/`, не из аргументов с жёсткими значениями.

```bash
python -m aicoach.cli ingest     --url "<YOUTUBE_URL>" --match-id <ID>
python -m aicoach.cli scoreboard --video <PATH_TO_SRC_MP4> [--force]   # часы и счёт табло → scoreboard.json рядом с видео
python -m aicoach.cli scoreboard-calibrate --video <PATH_TO_SRC_MP4> --workdir scratch/<ID>_calibration   # шаг 1: эталон табло и листы кластеров цифр
python -m aicoach.cli scoreboard-calibrate --video <PATH_TO_SRC_MP4> --workdir scratch/<ID>_calibration --labels labels.json   # шаг 2: образцы цифр
python -m aicoach.cli plates     --video <PATH_TO_SRC_MP4> [--match-id <ID>] [--sample-fps 2] [--cpu] [--force]   # плашки замен, голов, карточек → plates.json и plate_crops/ рядом с видео (нужен scoreboard.json)
python -m aicoach.cli features   --match-id <ID>          # в окружении .venv-tf для ветки A
python -m aicoach.cli spot       --match-id <ID> --branch a   # или --branch b
python -m aicoach.cli speech     --match-id <ID>          # VAD, Whisper, лексикон
python -m aicoach.cli fuse       --match-id <ID>          # объединение и запись в SQLite
python -m aicoach.cli eval       --match-id <ID> --split half2   # метрики, запуск в MLflow
python -m aicoach.cli export-clip --match-id <ID> --event-id <EVENT_ID>
```

Весь пайплайн подряд: `python -m aicoach.cli run --match-id <ID>`. Статус стадий пишется в `status.json` каталога матча (для стадии `scoreboard` это каталог видео).

**Стадия `scoreboard` (D03).** Видео передаётся только аргументом `--video`; результат (`scoreboard.json`, `status.json`) и образцы (`scoreboard_templates/`) лежат в каталоге видео, а не в репозитории. Без образцов стадия падает с подсказкой. Калибровка в два шага: шаг 1 строит эталон табло и пишет в `--workdir` листы `sheet_clock.png`, `sheet_score.png`, `sheet_added.png` (кластеры символов с номерами); человек смотрит листы и пишет файл подписей `{"clock": {"0": "colon", "1": "0"}, "score": {...}, "added": {...}}` (номер кластера → `0`..`9`, `colon`, `plus`); шаг 2 (`--labels`) режет образцы. Декодирование идёт на видеокарте, если она есть (`--cpu` отключает, для AV1 это в десятки раз медленнее); образцы и прогон должны идти в одном режиме, иначе стадия остановится. Результат кэшируется по ключу (видео, параметры, режим, образцы); `--force` читает видео заново.

**Стадия `plates` (D04, шаг 1).** Нужен `scoreboard.json` рядом с видео (без него понятная ошибка с подсказкой запустить табло) и тот же режим декодирования, что у табло (иначе остановка). Читает видео 2 раза в секунду (`--sample-fps`; `--cpu` декодирует на процессоре, `--match-id` по умолчанию имя каталога видео), находит плашки замены, гола и карточки по цвету и форме, склеивает кадры в плашки, определяет команду по логотипу и смене счёта. Фамилии не читаются (это следующий шаг). Пишет `plates.json`, `status.json` и `plate_crops/` (вырезка и логотип каждой плашки, вырезки отброшенных кандидатов) в каталог видео; всё вне git. Кэш по ключу (видео, версия `scoreboard.json`, режим декодирования, параметры и раскладка); `--force` читает видео заново. На матче `match01` стадия идёт около 9 минут на видеокарте.

Тест по реальному матчу пропускается, пока не задана переменная окружения: `$env:AICOACH_REAL_MATCH_VIDEO = "<PATH_TO_SRC_MP4>"; pytest tests/real_data --no-cov` (нужны готовый `scoreboard.json` и, для плашек, `plates.json` и `plates_reference.json` с контрольными временами владельца рядом с видео).

## 4. Качество и тесты

```bash
ruff check .  &&  ruff format --check .
mypy
pytest -m "unit or data_validation"                  # быстрый набор
pytest -m integration                                # интеграционные
pytest --cov=src/aicoach --cov-report=term-missing   # покрытие
pre-commit run --all-files
```

Подробности о структуре, маркерах и порогах покрытия в `TESTING.md`.

## 5. MLflow

```bash
export MLFLOW_TRACKING_URI="file:./mlruns"        # значение по умолчанию задаётся в Settings
mlflow ui --backend-store-uri "$MLFLOW_TRACKING_URI"
```

Запуски создаются только командами `eval` и обучения. Состав параметров, метрик и тегов описан в `CLAUDE.md`, раздел 7.

## 6. Контейнеры

> **Проверить до использования:** есть ли на сервере Docker и `nvidia-container-toolkit`. Если нет, работай через `venv` из раздела 1. Контейнеры не обязательны для сдачи.

```bash
docker build -t aicoach:dev -f docker/Dockerfile .
docker build -t aicoach-features-tf:dev -f docker/Dockerfile.features-tf .

# проверка качества в контейнере
docker run --rm -v "$PWD:/app" aicoach:dev python scripts/check.py

# стадия с видеокартой; данные и конфиги монтируются, секреты берутся из .env
docker run --rm --gpus all --env-file .env -v "$PWD/data:/app/data" \
  aicoach:dev python -m aicoach.cli spot --match-id <ID> --branch a

# извлечение признаков в образе с TensorFlow
docker run --rm --gpus all --env-file .env -v "$PWD/data:/app/data" \
  aicoach-features-tf:dev python -m aicoach.cli features --match-id <ID>
```

В образы не копируются `.env`, `data/`, `mlruns/` и видео (проверь `.dockerignore`).

## 7. Post-Task Workflow (обязательно)

Применяется к любой задаче, которая меняет файлы в репозитории. Чисто справочные ответы без изменений файлов этот процесс не запускают.

1. **Контроль качества.** Выполни `python scripts/check.py`. Если не зелёный, чини причину, а не проверку. Если починить не получается, не коммить: оставь изменения в рабочей директории и опиши блокер пользователю.
2. **Обнови `MEMORY.md`** (обязательно, если появилось что-то из списка):
   - новое решение: запись `D-xxx` с причиной и статусом;
   - найденная проблема или закрытый вопрос: раздел «Известные проблемы»;
   - подход, который не сработал: запись `X-xxx` с числами и идентификатором запуска MLflow;
   - смена фокуса: раздел «Текущий фокус».
   Если ничего из этого не появилось, оставь файл как есть, но в отчёте напиши «MEMORY.md без изменений, потому что...».
3. **Обнови `SKILLS.md`**, если изменились команды, `scripts/check.py` или окружения. Файлы `CLAUDE.md`, `GIT.md`, `TESTING.md` без согласия пользователя не меняются: предложи правку в чате.
4. **Закоммить по правилам `GIT.md`:** отдельная ветка нужного формата; логически связанные изменения в одном коммите; формат Conventional Commits; код, тесты и обновления `MEMORY.md` и `SKILLS.md`, которые относятся к этому изменению, идут в одном коммите. Чисто «памятные» изменения идут как `docs(memory): ...`.
5. **Не делай `push` и `merge`** без явного «да» в чате.
6. **Отчёт пользователю на русском** (коротко): что сделано и зачем; хэши коммитов; результат `python scripts/check.py`; что записано в `MEMORY.md`; что дальше или что блокирует.

Если задача прервана и работа не доведена до рабочего состояния: не коммить, оставь изменения в рабочей директории и опиши их состояние в отчёте.
