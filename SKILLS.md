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

**Ключ Gemini (D-057).** Ключ лежит только в файле `.env` в корне репозитория; `.env` и `*.env` в `.gitignore`, в репозитории только шаблон `.env.example` с пустым `GEMINI_API_KEY=`. Задать ключ: `Copy-Item .env.example .env`, затем открыть `.env` в редакторе и вписать значение после `GEMINI_API_KEY=` (без пробелов вокруг `=`, кавычки не нужны). Код читает ключ функцией `aicoach.config.read_gemini_api_key()`: читается только файл (переменная оболочки с тем же именем не учитывается, окружение процесса не меняется); нет файла, нет ключа или он пустой дают `ConfigError` с подсказкой. Агенту чтение и правка `.env` запрещены в `.claude/settings.json`.

**Поиск секретов (D-057).** Хук `detect-secrets` (pre-commit) сверяет изменённые файлы с `.secrets.baseline`. Фиктивный ключ в тесте помечается комментарием `# pragma: allowlist secret` в той же строке. Если хук сработал на ложном срабатывании, которое нельзя пометить pragma, обновить baseline: `pip install detect-secrets==1.5.0` во временном окружении (не в `.venv`), `detect-secrets scan > .secrets.baseline`, `detect-secrets audit .secrets.baseline` (каждое срабатывание разобрать), `git add .secrets.baseline`. Проверка всей истории git отложена до публикации репозитория.

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
python -m aicoach.cli spotting-eval --predictions <PATH_TO_TDEED_JSON> [--ground-truth data_labels/match01_ground_truth.json] [--half 1] [--thresholds 0.05 0.1 0.2 0.3 0.5] [--fps 25] [--radius-s 3] [--output <FILE>]   # оценка T-DEED на 1-м тайме → таблица и eval_<имя>_half1.json рядом с файлом
python -m aicoach.cli features   --match-id <ID>          # в окружении .venv-tf для ветки A
python -m aicoach.cli spot       --match-id <ID> --branch a   # или --branch b
python -m aicoach.cli speech     --match-id <ID>          # VAD, Whisper, лексикон
python -m aicoach.cli fuse       --video <PATH_TO_SRC_MP4> [--predictions <PATH_TO_TDEED_JSON>] [--config <FILE.json>] [--db <FILE>] [--match-id <ID>] [--force] [--discard-edits]   # слияние правилами: события в aicoach.db и candidates.json рядом с видео
python -m aicoach.cli run        --video <PATH_TO_SRC_MP4> [--predictions <PATH_TO_TDEED_JSON>] [--config <FILE.json>] [--match-id <ID>] [--cpu] [--force] [--discard-edits]   # scoreboard, plates, fuse по порядку (модели в нём нет)
python -m aicoach.cli ui         --video <PATH_TO_SRC_MP4> --youtube-id <ID> [--match-id <ID>] [--db <FILE>]   # события в браузере (Streamlit): просмотр и ручные правки (добавить, сдвинуть, сменить тип, удалить, отменить последнюю; пишутся в журнал `edits`, проверка времени по `candidates.json` и `scoreboard.json` рядом с видео); ID видео только параметр; id с «-» в начале писать как --youtube-id=<ID>; путь `--video` передаётся интерфейсу для экспорта клипа (кнопка «Экспорт клипа» в боковой панели)
python -m aicoach.cli eval       --match-id <ID> --split half2   # метрики, запуск в MLflow
python -m aicoach.cli clip       --video <PATH_TO_SRC_MP4> --event-id <EVENT_ID> [--before 8] [--after 12] [--match-id <ID>] [--db <FILE>]   # клип события (время из events_current, правки учтены) → clips/<id>_<тип>_t<мс>ms_b<до>_a<после>.mp4 рядом с видео; перекодирование H.264/AAC (D-056); готовый файл не режется заново; нужен ffmpeg в PATH
```

Весь пайплайн подряд (без модели, она идёт отдельно в Colab): `python -m aicoach.cli run --video <PATH_TO_SRC_MP4>`; стадии идут по порядку scoreboard, plates, fuse, каждая использует свой кэш, при сбое выполнение останавливается, а в `status.json` запись `run` называет стадию и причину. `--force` в `run` пересчитывает все стадии (видео читается дважды, это долго) и при наличии правок пользователя остановится только на третьей стадии, если не указан `--discard-edits`. Файл `--predictions` и `--config` проверяются до запуска первой стадии. Любая ошибка стадии выходит как `error: stage <имя> failed: ...` с кодом 2. Статус стадий пишется в `status.json` каталога матча (для стадий `scoreboard`, `plates` и `fuse` это каталог видео).

**Стадия `scoreboard` (D03).** Видео передаётся только аргументом `--video`; результат (`scoreboard.json`, `status.json`) и образцы (`scoreboard_templates/`) лежат в каталоге видео, а не в репозитории. Без образцов стадия падает с подсказкой. Калибровка в два шага: шаг 1 строит эталон табло и пишет в `--workdir` листы `sheet_clock.png`, `sheet_score.png`, `sheet_added.png` (кластеры символов с номерами); человек смотрит листы и пишет файл подписей `{"clock": {"0": "colon", "1": "0"}, "score": {...}, "added": {...}}` (номер кластера → `0`..`9`, `colon`, `plus`); шаг 2 (`--labels`) режет образцы. Декодирование идёт на видеокарте, если она есть (`--cpu` отключает, для AV1 это в десятки раз медленнее); образцы и прогон должны идти в одном режиме, иначе стадия остановится. Результат кэшируется по ключу (видео, параметры, режим, образцы); `--force` читает видео заново.

**Стадия `plates` (D04, шаг 1).** Нужен `scoreboard.json` рядом с видео (без него понятная ошибка с подсказкой запустить табло) и тот же режим декодирования, что у табло (иначе остановка). Читает видео 2 раза в секунду (`--sample-fps`; `--cpu` декодирует на процессоре, `--match-id` по умолчанию имя каталога видео), находит плашки замены, гола и карточки по цвету и форме, склеивает кадры в плашки, определяет команду по логотипу и смене счёта. Фамилии не читаются (это следующий шаг). Пишет `plates.json`, `status.json` и `plate_crops/` (вырезка и логотип каждой плашки, вырезки отброшенных кандидатов) в каталог видео; всё вне git. Кэш по ключу (видео, версия `scoreboard.json`, режим декодирования, параметры и раскладка); `--force` читает видео заново. На матче `match01` стадия идёт около 9 минут на видеокарте.

**Стадия `fuse` (D09).** Читает `scoreboard.json`, `plates.json` и необязательный файл T-DEED (`--predictions`) из каталога видео; видео не читает. Правила и стартовые параметры: D-047 в `MEMORY.md`; параметры лежат в `FuseConfig`, нужное переопределяется JSON-файлом `--config` (неизвестное поле или неверное значение даёт ошибку конфигурации). Без `--predictions` стадия работает в режиме «без модели»: голы, замены и фолы по плашкам карточек, в `candidates.json` и в логе предупреждение. Пишет события в SQLite `aicoach.db` (таблица `predictions`, новый `run_id` на каждый запуск, D-028) и `candidates.json` (события с источником и доказательствами, отброшенные, предупреждения, границы таймов) и `status.json` в каталог видео. Кэш по ключу (версии трёх входных файлов, параметры, версия схемы) и по тому, что запуск из `candidates.json` всё ещё последний в базе; `--force` считает заново. **Защита правок (D-028):** если у матча есть действующие правки пользователя, а запуск записал бы новый `run_id`, стадия отказывается; после экспорта проверенных меток запускать с `--discard-edits` (журнал правок остаётся, но правки предсказаний пропадают из текущего вида). Пример: `python -m aicoach.cli fuse --video <PATH_TO_SRC_MP4> --predictions <PATH_TO_TDEED_JSON>`.

**Экспорт клипа (D13, FR-7).** `clip` берёт время события из `events_current` (правки учтены), вырезает окно `[t_v − до, t_v + после]`, подрезанное по краям видео, и перекодирует его в H.264/AAC (копия потока не годится: исходник AV1 режется по ключевым кадрам). Клип лежит в `clips/` рядом с видео; имя содержит id, тип, время события в мс и поля, поэтому готовый файл отдаётся без ffmpeg, а сдвинутое событие получает новый клип. В интерфейсе то же делает кнопка «Экспорт клипа» в боковой панели. Подробности: D-056.

**Ветка B, T-DEED (D08).** Модель запускается в Colab неизменённым `inference.py` отдельным процессом; код T-DEED в репозиторий не копируется и не импортируется (D-046). Видео перед запуском перекодируется в 25 к/с без обрезки (шаг модели задан в кадрах, D-042); чтобы строки порогов 0,05 и 0,1 отличались от 0,2, `inference.py` запускается с `--inference_threshold 0.05` (D-045). Файл результата `results_inference.json` кладётся рядом с видео как `tdeed_<имя>.json` (вне git). Команда `spotting-eval` читает его и эталон, переводит `frame / fps` в секунды, оставляет шесть классов из 17 (D-043), сливает дубли ближе `--radius-s` (D-044) и для каждого порога считает Recall и Precision при ±5 и ±10 с, число найденных, медиану и p90 ошибки времени и bootstrap 95 % только по 1-му тайму; `--half 2` отклоняется, 2-й тайм зарезервирован для финальной оценки (D-045). Это не запуск MLflow, цифры в отчёт не идут. Эталон по умолчанию задан путём относительно текущего каталога, поэтому команду запускают из корня репозитория, иначе `--ground-truth`. Пример: `python -m aicoach.cli spotting-eval --predictions <PATH_TO_TDEED_JSON>`.

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
