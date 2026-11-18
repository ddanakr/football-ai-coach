"""Russian texts of the viewer (`aicoach.ui`) and of user-facing configuration errors."""

from aicoach.domain.events import EventType

TITLE = "Football AI Coach: события матча"

TYPE_LABELS: dict[EventType, str] = {
    EventType.GOAL: "Гол",
    EventType.SHOT: "Удар",
    EventType.CORNER: "Угловой",
    EventType.FOUL: "Фол",
    EventType.SUBST: "Замена",
}
SOURCE_LABELS: dict[str, str] = {
    "plates": "табло и плашки",
    "model": "модель",
    "merge": "модель и плашка",
    "speech": "речь",
}
TEAM_LABELS: dict[str, str] = {"home": "хозяева", "away": "гости"}
STATUS_LABELS: dict[str, str] = {
    "disallowed": "гол отменён",
    "manual": "добавлено вручную",
}
UNKNOWN = "—"

SIDEBAR_FILTERS = "Фильтры"
FILTER_TYPES = "Типы событий"
FILTER_HALF = "Тайм"
HALF_ALL = "Все"
HALF_TEXT = "{half}-й тайм"
FILTER_MIN_SCORE = "Минимальная уверенность"
FILTER_MIN_SCORE_HELP = "События без оценки (плашки, ручные) остаются при любом значении."

COLUMN_TIME = "Время"
COLUMN_TYPE = "Тип"
COLUMN_HALF = "Тайм"
COLUMN_SCORE = "Уверенность"
COLUMN_SOURCE = "Источник"
COLUMN_TEAM = "Команда"
COLUMN_EDIT = "Правка"
MARK_ADDED = "добавлено"
MARK_EDITED = "изменено"

TABLE_HINT = "Выберите строку, чтобы открыть момент в плеере."
SHOWN = "Показано {shown} из {total}"
PLAYING = "Сейчас: {type} в {time}"
LINK_TEXT = "Открыть на YouTube с этого момента"
TIME_NOTE = "Время отсчитывается от начала файла видео."

NO_DB = (
    "Базы событий нет: файл {path} не найден. "
    "Сначала выполните `python -m aicoach.cli run --video <путь к видео>`."
)
DB_ERROR = "Не удалось прочитать базу {path}: {error}"
NO_EVENTS = (
    "В базе нет событий матча «{match_id}». Проверьте параметр `--match-id` "
    "или выполните `python -m aicoach.cli run --video <путь к видео>`."
)
NOTHING_MATCHES = "Фильтры ничего не оставили. Ослабьте фильтры в боковой панели."
BAD_ARGS = (
    "Неверные параметры запуска: {error}. "
    "Запускайте интерфейс командой `python -m aicoach.cli ui --video <путь> --youtube-id <id>`."
)

# Manual edits (D12b). Messages with {braces} are filled by `aicoach.ui.edits` and the page.
EDIT_HEADER = "Правка событий"
EDIT_SELECTED = "Выбрано: {type} в {time}"
EDIT_PICK_FIRST = "Выберите строку в таблице, чтобы сдвинуть, переименовать или удалить событие."
EDIT_TIME_LABEL = "Новое время (мм:сс)"
EDIT_TYPE_LABEL = "Новый тип"
EDIT_MOVE = "Сдвинуть"
EDIT_RETYPE = "Сменить тип"
EDIT_DELETE = "Удалить"
EDIT_ADD_HEADER = "Добавить событие"
EDIT_ADD_TYPE_LABEL = "Тип"
EDIT_ADD_TIME_LABEL = "Время (мм:сс)"
EDIT_ADD_SUBMIT = "Добавить"
EDIT_UNDO = "Отменить последнюю правку"
EDIT_UNDO_HELP = "Отменяет самую свежую действующую правку; журнал правок остаётся целым."
EDIT_NO_EDITS = "Правок пока нет."
EDIT_COUNT = "Правок в силе: {count}"

EDIT_MOVED = "Событие «{type}» сдвинуто на {time} ({half}-й тайм)."
EDIT_RETYPED = "Тип события изменён: «{old}» → «{new}»."
EDIT_DELETED = "Событие «{type}» в {time} удалено. Его можно вернуть кнопкой отмены."
EDIT_ADDED = "Добавлено событие «{type}» в {time} ({half}-й тайм)."
EDIT_UNDONE = "Последняя правка отменена."
EDIT_FAILED = "Правка не записана: {error}"
EDIT_EVENT_GONE = "Событие больше не в списке. Выберите строку заново."

ERR_TIME_FORMAT = "Время нужно писать как мм:сс, например 12:05 (секунды от 00 до 59)."
ERR_TIME_AFTER_END = "Время {time} дальше конца видео ({end})."
ERR_TIME_OUTSIDE_HALVES = "Время {time} вне таймов (перерыв или студия). {spans}."
ERR_TIME_UNCHANGED = "Время не изменилось: впишите другое."
ERR_TYPE_UNCHANGED = "Это уже тот же тип."
ERR_NO_BOUNDS = (
    "Не найдены границы таймов и длина видео (файлы candidates.json и scoreboard.json рядом с "
    "видео). Время проверить нечем, правка не записана. Выполните "
    "`python -m aicoach.cli run --video <путь к видео>`."
)
HALF_SPAN = "{half}-й тайм {start}–{end}"

CLIP_HEADER = "Клип"
CLIP_BEFORE = "До, с"
CLIP_AFTER = "После, с"
CLIP_EXPORT = "Экспорт клипа"
CLIP_EXPORT_HELP = (
    "Вырезает окно вокруг выбранного события из видео матча (FFmpeg, перекодирование). "
    "Готовый клип с тем же окном берётся с диска без повторной нарезки."
)
CLIP_PICK_FIRST = "Выберите событие в таблице, чтобы вырезать клип."
CLIP_NO_VIDEO = "Видео матча не передано: клип вырезать не из чего. Запустите `ui --video <путь>`."
CLIP_READY = "Клип: «{type}» в {time}, {before} с до и {after} с после."
CLIP_FAILED = "Клип не сделан: {error}"
CLIP_DOWNLOAD = "Скачать клип"
CLIP_FILE_GONE = "Файл клипа больше не найден: {name}. Вырежьте его заново."

EPISODE_ANSWER_MARK = "Описание эпизода: по клипу, не проверено."
EPISODE_DESCRIBE = "Описать эпизод"
EPISODE_DESCRIBE_HELP = (
    "Видеомодель смотрит клип выбранного события (окно из полей «До» и «После») и пишет 2–4 "
    "предложения о том, что видно на поле. Платный запрос; готовое описание берётся из "
    "кэша бесплатно. Текст по клипу, не проверен."
)
EPISODE_PICK_FIRST = "Выберите событие в таблице, чтобы описать эпизод."
EPISODE_NO_VIDEO = "Видео матча не передано: описывать нечего. Запустите `ui --video <путь>`."
EPISODE_SPINNER = "Видеомодель смотрит клип…"
EPISODE_HEADER = "Эпизод: «{type}» в {time}"
EPISODE_UNVERIFIED = "По клипу, не проверено."
EPISODE_META = "модель {model}, токены {tokens_in}/{tokens_out}, стоимость {cost}"
EPISODE_META_CACHED = "из кэша, без нового запроса"
EPISODE_COST_UNKNOWN = "неизвестна"
EPISODE_FAILED = "Описание не получено: {error}"

MISSING_GEMINI_KEY = (
    "Ключ GEMINI_API_KEY не задан ни в переменной окружения, ни в файле .env в корне "
    "репозитория: скопируйте .env.example в .env и впишите ключ."
)
MISSING_OPENROUTER_KEY = (
    "Ключ OPENROUTER_API_KEY не задан ни в переменной окружения, ни в файле .env в корне "
    "репозитория: впишите ключ в .env (или выберите другой поставщик: --provider gemini)."
)
BUDGET_EXCEEDED = (
    "Бюджет месяца исчерпан: потрачено {spent:.4f} USD, оценка запроса {estimate:.4f} USD, "
    "потолок {budget:.2f} USD (AICOACH_BUDGET_USD). Запрос не отправлен."
)
UNREADABLE_ENV_FILE = (
    "Файл .env не читается (права доступа или не UTF-8); исправьте или удалите его, "
    "шаблон лежит в .env.example."
)
NO_DATA_ROOT = (
    "Не задан каталог данных: укажите --data-root или AICOACH_DATA_ROOT (переменная окружения "
    "или .env, см. .env.example)."
)

# New match from a YouTube link (D15): the ingest command and the sidebar block.
TDEED_WAITING = (
    "Ждём файл модели T-DEED. Обработайте видео {video} в ноутбуке T-DEED "
    "(перекодировать в 25 к/с без обрезки, `inference.py` с `--inference_threshold 0.05`) "
    "и положите его `results_inference.json` в каталог {dir} под именем `tdeed_<имя>.json`, "
    "например `tdeed_SoccerNet_small.json`; такой файл в каталоге должен быть один."
)
TDEED_RESUME_CLI = (
    "Затем запустите ту же команду `ingest` ещё раз: готовые стадии пропустятся, "
    "продолжится слияние."
)
TDEED_RESUME_UI = (
    "Затем нажмите «Обработать» с той же ссылкой ещё раз: готовые стадии пропустятся, "
    "продолжится слияние."
)
# The tdeed command (D-080): T-DEED on the video card of this machine or in the Colab notebook.
TDEED_NO_CUDA = (
    "На этой машине нет видеокарты NVIDIA с CUDA (или не установлен torch: "
    "`pip install .[gpu]`), а T-DEED без неё работает намного медленнее. Запустите шаг в Colab: "
    "{url} . Если всё же нужен процессор, укажите `--device cpu` явно."
)
TDEED_CPU_WARNING = (
    "Предупреждение: T-DEED идёт на процессоре, это намного медленнее, чем на видеокарте."
)
TDEED_DONE = "Готово: {file} ({n} срабатываний, устройство {device})."
TDEED_CACHED = (
    "Уже есть: {file} ({n} срабатываний). Ничего не запускалось; `--force` запустит заново."
)
TDEED_SEGMENT = (
    "Отрезок {start}–{end} ({length}); начало сдвинуто вниз на сетку модели "
    "(было запрошено {requested})."
)
TDEED_TIMINGS = (
    "Время: подготовка (код, чекпойнт) {prepare:.1f} с, FFmpeg {ffmpeg:.1f} с, "
    "инференс {inference:.1f} с, всего {total:.1f} с."
)
TDEED_PER_MINUTE = (
    "На минуту видео: FFmpeg {ffmpeg:.1f} с, инференс {inference:.1f} с "
    "(инференс включает загрузку модели)."
)
TDEED_ESTIMATE = (
    "Оценка на весь матч ({minutes:.1f} мин видео): FFmpeg {ffmpeg:.0f} с, инференс "
    "{inference:.0f} с, всего {total:.0f} с (около {total_min:.0f} мин). Это оценка по отрезку, "
    "не измерение."
)
TDEED_COMPARE_SAME = "совпадает"
TDEED_COMPARE_DIFF = "различия"
INGEST_HEADER = "Новый матч"
INGEST_URL = "Ссылка на YouTube"
INGEST_URL_HELP = "Например https://www.youtube.com/watch?v=… или https://youtu.be/…"
INGEST_START = "Обработать"
INGEST_LINEUPS = "Состав команд (lineups.csv), необязательно"
INGEST_LINEUPS_HELP = (
    "Колонки: team, side (home или away), number, name, position, started, minute_on, "
    "minute_off. С составом фамилии на плашках сверяются с игроками своей команды."
)
INGEST_LINEUP_BAD = "Файл состава не принят: {error}"
INGEST_NO_ROOT = (
    "Каталог данных не задан: запустите интерфейс с `--data-root <каталог>`, "
    "чтобы обрабатывать новые матчи."
)
INGEST_BAD_URL = "Ссылка не принята: {error}"
INGEST_BUSY = "Этот матч уже обрабатывается (процесс {pid}). Дождитесь окончания."
INGEST_START_FAILED = "Не удалось запустить обработку: {error}"
INGEST_STARTED = "Обработка запущена. Журнал: {log}"
INGEST_MATCH = "Матч {match_id}"
INGEST_LOG = "Журнал процесса: {log}"
INGEST_FAILED = "Обработка остановилась: {error}"
INGEST_PROCESS_GONE = "Процесс обработки завершился, а статус не обновлён. Смотрите журнал {log}."
INGEST_DONE = "Матч обработан, события в базе."
INGEST_OPEN = "Открыть матч"
INGEST_STEP_LABELS: dict[str, str] = {
    "download": "Загрузка видео",
    "audio": "Звук 16 кГц",
    "scoreboard": "Табло",
    "plates": "Плашки",
    "plate_names": "Фамилии с плашек",
    "transcribe": "Расшифровка комментария",
    "tdeed": "Файл T-DEED",
    "fuse": "Слияние в базу",
    "teams": "Команды и форма",
    "corner_tags": "Теги угловых",
    "review_candidates": "Кандидаты для проверки",
}
INGEST_STATE_LABELS: dict[str, str] = {
    "pending": "ещё не начато",
    "running": "идёт",
    "done": "готово",
    "failed": "ошибка",
    "waiting": "ждёт T-DEED",
    "skipped": "пропущено",
}
NO_MATCH_OPEN = (
    "Матч не открыт. Вставьте ссылку в блок «Новый матч» в боковой панели "
    "или запустите `python -m aicoach.cli ui --video <путь> --youtube-id <id>`."
)

# Names of the players on the plates (D-059).
COLUMN_PLAYER = "Игрок"
PLAYER_UNCHECKED_MARK = " (без сверки с составом)"
PLAYER_UNRECOGNIZED = "не распознано"
PLAYER_WENT_ON = "вышел"
PLAYER_WENT_OFF = "ушёл"
PLATE_NAMES_SKIPPED = (
    "Фамилии с плашек не прочитаны: {reason}. Установите Tesseract с языками kaz и rus "
    "и запустите обработку ещё раз; остальные стадии это не затрагивает."
)
MISSING_SPEECH_KEY = "нет ключа поставщика {provider} (см. .env.example)"
TRANSCRIBE_SKIPPED = (
    "Комментарий не расшифрован: {reason}. Впишите ключ OpenRouter в .env и запустите "
    "обработку ещё раз; остальные стадии это не затрагивает."
)


# The agent (FR-8): answers, refusal and the "Agent" tab.
AGENT_REFUSAL = "Не смогла подтвердить ответ данными, поэтому не отвечаю. Переформулируйте вопрос."
AGENT_VIDEO_TIME = "видео {time}"
# Icons of the links in the agent's answers; a foul with a card plate gets the card's icon.
EVENT_ICONS: dict[EventType, str] = {
    EventType.GOAL: "⚽",
    EventType.SHOT: "🎯",
    EventType.CORNER: "🚩",
    EventType.FOUL: "⚠️",
    EventType.SUBST: "🔄",
}
CARD_ICONS: dict[str, str] = {"yellow": "🟨", "red": "🟥"}
# A link to a stretch of the commentary (D-070).
SEGMENT_ICON = "🎙"
AGENT_TAB = "Агент"
AGENT_HINT = (
    "Вопросы о событиях матча: голы, удары, угловые, фолы, замены. Ответы строятся только по "
    "событиям, которые нашла система; это не официальная статистика."
)
AGENT_INPUT = "Вопрос о матче"
AGENT_OPEN = "▶ {label}"
AGENT_NO_KEY = "Агент недоступен: {error}"
AGENT_FAILED = "Не удалось получить ответ: {error}"
AGENT_CLIP = "Клип: {name}"
AGENT_THINKING = "Смотрю события матча…"
AGENT_CLEAR = "Очистить историю"
AGENT_META = (
    "Проверка: {outcome}; вызовов инструментов: {calls}; "
    "токенов: {tokens_in} вход, {tokens_out} выход; {seconds:.1f} с"
)
AGENT_OUTCOME_LABELS: dict[str, str] = {
    "first_try": "с первой попытки",
    "retry": "со второй попытки",
    "refused": "отказ",
}


# The review mode (D-075): the events and the candidates of the model with a check on every row.
REVIEW_MODE = "Режим проверки"
REVIEW_MODE_HELP = (
    "Список событий и кандидатов (ударов модели, которых нет среди событий) с кнопками ✓ и ✗ "
    "у каждой строки. "
    "Агент кандидатов не видит."
)
REVIEW_SHOW_ALL = "Показать всё"
REVIEW_SHOW_ALL_HELP = "Вместе с уже решёнными строками."
REVIEW_HEADER = "Проверка"
REVIEW_PROGRESS = "Проверено {checked} из {total}"
REVIEW_NEXT = "Следующий"
REVIEW_CANDIDATE_MARK = "кандидат"
REVIEW_DECIDED_MARK = "решено"
REVIEW_NO_LAYER = (
    "Файла кандидатов нет: показаны только события. Постройте его командой "
    "`python -m aicoach.cli review-candidates --match-dir <каталог матча>`."
)
REVIEW_NOTHING_LEFT = "Всё проверено: в списке не осталось строк."
REVIEW_NO_MATCH_DIR = "Режим проверки не работает без каталога матча."
REVIEW_OK_HELP_CANDIDATE = "Добавить событие"
REVIEW_NO_HELP_CANDIDATE = "Отклонить кандидата"
REVIEW_OK_HELP_EVENT = "Подтвердить событие"
REVIEW_NO_HELP_EVENT = "Удалить событие"
REVIEW_PLAY_HELP = "Показать в плеере"
REVIEW_ACCEPTED = "Кандидат добавлен как событие: {type} {time}."
REVIEW_REJECTED = "Кандидат отклонён: {type} {time}."
REVIEW_CONFIRMED = "Событие подтверждено: {type} {time}."
REVIEW_GONE = "Этой строки уже нет в списке: страница обновилась, повторите."
REVIEW_LOG_FAILED = (
    "Не удалось записать решение в журнал проверки: {error}. "
    "Событие уже изменено, строку решённой не считаем."
)
REVIEW_LOG_BROKEN = "Журнал проверки не читается: {error}"

REEL_CAPTION = "{type} · {minute}"
REEL_CARD_LABEL = "Карточка"
REEL_HEADER = "Ролик по категории"
REEL_KIND = "Категория"
REEL_KIND_CARD = "Карточки (фолы с карточкой)"
REEL_HALF = "Тайм"
REEL_SIDE = "Команда"
REEL_SIDE_ALL = "Обе"
REEL_BUILD = "Собрать ролик"
REEL_BUILD_HELP = (
    "Нарезает все события выбранной категории (с учётом правок) и склеивает их в один ролик. "
    "В начале каждого куска подпись с категорией и минутой. Готовый ролик с тем же набором "
    "событий берётся с диска."
)
REEL_NO_VIDEO = "Видео матча не передано: ролик собрать не из чего. Запустите `ui --video <путь>`."
REEL_COUNT = "Событий для ролика: {count}."
REEL_EMPTY = "Нет событий для ролика с такими фильтрами."
REEL_TOO_MANY = "Событий {count}, это больше предела {limit}: сузьте ролик по тайму или команде."
REEL_TEAM_UNKNOWN = "Не вошло событий с неизвестной командой: {count}."
REEL_PROGRESS = "Кусок {done} из {total}"
REEL_READY = "Ролик: {count} событий, {duration}."
REEL_FROM_CACHE = "готовый файл с диска"
REEL_NO_CAPTIONS = (
    "Подписей нет: не найден шрифт с кириллицей. Задайте файл шрифта в `AICOACH_FONT_FILE`."
)
REEL_REENCODED = "Куски имели разные параметры, ролик перекодирован."
REEL_FAILED = "Ролик не собран: {error}"
REEL_DOWNLOAD = "Скачать ролик"
REEL_FILE_GONE = "Файл ролика больше не найден: {name}. Соберите его заново."
REEL_DURATION_UNKNOWN = "длина неизвестна"

LABELS_HEADER = "Разметка"
LABELS_REVIEWED_ONLY = "Только проверенные"
LABELS_REVIEWED_HELP = (
    "В файл войдут только события, которые вы подтвердили в режиме проверки, добавили или "
    "исправили (сдвиг, смена типа)."
)
LABELS_DOWNLOAD = "Скачать разметку (JSON)"
LABELS_SUMMARY = "В файле {count} событий."
LABELS_SKIPPED = "Не вошло событий без определённого тайма: {count}."
LABELS_FAILED = "Разметка не собрана: {error}"


# Teams and kit colours (D-078).
KIT_LABELS: dict[str, str] = {
    "white": "белый",
    "black": "чёрный",
    "red": "красный",
    "blue": "синий",
    "light_blue": "голубой",
    "green": "зелёный",
    "yellow": "жёлтый",
    "orange": "оранжевый",
    "purple": "фиолетовый",
    "other": "другой",
    "unclear": "не видно",
}
KIT_UNDETERMINED = "не определён"
TEAMS_KITS_SKIPPED = (
    "Цвета формы не определены: {reason}. Впишите ключ OpenRouter в .env и запустите шаг ещё раз "
    "или задайте цвет вручную в интерфейсе."
)
TEAMS_NO_VIDEO = "Цвета формы не определены: видео матча (src.mp4) не найдено."
TEAMS_NO_DB = "Цвета формы не определены: базы событий (aicoach.db) ещё нет."
TEAMS_NO_SOURCES = (
    "Цвета формы не определены: нет голов и карточек с известной командой, по которым можно "
    "смотреть клипы."
)
TEAMS_HEADER = "Команды и форма"
TEAMS_HELP = (
    "Названия берутся из состава, иначе из заголовка видео (допущение: первая команда в заголовке "
    "хозяева; сверьте), иначе «хозяева» и «гости». Цвет "
    "формы определяет видеомодель по клипам голов и карточек. Ваше значение важнее: оно "
    "хранится отдельно и не стирается новым запуском."
)
TEAMS_NAME_LABEL = "Название: {side}"
TEAMS_KIT_LABEL = "Цвет формы: {side}"
TEAMS_KIT_AUTO = "как определено"
TEAMS_SAVE = "Сохранить"
TEAMS_RESET = "Сбросить мои значения"
TEAMS_SAVED = "Сохранено."
TEAMS_SAVE_FAILED = "Не сохранено: {error}"
TEAMS_SOURCE_LABELS: dict[str, str] = {
    "lineup": "состав",
    "title": "заголовок видео",
    "default": "по умолчанию",
    "manual": "вручную",
    "vlm": "по клипам",
    "undetermined": "не определено",
}
TEAMS_LINE = "{side}: {name} ({name_source}), форма {kit} ({kit_source})"

# Corner tags (D-078): an extension beyond the specification; a model's reading of a clip.
CORNER_TAGS_SKIPPED = "теги угловых не получены: {reason}"
CORNER_TAGS_NO_DB = "теги угловых не получены: базы событий (aicoach.db) ещё нет"
CORNER_TAGS_NO_VIDEO = "теги угловых не получены: видео матча (src.mp4) не найдено"
CORNER_UNVERIFIED = "по клипу, не проверено"
CORNER_TAGS_ANSWER_MARK = "Теги угловых: по клипу, не проверено."
# The `zone` tag was right in 4 of 8 corners of match01 (docs/metrics.md, D-078).
CORNER_LOW_RELIABILITY = "низкая надёжность"
CORNER_ZONE_ANSWER_MARK = (
    "Поле «куда прилетел мяч» (zone) низкой надёжности: на пилоте совпало 4 из 8."
)
CORNER_ATTACKING = "подаёт"
CORNER_TEAM_UNKNOWN = "команда не определена"
CORNER_NOT_A_CORNER = "по клипу это не угловой"
COLUMN_CORNER = "Теги углового"
CORNER_FIELD_LABELS: dict[str, str] = {
    "is_corner": "Это угловой",
    "attacking_kit": "Цвет формы подающих",
    "delivery": "Подача",
    "zone": "Куда прилетел мяч (низкая надёжность)",
    "first_contact": "Первое касание",
    "outcome": "Чем кончилось",
}
CORNER_FIELD_SHORT: dict[str, str] = {
    "delivery": "подача",
    "zone": "зона",
    "first_contact": "контакт",
    "outcome": "исход",
}
CORNER_VALUE_LABELS: dict[str, dict[str, str]] = {
    "is_corner": {"yes": "да", "no": "нет", "unclear": "не видно"},
    "attacking_kit": KIT_LABELS,
    "delivery": {
        "cross": "навес",
        "short": "короткий",
        "low_ball": "прострел низом",
        "unclear": "не видно",
    },
    "zone": {
        "near_post": "ближняя штанга",
        "center": "центр",
        "far_post": "дальняя штанга",
        "edge_of_box": "край штрафной",
        "unclear": "не видно",
    },
    "first_contact": {
        "attack": "атака",
        "defence": "защита",
        "keeper": "вратарь",
        "nobody": "никто",
        "unclear": "не видно",
    },
    "outcome": {
        "shot": "удар",
        "goal": "гол",
        "cleared": "выбили",
        "keeper_caught": "вратарь забрал",
        "corner_or_out": "угловой или аут",
        "attack_continues": "атака продолжилась",
        "unclear": "не видно",
    },
}
CORNER_FILTERS_HEADER = "Фильтры угловых (по клипу, не проверено)"
CORNER_FILTERS_NO_TAGS = (
    "Тегов угловых нет. Запустите `python -m aicoach.cli corner-tags --match-dir <каталог>`."
)
CORNER_FILTER_TEAM = "Команда подачи"
CORNER_FILTER_ANY = "любое"
CORNER_FILTER_TEAM_UNKNOWN = "команда не определена"
CORNER_FILTER_NOTE = "Фильтры относятся только к угловым; остальные события они не скрывают."
