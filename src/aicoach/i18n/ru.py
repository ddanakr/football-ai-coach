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

MISSING_GEMINI_KEY = (
    "Ключ GEMINI_API_KEY не найден в файле .env в корне репозитория: "
    "скопируйте .env.example в .env и впишите ключ."
)
