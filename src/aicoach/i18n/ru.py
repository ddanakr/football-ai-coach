"""Russian texts of the viewer (`aicoach.ui`)."""

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
SIDEBAR_PLAYER = "Плеер"
PLAYER_VARIANT = "Вариант плеера"
PLAYER_IFRAME = "Встроенный iframe"
PLAYER_ST_VIDEO = "st.video"

COLUMN_TIME = "Время"
COLUMN_TYPE = "Тип"
COLUMN_HALF = "Тайм"
COLUMN_SCORE = "Уверенность"
COLUMN_SOURCE = "Источник"
COLUMN_TEAM = "Команда"

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
