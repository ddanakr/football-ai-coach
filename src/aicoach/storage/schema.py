"""SQL schema of the event store (TZ section 9, "База").

`predictions` and `edits` are append-only, enforced by triggers. The current state of an
event is never stored: it is computed by the view `events_current` from the predictions of
the latest run of the match plus the effective edits (those that were not undone).
Deleting an event only hides it. Older runs stay in `predictions` for audit but are not
shown, and edits of their predictions stay in the journal but are not applied (D-028).

Views hold no data, so `init_db` drops and recreates them on every call: a changed view
definition reaches existing databases without a schema version bump.
"""

SCHEMA_VERSION = 1

_TYPES = "'goal', 'shot', 'corner', 'foul', 'subst'"
_NOW = "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"

PREDICTIONS_TABLE = f"""
CREATE TABLE IF NOT EXISTS predictions (
    id         INTEGER PRIMARY KEY,
    match_id   TEXT NOT NULL CHECK (match_id <> ''),
    type       TEXT NOT NULL CHECK (type IN ({_TYPES})),
    t_v        REAL NOT NULL CHECK (t_v >= 0),
    half       INTEGER CHECK (half IN (1, 2)),
    score      REAL,
    status     TEXT NOT NULL DEFAULT 'auto',
    source     TEXT NOT NULL CHECK (source IN ('plates', 'model', 'speech', 'merge')),
    team       TEXT CHECK (team IN ('home', 'away')),
    run_id     TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT ({_NOW})
)
"""

# A target is 'p:<prediction id>' or 'e:<id of the add edit>'. Which fields an action needs
# is fixed by the table-level CHECK, so a malformed row cannot be written even by raw SQL.
EDITS_TABLE = f"""
CREATE TABLE IF NOT EXISTS edits (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id   TEXT NOT NULL CHECK (match_id <> ''),
    action     TEXT NOT NULL
               CHECK (action IN ('add', 'move', 'retype', 'delete', 'verify', 'undo')),
    target     TEXT CHECK (target GLOB 'p:[0-9]*' OR target GLOB 'e:[0-9]*'),
    type       TEXT CHECK (type IN ({_TYPES})),
    t_v        REAL CHECK (t_v >= 0),
    undo_of    INTEGER REFERENCES edits (id),
    author     TEXT NOT NULL CHECK (author <> ''),
    created_at TEXT NOT NULL DEFAULT ({_NOW}),
    CHECK (
        (action = 'add'
            AND target IS NULL AND type IS NOT NULL AND t_v IS NOT NULL AND undo_of IS NULL)
        OR (action = 'move'
            AND target IS NOT NULL AND type IS NULL AND t_v IS NOT NULL AND undo_of IS NULL)
        OR (action = 'retype'
            AND target IS NOT NULL AND type IS NOT NULL AND t_v IS NULL AND undo_of IS NULL)
        OR (action IN ('delete', 'verify')
            AND target IS NOT NULL AND type IS NULL AND t_v IS NULL AND undo_of IS NULL)
        OR (action = 'undo'
            AND target IS NULL AND type IS NULL AND t_v IS NULL AND undo_of IS NOT NULL)
    )
)
"""

STATEMENTS: tuple[str, ...] = (
    PREDICTIONS_TABLE,
    EDITS_TABLE,
    "CREATE INDEX IF NOT EXISTS idx_predictions_match ON predictions (match_id)",
    "CREATE INDEX IF NOT EXISTS idx_edits_match_target ON edits (match_id, target)",
    # Backstop for the check in undo_edit: one edit can be undone only once.
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_edits_one_undo ON edits (undo_of) "
    "WHERE undo_of IS NOT NULL",
    *(
        f"CREATE TRIGGER IF NOT EXISTS {table}_no_{verb.lower()} "
        f"BEFORE {verb} ON {table} "
        f"BEGIN SELECT RAISE(ABORT, '{table} is append-only: {verb} is not allowed'); END"
        for table in ("predictions", "edits")
        for verb in ("UPDATE", "DELETE")
    ),
)

# Edits that count: everything except undo records and the edits that an undo cancelled.
EFFECTIVE_EDITS_VIEW = """
CREATE VIEW effective_edits AS
SELECT e.*
FROM edits AS e
WHERE e.action <> 'undo'
  AND NOT EXISTS (SELECT 1 FROM edits AS u WHERE u.undo_of = e.id)
"""

# The latest run of a match is the run_id of its prediction row with the largest id. One run
# is the complete set of events of the match (written by the fusion stage only), so the
# rows of earlier runs are superseded, not merged.
LATEST_RUNS_VIEW = """
CREATE VIEW latest_runs AS
SELECT p.match_id, p.run_id
FROM predictions AS p
WHERE p.id = (SELECT MAX(q.id) FROM predictions AS q WHERE q.match_id = p.match_id)
"""

# One row per event: every prediction of the latest run and every effective `add`, with the
# effective edits applied. The latest move sets t_v, the latest retype sets type, any
# delete sets `deleted`, any verify sets `verified`. Nothing is removed from the base
# tables; edits that target a prediction of an older run match no row and are not applied.
EVENTS_CURRENT_VIEW = """
CREATE VIEW events_current AS
WITH base AS (
    SELECT 'p:' || p.id AS event_id, p.match_id, p.type, p.t_v, p.half, p.score,
           p.status, p.team, p.source, 'prediction' AS origin
    FROM predictions AS p
    JOIN latest_runs AS r ON r.match_id = p.match_id AND r.run_id = p.run_id
    UNION ALL
    SELECT 'e:' || a.id, a.match_id, a.type, a.t_v, NULL, NULL,
           'manual', NULL, NULL, 'user'
    FROM effective_edits AS a
    WHERE a.action = 'add'
)
SELECT
    b.event_id,
    b.match_id,
    COALESCE((SELECT x.type FROM effective_edits AS x
              WHERE x.target = b.event_id AND x.action = 'retype'
              ORDER BY x.id DESC LIMIT 1), b.type) AS type,
    COALESCE((SELECT x.t_v FROM effective_edits AS x
              WHERE x.target = b.event_id AND x.action = 'move'
              ORDER BY x.id DESC LIMIT 1), b.t_v) AS t_v,
    b.half,
    b.score,
    b.status,
    b.team,
    b.source,
    b.origin,
    EXISTS (SELECT 1 FROM effective_edits AS x
            WHERE x.target = b.event_id AND x.action = 'verify') AS verified,
    EXISTS (SELECT 1 FROM effective_edits AS x
            WHERE x.target = b.event_id AND x.action = 'delete') AS deleted,
    (SELECT COUNT(*) FROM effective_edits AS x WHERE x.target = b.event_id) AS n_edits
FROM base AS b
"""

# Name -> definition, in dependency order (a view comes after the views it reads).
VIEWS: dict[str, str] = {
    "effective_edits": EFFECTIVE_EDITS_VIEW,
    "latest_runs": LATEST_RUNS_VIEW,
    "events_current": EVENTS_CURRENT_VIEW,
}
