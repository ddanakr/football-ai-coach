"""The lineup of the match (`lineups.csv`, optional): who plays for which side, and the search of
a read name among the players of one side.

Columns: team, side, number, name, position, started, minute_on, minute_off (only `side` and
`name` are needed here). The file may be saved by a spreadsheet: BOM, spaces around cells and
`;` as the separator are accepted.
"""

import csv
import io
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Literal

from aicoach.errors import PlatesError
from aicoach.plate_names.text import sorted_key

LINEUP_FILE = "lineups.csv"
Side = Literal["home", "away"]
_SIDES = ("home", "away")
_REQUIRED = ("side", "name")
_MAX_NUMBER_DIGITS = 3


@dataclass(frozen=True)
class Player:
    """One line of the lineup. `number` is None when the cell is empty or not a number."""

    side: Side
    name: str
    number: int | None = None


@dataclass(frozen=True)
class LineupMatch:
    """The player a read name was matched to and how alike the two texts are (0..1)."""

    player: Player
    similarity: float


def _delimiter(text: str) -> str:
    """`;` when the header line has more of them than commas (Russian spreadsheets)."""
    header = text.splitlines()[0] if text.strip() else ""
    return ";" if header.count(";") > header.count(",") else ","


def _number(cell: str) -> int | None:
    cell = cell.strip()
    # A shirt number has at most three digits; a longer digit string is garbage (and a very long
    # one would make `int` raise ValueError).
    return int(cell) if cell.isdecimal() and len(cell) <= _MAX_NUMBER_DIGITS else None


def parse_lineup(data: bytes) -> list[Player]:
    """Read the lineup from the bytes of a CSV file.

    Raises:
        PlatesError: not UTF-8 text or not a CSV, a required column is missing, a `side` is
            neither `home` nor `away`, a name is empty, or there is no player at all. The
            message names the line; the file content is not quoted.
    """
    try:
        text = data.decode("utf-8-sig")
        rows = list(
            csv.reader(io.StringIO(text), delimiter=_delimiter(text), skipinitialspace=True)
        )
    except UnicodeDecodeError as err:
        msg = "lineups.csv is not UTF-8 text"
        raise PlatesError(msg) from err
    except csv.Error as err:
        msg = f"lineups.csv is not a CSV file: {err}"
        raise PlatesError(msg) from err
    if not rows:
        msg = "lineups.csv is empty"
        raise PlatesError(msg)
    column = {cell.strip().lower(): i for i, cell in enumerate(rows[0])}
    missing = [name for name in _REQUIRED if name not in column]
    if missing:
        msg = f"lineups.csv has no column {', '.join(missing)} (needed: {', '.join(_REQUIRED)})"
        raise PlatesError(msg)
    players = [
        _player(row, column, line)
        for line, row in enumerate(rows[1:], start=2)
        if any(cell.strip() for cell in row)
    ]
    if not players:
        msg = "lineups.csv has a header but no players"
        raise PlatesError(msg)
    return players


def _cell(row: list[str], column: dict[str, int], name: str) -> str:
    index = column.get(name)
    return row[index].strip() if index is not None and index < len(row) else ""


def _player(row: list[str], column: dict[str, int], line: int) -> Player:
    side = _cell(row, column, "side").lower()
    name = " ".join(_cell(row, column, "name").split())
    if side not in _SIDES:
        msg = f"lineups.csv line {line}: side must be home or away"
        raise PlatesError(msg)
    if not name:
        msg = f"lineups.csv line {line}: the name is empty"
        raise PlatesError(msg)
    return Player(
        side="home" if side == "home" else "away",
        name=name,
        number=_number(_cell(row, column, "number")),
    )


def load_lineup(path: Path) -> list[Player]:
    """Read `lineups.csv`.

    Raises:
        PlatesError: the file cannot be read or is invalid (see `parse_lineup`).
    """
    try:
        data = path.read_bytes()
    except OSError as err:
        msg = f"cannot read {path}: {type(err).__name__}"
        raise PlatesError(msg) from err
    return parse_lineup(data)


def similarity(read: str, candidate: str) -> float:
    """How alike two names are, 0..1; the order of the words does not matter."""
    a, b = sorted_key(read), sorted_key(candidate)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def find_player(
    read: str, players: list[Player], side: Side, min_similarity: float
) -> tuple[LineupMatch | None, float | None]:
    """The player of `side` whose name is most like `read`.

    Returns the match (None when the best similarity is below `min_similarity`, when two
    different players tie for the best place, or when `side` has no players) and the best
    similarity seen (None when nothing was compared). Players of the other side are never
    considered. Nothing is guessed: a doubtful text stays unrecognized.
    """
    scored = sorted(
        ((similarity(read, p.name), p) for p in players if p.side == side),
        key=lambda item: -item[0],
    )
    if not scored:
        return None, None
    best_score, best = scored[0]
    if best_score < min_similarity:
        return None, best_score
    if len(scored) > 1 and scored[1][0] == best_score and scored[1][1].name != best.name:
        return None, best_score
    return LineupMatch(best, best_score), best_score


def parse_team_names(data: bytes) -> dict[Side, str]:
    """The team name of each side from the `team` column; empty where it is missing.

    Never raises: the names are decoration, a file that `parse_lineup` would reject simply has
    none here (the caller reads the players with `parse_lineup`, which reports the problem).
    """
    try:
        text = data.decode("utf-8-sig")
        rows = list(
            csv.reader(io.StringIO(text), delimiter=_delimiter(text), skipinitialspace=True)
        )
    except (UnicodeDecodeError, csv.Error):
        return {}
    if not rows:
        return {}
    column = {cell.strip().lower(): i for i, cell in enumerate(rows[0])}
    if "team" not in column or "side" not in column:
        return {}
    names: dict[Side, str] = {}
    for row in rows[1:]:
        side = _cell(row, column, "side").lower()
        team = " ".join(_cell(row, column, "team").split())
        if team and side in _SIDES:
            names.setdefault("home" if side == "home" else "away", team)
    return names
