"""Pure assembly of the names of one plate from the texts read: the lineup check and the rules
for what is reported (D-059). No files and no OCR here."""

from collections.abc import Mapping

from aicoach.namematch import find_player
from aicoach.plate_names.lineup import Player
from aicoach.plate_names.schemas import (
    PlateNames,
    PlateNamesStats,
    PlayerRead,
    Role,
)
from aicoach.plate_names.text import clean_text
from aicoach.plates.schemas import Plate, Team

_DIGITS = 3  # similarity is stored with three decimals: the file stays stable and short


def read_player(
    role: Role,
    raw_text: str,
    side: Team | None,
    lineup: list[Player] | None,
    min_similarity: float,
) -> PlayerRead:
    """What one read line says about a player.

    With a lineup and a known side the cleaned text is searched among the players of that side:
    found means `matched` with the lineup spelling, a text too unlike every player means
    `unrecognized` (no name, nothing is guessed). Without a lineup, or when the side of the
    plate is unknown, the cleaned text itself is the name and the status is `unchecked`. A line
    with no readable letters is `unrecognized` in every case.
    """
    cleaned = clean_text(raw_text)
    if not cleaned:
        return PlayerRead(
            role=role, raw_text=raw_text, name=None, source="ocr", status="unrecognized"
        )
    if lineup is None or side is None:
        return PlayerRead(
            role=role, raw_text=raw_text, name=cleaned, source="ocr", status="unchecked"
        )
    match, best = find_player(cleaned, lineup, side, min_similarity)
    best_score = None if best is None else round(best, _DIGITS)
    if match is None:
        return PlayerRead(
            role=role,
            raw_text=raw_text,
            name=None,
            source="ocr",
            similarity=best_score,
            status="unrecognized",
        )
    return PlayerRead(
        role=role,
        raw_text=raw_text,
        name=match.player.name,
        source="ocr+lineup",
        similarity=best_score,
        status="matched",
    )


def build_plate_names(
    plate: Plate,
    texts: Mapping[Role, str],
    lineup: list[Player] | None,
    min_similarity: float,
) -> PlateNames:
    """The names of a plate from its read lines (`texts` by role, in reading order)."""
    return PlateNames(
        plate_id=plate.id,
        type=plate.type,
        t_v=plate.t_first_seen,
        half=plate.half,
        side=plate.team,
        players=[
            read_player(role, text, plate.team, lineup, min_similarity)
            for role, text in texts.items()
        ],
    )


def count_names(names: list[PlateNames]) -> PlateNamesStats:
    """Counters over all names of all plates."""
    players = [p for plate in names for p in plate.players]
    return PlateNamesStats(
        n_plates=len(names),
        n_names=len(players),
        n_matched=sum(p.status == "matched" for p in players),
        n_unchecked=sum(p.status == "unchecked" for p in players),
        n_unrecognized=sum(p.status == "unrecognized" for p in players),
    )
