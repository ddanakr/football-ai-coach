"""Event taxonomy. A new category is a change to the specification (CLAUDE.md section 5)."""

from enum import StrEnum


class EventType(StrEnum):
    """The five event categories of the catalog."""

    GOAL = "goal"
    SHOT = "shot"
    CORNER = "corner"
    FOUL = "foul"
    SUBST = "subst"
