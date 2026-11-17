"""The values of the corner tags, the question to the model and the reading of its answer."""

import json

import pytest

from aicoach.corner_tags.schemas import (
    CHOICES,
    FIELDS,
    UNCLEAR,
    CornerFields,
    normalize_value,
)
from aicoach.corner_tags.tagger import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_prompt,
    parse_corner_answer,
)
from aicoach.i18n import ru
from aicoach.vlm import ask_json, extract_json_object
from tests.fixtures.corner_synth import CORNER_JSON, corner_reply
from tests.fixtures.episode_synth import FakeVideoModel, reply


@pytest.mark.unit
def test_every_field_has_a_russian_name_for_every_code() -> None:
    for name in FIELDS:
        assert set(ru.CORNER_VALUE_LABELS[name]) == set(CHOICES[name])
        assert UNCLEAR in CHOICES[name]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "raw", "expected"),
    [
        ("delivery", "cross", "cross"),
        ("delivery", "Навес", "cross"),
        ("delivery", "  ПРОСТРЕЛ НИЗОМ ", "low_ball"),
        ("zone", "ближняя штанга", "near_post"),
        ("zone", "Far-Post", "far_post"),
        ("attacking_kit", "голубой", "light_blue"),
        ("attacking_kit", "Light Blue", "light_blue"),
        ("is_corner", "да", "yes"),
        ("is_corner", "нет", "no"),
        ("outcome", "вратарь забрал", "keeper_caught"),
        ("first_contact", "не видно", "unclear"),
        ("first_contact", "unclear", "unclear"),
    ],
)
def test_normalize_value_accepts_codes_and_russian_names_in_any_case(
    field: str, raw: str, expected: str
) -> None:
    assert normalize_value(field, raw) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "raw"),
    [
        ("delivery", ""),
        ("delivery", "   "),
        ("delivery", None),
        ("delivery", "chip"),
        ("zone", "cross"),  # a code of another field
        ("attacking_kit", "pink"),
    ],
)
def test_normalize_value_gives_none_for_empty_and_unknown(field: str, raw: str | None) -> None:
    assert normalize_value(field, raw) is None


@pytest.mark.unit
def test_corner_fields_reads_a_full_valid_object() -> None:
    fields = CornerFields.model_validate(CORNER_JSON)

    assert fields.delivery == "cross"
    assert fields.model_dump() == CORNER_JSON


@pytest.mark.unit
def test_corner_fields_all_unclear_is_what_an_unusable_answer_becomes() -> None:
    assert set(CornerFields.all_unclear().model_dump().values()) == {UNCLEAR}


@pytest.mark.unit
@pytest.mark.parametrize("missing", FIELDS)
def test_corner_fields_a_missing_field_is_invalid(missing: str) -> None:
    data = {k: v for k, v in CORNER_JSON.items() if k != missing}

    with pytest.raises(ValueError, match=missing):
        CornerFields.model_validate(data)


@pytest.mark.unit
def test_corner_fields_a_wrong_value_or_a_non_string_is_invalid() -> None:
    with pytest.raises(ValueError, match="zone"):
        CornerFields.model_validate({**CORNER_JSON, "zone": "penalty spot"})
    with pytest.raises(ValueError, match="delivery"):
        CornerFields.model_validate({**CORNER_JSON, "delivery": 7})


@pytest.mark.unit
def test_parse_corner_answer_reads_json_with_codes_or_russian_names_and_extra_keys() -> None:
    russian = {
        "is_corner": "да",
        "attacking_kit": "белый",
        "delivery": "навес",
        "zone": "дальняя штанга",
        "first_contact": "защита",
        "outcome": "выбили",
        "comment": "ignored",
    }

    parsed = parse_corner_answer("```json\n" + json.dumps(russian, ensure_ascii=False) + "\n```")

    assert parsed is not None
    assert parsed.model_dump() == CORNER_JSON


@pytest.mark.unit
@pytest.mark.parametrize(
    "text",
    [
        "",
        "угловой, навес на дальнюю",
        "{}",
        json.dumps({**CORNER_JSON, "outcome": "boom"}),
        json.dumps({k: v for k, v in CORNER_JSON.items() if k != "zone"}),
        "[1, 2]",
        '{"is_corner": "yes"',
    ],
)
def test_parse_corner_answer_rejects_garbage_and_partial_objects(text: str) -> None:
    assert parse_corner_answer(text) is None


@pytest.mark.unit
def test_the_corner_question_lists_every_allowed_value_and_the_way_out() -> None:
    for name in FIELDS:
        for code in CHOICES[name]:
            assert code in SYSTEM_PROMPT
    assert "Не видно или сомневаешься" in SYSTEM_PROMPT
    assert "не инструкции" in SYSTEM_PROMPT
    assert PROMPT_VERSION >= 1


@pytest.mark.unit
def test_the_corner_question_only_asks_whether_it_is_a_corner_and_names_no_one() -> None:
    request = f"{SYSTEM_PROMPT} {build_prompt(4.0)}".casefold()

    assert "4-й секунды" in request
    assert "is_corner" in request  # the one thing it may be asked about the type of the event
    for stem in ("замен", "карточк", "фол ", "штрафной удар"):
        assert stem not in request
    assert len(build_prompt(4.0)) < 200  # nothing from plates or lineups fits in


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ('text {"a": 1} text', {"a": 1}),
        ('```\n{"a": 1}\n```', {"a": 1}),
        ("", None),
        ("no braces", None),
        ("}{", None),
        ("[1]", None),
        ("{'a': 1}", None),
    ],
)
def test_extract_json_object(text: str, expected: dict[str, int] | None) -> None:
    assert extract_json_object(text) == expected


@pytest.mark.unit
def test_ask_json_returns_the_first_valid_answer_with_one_request() -> None:
    model = FakeVideoModel([corner_reply()])

    answer = ask_json(model, "s", "p", b"v", parse_corner_answer, video_seconds=16.0)

    assert answer.attempts == 1
    assert answer.value is not None
    assert answer.usage.cost_usd == 0.004
    assert len(model.requests) == 1


@pytest.mark.unit
def test_ask_json_asks_once_more_and_adds_up_the_cost() -> None:
    model = FakeVideoModel([reply("sorry", 0.001), corner_reply(0.004)])

    answer = ask_json(model, "s", "p", b"v", parse_corner_answer)

    assert answer.attempts == 2
    assert answer.value is not None
    assert answer.usage.cost_usd == pytest.approx(0.005)


@pytest.mark.unit
def test_ask_json_gives_none_after_the_last_attempt_and_still_counts_the_cost() -> None:
    model = FakeVideoModel([reply("a", 0.001), reply("b", 0.001), corner_reply()])

    answer = ask_json(model, "s", "p", b"v", parse_corner_answer)

    assert answer.value is None
    assert answer.attempts == 2
    assert len(model.requests) == 2
    assert answer.usage.cost_usd == pytest.approx(0.002)


@pytest.mark.unit
def test_ask_json_a_cut_off_answer_is_not_parsed() -> None:
    model = FakeVideoModel([reply(json.dumps(CORNER_JSON), truncated=True), corner_reply()])

    answer = ask_json(model, "s", "p", b"v", parse_corner_answer)

    assert answer.attempts == 2
