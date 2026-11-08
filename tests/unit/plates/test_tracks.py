"""Joining detections into plates."""

import pytest

from aicoach.plates.detect import FrameReading
from aicoach.plates.tracks import Hit, Track, build_tracks, card_color_of, count_overlaps
from tests.fixtures.plates_synth import hit, series


@pytest.mark.unit
def test_build_tracks_no_hits_gives_nothing() -> None:
    assert build_tracks([], 3.0, 0.02, 2) == ([], [])


@pytest.mark.unit
def test_build_tracks_one_run_is_one_plate() -> None:
    plates, rejected = build_tracks(series(10.0, 14.0), 3.0, 0.02, 2)

    assert rejected == []
    [plate] = plates
    assert (plate.kind, plate.t_first, plate.t_last, len(plate.hits)) == ("goal", 10.0, 14.0, 9)


@pytest.mark.unit
@pytest.mark.parametrize(("gap", "expected"), [(2.5, 1), (3.0, 1), (3.5, 2)])
def test_build_tracks_gap_decides_between_one_and_two_plates(gap: float, expected: int) -> None:
    hits = series(10.0, 12.0) + series(12.0 + gap, 14.0 + gap)

    plates, _ = build_tracks(hits, 3.0, 0.02, 2)

    assert len(plates) == expected


@pytest.mark.unit
def test_build_tracks_two_plates_of_different_types_in_a_row_do_not_merge() -> None:
    hits = series(10.0, 13.0, kind="card", color="yellow") + series(13.5, 16.0, kind="goal")

    plates, _ = build_tracks(hits, 3.0, 0.02, 2)

    assert [p.kind for p in plates] == ["card", "goal"]
    assert plates[0].t_last < plates[1].t_first


@pytest.mark.unit
def test_build_tracks_double_substitution_splits_where_the_panel_changes() -> None:
    first = series(10.0, 13.0, kind="subst", sig=0.2)
    second = series(13.5, 16.5, kind="subst", sig=0.6)

    plates, _ = build_tracks(first + second, 3.0, 0.02, 2)

    assert len(plates) == 2
    assert (plates[0].t_last, plates[1].t_first) == (13.0, 13.5)


@pytest.mark.unit
def test_build_tracks_small_picture_noise_does_not_split() -> None:
    hits = [hit(10.0 + i * 0.5, "subst", sig=0.2 + 0.003 * (i % 2)) for i in range(8)]

    plates, _ = build_tracks(hits, 3.0, 0.02, 2)

    assert len(plates) == 1


@pytest.mark.unit
def test_build_tracks_plate_at_the_start_and_at_the_end_of_the_video() -> None:
    hits = series(0.0, 2.0, kind="card", color="yellow") + series(98.0, 100.0)

    plates, _ = build_tracks(hits, 3.0, 0.02, 2)

    assert [(p.kind, p.t_first, p.t_last) for p in plates] == [
        ("card", 0.0, 2.0),
        ("goal", 98.0, 100.0),
    ]


@pytest.mark.unit
def test_build_tracks_single_hit_is_a_too_short_reject() -> None:
    plates, rejected = build_tracks([hit(5.0)], 3.0, 0.02, 2)

    assert plates == []
    [item] = rejected
    assert item.reject == "too_short"


@pytest.mark.unit
def test_build_tracks_single_hit_is_a_plate_when_min_hits_is_one() -> None:
    plates, rejected = build_tracks([hit(5.0)], 3.0, 0.02, 1)

    assert len(plates) == 1
    assert rejected == []


@pytest.mark.unit
def test_build_tracks_rejected_candidates_are_grouped_by_reason() -> None:
    hits = series(20.0, 22.0, kind=None, reject="no_panel") + series(
        30.0, 32.0, kind=None, reject="no_marker"
    )

    plates, rejected = build_tracks(hits, 3.0, 0.02, 2)

    assert plates == []
    assert [(r.reject, r.t_first) for r in rejected] == [("no_panel", 20.0), ("no_marker", 30.0)]


@pytest.mark.unit
def test_build_tracks_flicker_of_another_label_does_not_cut_a_plate() -> None:
    hits = series(10.0, 14.0)
    hits.append(hit(12.25, kind=None, reject="no_marker"))

    plates, rejected = build_tracks(hits, 3.0, 0.02, 2)

    assert len(plates) == 1
    assert len(plates[0].hits) == 9
    assert [r.reject for r in rejected] == ["no_marker"]


@pytest.mark.unit
def test_build_tracks_hits_given_out_of_order_are_sorted() -> None:
    hits = list(reversed(series(10.0, 12.0)))

    [plate], _ = build_tracks(hits, 3.0, 0.02, 2)

    assert [h.t_v for h in plate.hits] == sorted(h.t_v for h in plate.hits)


@pytest.mark.unit
def test_build_tracks_missing_signature_joins_by_gap_only() -> None:
    a = Hit(0, 0.0, FrameReading("goal", None, None, 1.0, 0.9))
    b = Hit(1, 0.5, FrameReading("goal", None, None, 1.0, 0.9))

    plates, _ = build_tracks([a, b], 3.0, 0.02, 2)

    assert len(plates) == 1


@pytest.mark.unit
def test_track_mid_hit_is_the_hit_nearest_to_the_middle() -> None:
    track = Track("goal", None, series(10.0, 14.0))

    assert track.mid_hit.t_v == 12.0


@pytest.mark.unit
def test_track_mid_hit_on_a_tie_takes_the_earlier_one() -> None:
    assert Track("goal", None, [hit(10.0), hit(11.0)]).mid_hit.t_v == 10.0


@pytest.mark.unit
def test_card_color_of_takes_the_most_frequent_colour() -> None:
    hits = [
        hit(1.0, "card", color="yellow"),
        hit(1.5, "card", color="other"),
        hit(2.0, "card", color="yellow"),
    ]

    assert card_color_of(Track("card", None, hits)) == "yellow"


@pytest.mark.unit
def test_card_color_of_non_card_or_no_colour_is_none() -> None:
    assert card_color_of(Track("goal", None, series(1.0, 2.0))) is None
    assert card_color_of(Track("card", None, series(1.0, 2.0, kind="card"))) is None


@pytest.mark.unit
def test_count_overlaps_counts_only_different_types_that_intersect() -> None:
    a = Track("goal", None, series(10.0, 14.0))
    b = Track("card", None, series(12.0, 16.0, kind="card"))
    c = Track("subst", None, series(20.0, 22.0, kind="subst"))
    d = Track("goal", None, series(21.0, 23.0))

    assert count_overlaps([a, b, c, d]) == 2
    assert count_overlaps([a, c]) == 0
    assert count_overlaps([]) == 0
