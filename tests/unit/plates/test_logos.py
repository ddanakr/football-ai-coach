"""Grouping logos by similarity."""

import numpy as np
import pytest

from aicoach.errors import PlatesError
from aicoach.plates.logos import cluster_logos, logo_distance, logo_vector
from aicoach.scoreboard.video import Frame

RED = (40, 40, 220)
YELLOW = (30, 220, 240)
GREEN = (60, 200, 40)


def logo(colour: tuple[int, int, int], size: int = 50, noise: float = 0.0, seed: int = 0) -> Frame:
    """A square logo crop: a blue field with a coloured disc."""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:] = (163, 76, 8)
    centre = size // 2
    ys, xs = np.ogrid[:size, :size]
    image[(ys - centre) ** 2 + (xs - centre) ** 2 <= (size // 3) ** 2] = colour
    if noise:
        rng = np.random.default_rng(seed)
        noisy = np.clip(image + rng.normal(0, noise, image.shape), 0, 255)
        image = np.asarray(noisy, dtype=np.uint8)
    return image


@pytest.mark.unit
def test_cluster_logos_two_teams_give_two_clusters() -> None:
    colours = [RED, YELLOW, RED, YELLOW]
    vectors = [logo_vector(logo(c, noise=4, seed=i)) for i, c in enumerate(colours)]

    assert cluster_logos(vectors, 0.07) == [0, 1, 0, 1]


@pytest.mark.unit
def test_cluster_logos_larger_cluster_gets_id_zero() -> None:
    vectors = [logo_vector(logo(c)) for c in (YELLOW, RED, RED, RED)]

    assert cluster_logos(vectors, 0.07) == [1, 0, 0, 0]


@pytest.mark.unit
def test_cluster_logos_different_sizes_of_the_same_logo_are_one_team() -> None:
    vectors = [logo_vector(logo(RED, size=50)), logo_vector(logo(RED, size=80))]

    assert cluster_logos(vectors, 0.07) == [0, 0]


@pytest.mark.unit
def test_cluster_logos_empty_gives_empty() -> None:
    assert cluster_logos([], 0.07) == []


@pytest.mark.unit
def test_cluster_logos_single_logo_is_one_cluster() -> None:
    assert cluster_logos([logo_vector(logo(RED))], 0.07) == [0]


@pytest.mark.unit
def test_cluster_logos_three_different_logos_give_three_clusters() -> None:
    vectors = [logo_vector(logo(c)) for c in (RED, YELLOW, GREEN)]

    assert sorted(cluster_logos(vectors, 0.07)) == [0, 1, 2]


@pytest.mark.unit
def test_cluster_logos_grouping_does_not_depend_on_the_order() -> None:
    colours = [RED, YELLOW, RED, GREEN, YELLOW, RED]
    vectors = [logo_vector(logo(c, noise=3, seed=i)) for i, c in enumerate(colours)]
    order = [3, 0, 5, 1, 4, 2]

    straight = cluster_logos(vectors, 0.07)
    shuffled = cluster_logos([vectors[i] for i in order], 0.07)

    def groups(labels: list[int], idx: list[int]) -> set[frozenset[int]]:
        return {frozenset(idx[j] for j, lab in enumerate(labels) if lab == c) for c in set(labels)}

    assert groups(straight, list(range(6))) == groups(shuffled, order)


@pytest.mark.unit
def test_cluster_logos_equal_sized_clusters_are_numbered_by_first_appearance() -> None:
    vectors = [logo_vector(logo(c)) for c in (YELLOW, RED)]

    assert cluster_logos(vectors, 0.07) == [0, 1]


@pytest.mark.unit
def test_logo_distance_is_zero_for_the_same_logo_and_large_for_different_ones() -> None:
    a, b = logo_vector(logo(RED)), logo_vector(logo(YELLOW))

    assert logo_distance(a, a) == 0.0
    assert logo_distance(a, b) > 0.07


@pytest.mark.unit
def test_logo_vector_empty_crop_raises() -> None:
    with pytest.raises(PlatesError, match="no pixels"):
        logo_vector(np.zeros((0, 0, 3), dtype=np.uint8))
