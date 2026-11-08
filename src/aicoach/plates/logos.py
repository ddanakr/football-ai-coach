"""Group plate logos by similarity: one group is one team (pure functions)."""

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from aicoach.errors import PlatesError
from aicoach.scoreboard.video import Frame

_PATCH = 24  # logos are compared as 24x24 colour patches

LogoVector = NDArray[np.float32]


def logo_vector(logo: Frame) -> LogoVector:
    """A 24x24 colour patch of the logo crop in 0..1, flattened.

    Raises:
        PlatesError: the crop has no pixels.
    """
    if logo.size == 0:
        msg = "logo crop has no pixels"
        raise PlatesError(msg)
    small = cv2.resize(logo, (_PATCH, _PATCH), interpolation=cv2.INTER_AREA)
    return np.asarray(small, dtype=np.float32).ravel() / 255.0


def logo_distance(a: LogoVector, b: LogoVector) -> float:
    """Mean absolute difference of two logo vectors (0..1)."""
    return float(np.abs(a - b).mean())


def cluster_logos(vectors: Sequence[LogoVector], max_dist: float) -> list[int]:
    """Cluster id of every vector. A vector joins the nearest cluster centre if it is closer
    than `max_dist`, otherwise it starts a new cluster; the centre is the mean of its members.
    Ids are numbered by cluster size (largest first), ties by first appearance. The result does
    not depend on anything but the order of `vectors`.
    """
    centres: list[LogoVector] = []
    members: list[list[int]] = []
    for i, vector in enumerate(vectors):
        distances = [logo_distance(vector, c) for c in centres]
        best = int(np.argmin(distances)) if distances else -1
        if best >= 0 and distances[best] <= max_dist:
            members[best].append(i)
            centres[best] = np.mean([vectors[j] for j in members[best]], axis=0).astype(np.float32)
        else:
            centres.append(vector)
            members.append([i])
    order = sorted(range(len(members)), key=lambda c: (-len(members[c]), members[c][0]))
    rank = {cluster: new for new, cluster in enumerate(order)}
    labels = [0] * len(vectors)
    for cluster, indices in enumerate(members):
        for i in indices:
            labels[i] = rank[cluster]
    return labels
