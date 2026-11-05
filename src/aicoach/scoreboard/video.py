"""Reading frames from the match video in one sequential pass.

`t_v` is `frame_index / fps` by the owner's rule: the broadcast is assumed to have a constant
frame rate, and a mismatch between the decoded frame count and the metadata is reported by the
stage (`duration_tol_s`). Frames between two samples are skipped with `grab()`, which decodes
without converting; only sampled frames are converted with `retrieve()`. The decoding mode
(GPU or CPU) is recorded because colours differ slightly between modes: calibration and the run
must use the same one.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from aicoach.errors import ScoreboardParseError
from aicoach.scoreboard.schemas import DecodeMode

Frame = NDArray[np.uint8]
DECODE_GPU: DecodeMode = "opencv_gpu"
DECODE_CPU: DecodeMode = "opencv_cpu"


@dataclass(frozen=True)
class VideoInfo:
    """Container metadata. `duration_s` is `frame_count / fps`, the metadata duration."""

    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        """Duration according to the metadata."""
        return self.frame_count / self.fps


def open_capture(path: Path, hw_accel: bool = True) -> cv2.VideoCapture:
    """Open a video for reading. With `hw_accel` the decoder may use the GPU and falls back
    to the CPU by itself (OpenCV's software AV1 decoder is far slower)."""
    if hw_accel:
        return cv2.VideoCapture(
            str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY]
        )
    return cv2.VideoCapture(str(path))


def decode_mode_of(cap: cv2.VideoCapture) -> DecodeMode:
    """`opencv_gpu` if the opened capture really decodes on the GPU, otherwise `opencv_cpu`."""
    accel = int(cap.get(cv2.CAP_PROP_HW_ACCELERATION))
    return DECODE_CPU if accel == cv2.VIDEO_ACCELERATION_NONE else DECODE_GPU


def probe_video(path: Path) -> VideoInfo:
    """Read the metadata of a video file.

    Raises:
        ScoreboardParseError: the file is missing, cannot be opened or has no usable metadata.
    """
    if not path.is_file():
        msg = f"video file not found: {path}"
        raise ScoreboardParseError(msg)
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            msg = f"cannot open video (broken file?): {path}"
            raise ScoreboardParseError(msg)
        info = VideoInfo(
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()
    if not (info.fps > 0 and info.frame_count > 0 and info.width > 0 and info.height > 0):
        msg = f"video has no usable metadata: {path}"
        raise ScoreboardParseError(msg)
    return info


def frame_step(fps: float, sample_fps: float) -> int:
    """Frames between two samples: `round(fps / sample_fps)`, never below 1."""
    if not (fps > 0 and sample_fps > 0):
        msg = "fps and sample_fps must be positive"
        raise ScoreboardParseError(msg)
    return max(1, round(fps / sample_fps))


def sample_count(frame_count: int, step: int) -> int:
    """Number of sampled frames in a video of `frame_count` frames (frames 0, step, 2*step...)."""
    if frame_count <= 0 or step <= 0:
        return 0
    return (frame_count + step - 1) // step


class SampledReader:
    """Yields `(sample_index, frame)` for frames 0, step, 2*step... in one sequential pass.

    After the pass `frames_decoded` is the real number of frames in the file (compare it with
    the metadata to catch a truncated or damaged file) and `decode_mode` says how it was
    decoded. A file that ends or breaks early simply ends the iteration. Before the pass
    `decode_mode` is only the requested mode.
    """

    def __init__(self, path: Path, step: int, hw_accel: bool = True) -> None:
        """Remember the file and the sampling step; nothing is opened until iteration.

        Raises:
            ScoreboardParseError: `step` is below 1.
        """
        if step < 1:
            msg = f"sampling step must be at least 1 frame, got {step}"
            raise ScoreboardParseError(msg)
        self._path = path
        self._step = step
        self._hw_accel = hw_accel
        self.frames_decoded = 0
        self.decode_mode: DecodeMode = DECODE_GPU if hw_accel else DECODE_CPU

    def __iter__(self) -> Iterator[tuple[int, Frame]]:
        """Read the whole file.

        Raises:
            ScoreboardParseError: the file cannot be opened.
        """
        cap = open_capture(self._path, self._hw_accel)
        try:
            if not cap.isOpened():
                msg = f"cannot open video (broken file?): {self._path}"
                raise ScoreboardParseError(msg)
            self.decode_mode = decode_mode_of(cap)
            self.frames_decoded = 0
            while cap.grab():
                if self.frames_decoded % self._step == 0:
                    ok, frame = cap.retrieve()
                    if ok:
                        yield self.frames_decoded // self._step, np.asarray(frame, dtype=np.uint8)
                self.frames_decoded += 1
        finally:
            cap.release()
