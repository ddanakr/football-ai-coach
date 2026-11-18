"""Fixed parameters of the T-DEED run (D-080). They reproduce the frozen predictions (D-052).

Where a value comes from is stated beside it. Nothing here is tuned: T-DEED was trained at these
sizes and rates, so changing them would not be a setting but a different experiment.
"""

# The repository and the commit its code was read at (D-042); the full hash of `d953b4d`. The
# commit and the file hash below are public identifiers, not secrets (the pragmas say so).
REPO_URL = "https://github.com/arturxe2/T-DEED.git"
REPO_COMMIT = "d953b4de8a2ad941c565e1683c5b07255cbe644e"  # pragma: allowlist secret

# The model of the frozen predictions (D-052): the checkpoint `SoccerNet_small`.
MODEL = "SoccerNet_small"
MODEL_DATASET = "SoccerNet"  # `inference.py` takes the part of the model name before `_`
# The file in the public Google Drive folder that the T-DEED README links to (listed 2026-11-18);
# the direct form of the link, which serves a file under 100 MB without a confirmation page.
CHECKPOINT_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=1Hl5QS6cXZurUYEC-XkONBC1Yec9WFEdm&export=download&confirm=t"
)
CHECKPOINT_SHA256 = (
    "69b319badc5dcdf5026e7769f1276d947aeec8a822bf6f172852c67089cc61ea"  # pragma: allowlist secret
)

# The input: frames of the size and rate T-DEED was trained at (README T-DEED, D-042).
FRAME_WIDTH = 796
FRAME_HEIGHT = 448
FPS = 25
# The threshold of `inference.py` itself (D-045) and the batch of the frozen file (its name).
THRESHOLD = 0.05
BATCH_SIZE = 2
# `inference.py` reads the video in every data-loader worker (an iterable dataset without
# sharding): four workers (the config's value) do the same work four times and sum identical
# scores that are then divided by the same count, so one worker gives the same output.
NUM_WORKERS = 1

# The model looks at every 12th frame, 100 of them at a time, and moves its window by 50 samples.
SAMPLE_STEP_FRAMES = 12
WINDOW_STEP_S = SAMPLE_STEP_FRAMES * 50 / FPS  # 24 s
# A segment starts on this grid, so that its windows are those of the whole-match run.
SEGMENT_GRID_S = WINDOW_STEP_S
# The last, incomplete window of a video is not predicted (D-042); the segment is cut this much
# beyond its end so that its own end is predicted as in the whole-match run.
SEGMENT_TAIL_S = 60.0
MIN_SEGMENT_S = 60.0

# ffmpeg: the quality of the 796x448 copy (visually lossless; the model looks at fine detail).
INPUT_CRF = 18
INPUT_PRESET = "veryfast"
FFMPEG_TIMEOUT_S = 4 * 3600.0
GIT_TIMEOUT_S = 600.0
DOWNLOAD_TIMEOUT_S = 60.0
INFERENCE_TIMEOUT_S = 12 * 3600.0

OUTPUT_PREFIX = "tdeed_"  # `tdeed_*.json` is what `ingest`, `fuse` and `final-eval` look for
SEGMENT_PREFIX = "tdeedseg_"  # a segment file must not match the pattern above

NOTEBOOK_URL = (
    "https://colab.research.google.com/github/ddanakr/football-ai-coach/blob/main/"
    "notebooks/tdeed_colab.ipynb"
)
