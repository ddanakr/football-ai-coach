# Football AI Coach: viewer, ingest and evaluation commands without installing anything by hand.
# CPU only: no CUDA, no TensorFlow. T-DEED is not here, it runs separately in Colab (D-046).
# The tag names the Debian release: the package names below were checked for trixie only.
FROM python:3.11-slim-trixie

# Debian packages (checked on packages.debian.org, trixie):
#   ffmpeg               clips, reels, audio extraction (ffprobe comes with it)
#   tesseract-ocr        plate names; tesseract-ocr-kaz and tesseract-ocr-rus are the languages
#   fonts-dejavu-core    DejaVu Sans with Cyrillic and Kazakh letters for reel captions; the code
#                        looks for /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf (reel.py)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        tesseract-ocr \
        tesseract-ocr-kaz \
        tesseract-ocr-rus \
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# An unprivileged user; /data is the mount point of the match directories.
RUN useradd --create-home --uid 1000 app \
    && mkdir /data \
    && chown app:app /data

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    AICOACH_DATA_ROOT=/data \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Only what .dockerignore lets through: package sources and the labels. No keys, data or video.
COPY pyproject.toml README.md ./
COPY src ./src
COPY data_labels ./data_labels
RUN pip install .

USER app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request as u; u.urlopen('http://127.0.0.1:%s/_stcore/health' % os.environ['STREAMLIT_SERVER_PORT'], timeout=3)"]

ENTRYPOINT ["python", "-m", "aicoach.cli"]
CMD ["ui"]
