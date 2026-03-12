FROM python:3.12-slim

WORKDIR /app

# libportaudio2: required by sounddevice (voice mode)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml .
RUN uv pip install --system ".[all]"

COPY . .

# Default: text mode. Voice mode: docker run --device /dev/snd ... python main.py --voice
CMD ["python", "main.py"]
