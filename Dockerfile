FROM python:3.12-slim

WORKDIR /app

ENV VERBOSE=1

# libportaudio2: required by sounddevice (voice mode)
# libpulse0 + libasound2-plugins: allow ALSA to route through PulseAudio over TCP (macOS Docker)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libportaudio2 \
    libpulse0 \
    libasound2-plugins \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml .
# Stub so setuptools can resolve the local package during dep install.
# This layer is cached as long as pyproject.toml doesn't change.
RUN mkdir -p orderbot && touch orderbot/__init__.py && \
    uv pip install --system ".[all]" && \
    rm -rf orderbot

COPY . .
# Reinstall only the local package (deps already cached above)
RUN uv pip install --system --no-deps .

COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["docker-entrypoint.sh"]

# Default: text mode. Voice mode: docker run --device /dev/snd ... python main.py --voice
CMD ["python", "main.py"]
