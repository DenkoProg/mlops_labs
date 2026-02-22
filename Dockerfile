# ─────────────────────────────────────────
# Stage 1: builder — install dependencies
# ─────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN pip install uv

WORKDIR /app

# Copy only dependency manifests first (layer-cache friendly)
COPY pyproject.toml uv.lock ./

# Install production dependencies into an isolated venv
RUN uv sync --frozen --no-dev --no-install-project

# ─────────────────────────────────────────
# Stage 2: runtime — lean production image
# ─────────────────────────────────────────
FROM python:3.11-slim AS runtime

# System dependencies for Pillow / numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the pre-built venv from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application source and config
COPY src/ ./src/
COPY conf/ ./conf/

# Make uv venv the default Python
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# Default entrypoint — can be overridden at runtime
ENTRYPOINT ["python"]
CMD ["src/train.py", "--help"]
