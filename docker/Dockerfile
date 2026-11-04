FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Minimal scientific stack; add scipy etc. as needed
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scikit-learn==1.5.1 \
    scipy==1.14.0

# Non-root user — prevents privilege escalation inside container
RUN useradd -m -u 1000 sandbox && mkdir -p /workspace && chown sandbox:sandbox /workspace
USER sandbox
WORKDIR /workspace

# Entrypoint is runner.py, mounted at runtime (no shell available to scripts)
ENTRYPOINT ["python", "/workspace/runner.py"]