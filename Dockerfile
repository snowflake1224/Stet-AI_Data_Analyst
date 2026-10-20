# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder
WORKDIR /build
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir hatchling && \
    pip install --no-cache-dir \
        "openai>=1.30.0" \
        "pdfplumber>=0.11" \
        "pillow>=10.0" \
        "pandas>=2.0" \
        "numpy>=1.26" \
        "openpyxl>=3.1" \
        "python-dotenv>=1.0"

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY src/ src/
COPY main.py .
COPY ask.py serve.py ./
COPY examples/ examples/
RUN mkdir -p /input /output /logs
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
# Competition default. A portfolio host can override with: python serve.py
CMD ["python", "main.py"]
