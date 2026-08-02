FROM python:3.13-slim

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /usr/local/bin/uv

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY uv.lock .
COPY README.md .
COPY src/ src/
COPY skills/ skills/

# Install dependencies with uv
ENV PATH="/app/.venv/bin:$PATH"
RUN uv sync --frozen --no-dev

# Create non-root user for security
RUN useradd -m analyst && chown -R analyst /app
USER analyst
ENV PYTHONUNBUFFERED=1

# Expose Gradio port
EXPOSE 7860

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860')" || exit 1

# Default command - Run Gradio UI to keep container alive
CMD ["python", "-m", "market_analyst.app"]
