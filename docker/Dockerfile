FROM python:3.13-slim

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY README.md .
COPY src/ src/

# Install dependencies with uv
RUN uv venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
RUN uv pip install -e .

# Expose Gradio port
EXPOSE 7860

# Default command - Run Gradio UI to keep container alive
CMD ["python", "-m", "market_analyst.app"]
