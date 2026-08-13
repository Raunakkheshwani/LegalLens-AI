FROM python:3.13-slim

WORKDIR /app

# System dependencies needed for some packages (pdf parsing, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast, reproducible dependency installs
RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock* ./
RUN uv pip install --system -r pyproject.toml || uv sync --frozen

COPY . .

# Directories the app writes to at runtime
RUN mkdir -p data/chroma_store data/raw_contracts data/uploaded_docs

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]