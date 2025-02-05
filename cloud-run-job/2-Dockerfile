# Use a lighter base image
FROM python:3.12.8-slim as builder

# Set environment variables to reduce unnecessary files
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.6.1

# Install necessary system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends libsqlite3-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install "poetry==$POETRY_VERSION"

# Set the working directory
WORKDIR /code

# Copy only the necessary files for dependency installation
COPY pyproject.toml poetry.lock README.md ./

# Install dependencies in a virtual environment
RUN poetry config virtualenvs.create true && \
    poetry install --no-interaction --no-ansi

# -- PRODUCTION STAGE -- 
FROM python:3.12.8-slim

# Set working directory
WORKDIR /code

# Copy required system libraries
COPY --from=builder /usr/lib/x86_64-linux-gnu/libsqlite3.so.0 /usr/lib/x86_64-linux-gnu/

# Copy installed dependencies from builder stage
COPY --from=builder /root/.cache/pypoetry/virtualenvs /opt/venv

# Set path to virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Copy the application code
COPY ./app ./app

# Run the application
CMD ["python3", "app/main.py"]
