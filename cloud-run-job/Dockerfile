FROM python:3.12.8-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsqlite3-0 \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install poetry==1.6.1

RUN poetry config virtualenvs.create false

WORKDIR /code

COPY ./pyproject.toml ./README.md ./poetry.lock* ./
RUN poetry install  --no-interaction --no-ansi --no-root

COPY ./app ./app

RUN poetry cache clear pypi --all && rm -rf /root/.cache/pypoetry
CMD ["python3", "app/main.py"]