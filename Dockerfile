FROM python:3.9

RUN pip install poetry

WORKDIR /app

EXPOSE 8000

COPY pyproject.toml poetry.lock /app/
COPY app /app/

RUN poetry config virtualenvs.create false \
    && poetry install --no-root

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


