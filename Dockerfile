FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY samples ./samples
RUN pip install --no-cache-dir -e .

EXPOSE 8000
CMD ["python", "-m", "local_rag", "serve", "--host", "0.0.0.0", "--port", "8000"]
