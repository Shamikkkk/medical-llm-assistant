# syntax=docker/dockerfile:1.7
# Created by Codex - Section 1
FROM node:20-bookworm-slim AS frontend-build

WORKDIR /frontend

COPY frontend/package*.json ./
RUN --mount=type=cache,target=/root/.npm npm ci

COPY frontend/ ./
RUN npm run build -- --configuration production

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    API_PORT=8000

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    PIP_NO_CACHE_DIR=0 pip install --upgrade pip && \
    PIP_NO_CACHE_DIR=0 pip install -r requirements.txt

COPY . .
COPY --from=frontend-build /frontend/dist ./frontend/dist

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
