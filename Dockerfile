FROM bitnami/pytorch:2.5.1 as production

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

USER root
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY . /app

WORKDIR /app

# RUN pip install --no-cache-dir \
#     git+https://github.com/huggingface/transformers \
#     accelerate \
#     qwen-vl-utils[decord]==0.0.8
ENV UV_HTTP_TIMEOUT=500
RUN uv sync --frozen --no-cache

CMD [".venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8017"]
