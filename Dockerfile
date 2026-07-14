FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_RETRIES=20

WORKDIR /app

COPY requirements.txt .

# CPU-only PyTorch pehle install karo
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Baqi requirements install karo
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY day2/ ./day2/
COPY day3/ ./day3/
COPY day4/ ./day4/
COPY day5/ ./day5/
COPY day6/ ./day6/
COPY day7/ ./day7/
COPY day8/ ./day8/
COPY day9/ ./day9/
COPY day10/ ./day10/
COPY day11/ ./day11/

EXPOSE 8000

CMD ["python", "day10/persistent_api.py"]