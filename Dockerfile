FROM python:3.11-slim

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
RUN mkdir -p data && chown -R user:user /home/user/app
USER user

ENV PORT=10000 PYTHONUNBUFFERED=1 BENCH_MAX_EPISODES=20
EXPOSE 10000

# One worker: matches and the matchmaking queue live in this process's memory.
CMD gunicorn --workers 1 --threads 64 --timeout 300 --bind 0.0.0.0:${PORT} server:app
