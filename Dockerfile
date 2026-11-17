# Debian stable + Python stdlib + the standard Postgres client. No pip packages.
FROM debian:bookworm-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 postgresql-client ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY bot.py db.py ./
ENTRYPOINT ["python3", "/app/bot.py"]
