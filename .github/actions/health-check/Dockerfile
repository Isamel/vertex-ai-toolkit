FROM python:3.12-slim

RUN groupadd -r vaig && useradd -r -g vaig vaig

WORKDIR /action

COPY pyproject.toml README.md ./
COPY src/ src/

RUN pip install --no-cache-dir ".[live]" && rm -rf /root/.cache

COPY .github/actions/health-check/entrypoint.py /action/entrypoint.py

USER vaig

ENTRYPOINT ["python", "/action/entrypoint.py"]
