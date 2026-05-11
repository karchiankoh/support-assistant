# Support Assistant API

FastAPI backend for uploading support tickets and logs to the OpenAI Responses API. The API returns a concise issue summary, likely cause, evidence, and suggested debugging steps.

Note: this project is vibe coded with Codex.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your-api-key"
```

Optional:

```bash
export OPENAI_MODEL="gpt-4.1-mini"
export OPENAI_EMBEDDING_MODEL="text-embedding-3-small"
export SUPPORT_KB_DB_PATH="data/support_knowledge.db"
export LOG_LEVEL="INFO"
```

Logs are emitted as JSON. Each HTTP response includes an `X-Request-ID` header, and callers can pass
their own `X-Request-ID` to correlate all logs for a request.

Successful OpenAI completion logs include token usage fields from the API response:
`input_tokens`, `output_tokens`, `total_tokens`, `cached_input_tokens`, and `reasoning_tokens`.

For local development without calling the OpenAI API:

```bash
export APP_ENV="dev"
```

## Run

```bash
fastapi dev
```

Open `http://127.0.0.1:8000/docs` for the interactive API docs.

## Docker

Build and run the API with Docker Compose:

```bash
docker compose up --build
```

Open `http://127.0.0.1:8000/docs` for the interactive API docs.

The Compose setup mounts `./data` into the container so the local SQLite knowledge database persists across runs. Configure the app with the same environment variables shown above, either in your shell or in a local `.env` file.

Build or refresh the knowledge index inside Docker:

```bash
docker compose run --rm api python scripts/ingest_knowledge.py
```

## Endpoints

- `GET /health` checks the service, configured model, and local knowledge database status.
- `POST /analyze` accepts a ticket, one or more logs, or both.

Example:

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -F "ticket=@data/tickets/ticket1.txt" \
  -F "logs=@data/logs/main1.log"
```

## SQLite Retrieval

Reusable support knowledge, such as runbooks and past incidents, is embedded with OpenAI and stored locally in SQLite. The `/analyze` endpoint retrieves matching chunks automatically and passes them to the Responses API as hidden support context.

Build or refresh the local index:

```bash
python scripts/ingest_knowledge.py
```

By default, the script indexes:

- `data/runbooks`
- `data/incidents`

Retrieval settings:

```bash
export SUPPORT_KB_TOP_K="4"
export SUPPORT_KB_MIN_SCORE="0.2"
```

OpenAI retry settings:

```bash
export OPENAI_RETRY_ATTEMPTS="3"
export OPENAI_RETRY_BACKOFF_SECONDS="0.5"
```
