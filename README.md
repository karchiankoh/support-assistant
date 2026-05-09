# Support Assistant API

FastAPI backend for uploading support tickets and logs to the OpenAI Responses API. The API returns a concise issue summary, likely cause, evidence, and suggested debugging steps.

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
```

## Run

```bash
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API docs.

## Endpoints

- `GET /health` checks the service and configured model.
- `POST /analyze` accepts a ticket, one or more logs, or both.

Example:

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -F "ticket=@data/tickets/ticket1.txt" \
  -F "logs=@data/logs/main1.log"
```
