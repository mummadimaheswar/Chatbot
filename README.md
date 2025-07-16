# DevBot FastAPI Service

This repository contains a simple FastAPI backend that can:

1. Evaluate code snippets in multiple languages through the public **Judge0** API.
2. Generate answers using a local LLM served by **Ollama**.

---

## Local Development

### 1. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Start Ollama (optional)

If you want to use the AI functionality, you need an Ollama instance running *locally*:
```bash
ollama serve &
ollama pull llama3
```

> Alternatively, set different values for `LLM_URL` and `LLM_MODEL`.

### 3. Run the API
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The service exposes:
- `GET /` – health-check endpoint.
- `POST /chat` – main endpoint for code execution + LLM answer.

---

## Deployment

### Docker
A ready-to-use `Dockerfile` is included. Build and run:
```bash
docker build -t devbot-api .
docker run -p 8000:8000 devbot-api
```

Set optional environment variables:
- `PORT` – port uvicorn should listen on (default: 8000)
- `LLM_URL` – URL of the Ollama `generate` endpoint
- `LLM_MODEL` – model name to use (default: `llama3`)

### Render / Heroku / Railway

These platforms detect Python apps automatically. You just need a **Procfile** (already added):
```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

Make sure to add `PORT`, `LLM_URL`, and `LLM_MODEL` as environment variables in your dashboard.

---

## Troubleshooting

1. **Cannot reach `/chat`** – check logs; ensure the container is running and listening on the correct port.
2. **`Error contacting local LLM`** – verify that the Ollama server is reachable from inside the container / deployment.
3. **`Error contacting Judge0 API`** – the public Judge0 endpoint might have rate-limits; consider self-hosting or adding an API key with higher limits.

---

Feel free to open an issue or PR if you encounter problems.