
from fastapi import FastAPI
from pydantic import BaseModel
import os
import logging
import requests
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Using Judge0 open API (no RapidAPI key required)
JUDGE0_API = "https://api.judge0.com/submissions?base64_encoded=false&wait=true"
JUDGE0_HEADERS = {
    "Content-Type": "application/json"
}

language_map = {
    "python": 71,
    "cpp": 54,
    "c": 50,
    "java": 62,
    "r": 80
}

class ChatRequest(BaseModel):
    message: str
    code: str = ""
    language: str = "python"

@app.post("/chat")
async def chat(req: ChatRequest):
    output = ""
    if req.code:
        payload = {
            "language_id": language_map[req.language.lower()],
            "source_code": req.code,
            "stdin": ""
        }
        try:
            response = requests.post(JUDGE0_API, json=payload, headers=JUDGE0_HEADERS, timeout=30)
            response.raise_for_status()
            output = response.json().get("stdout", "") or response.json().get("stderr", "Error running code.")
        except Exception as e:
            logging.exception("Failed to execute code via Judge0")
            output = f"Error contacting Judge0 API: {e}"

    prompt = generate_prompt(req.message, req.code, req.language)
    if output:
        prompt += f"\n\nExpected output:\n{output}"

    # Call to local LLM via Ollama
    reply = query_local_llm(prompt)
    return {"reply": reply, "output": output}

def query_local_llm(prompt):
    """Query the local Ollama LLM instance.

    The endpoint URL and model can be overridden via the `LLM_URL` and
    `LLM_MODEL` environment variables to make the service configurable in
    different deployment environments.
    """

    llm_url = os.getenv("LLM_URL", "http://localhost:11434/api/generate")
    llm_model = os.getenv("LLM_MODEL", "llama3")

    try:
        response = requests.post(
            llm_url,
            json={
                "model": llm_model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        logging.exception("Failed to query local LLM")
        return f"Error contacting local LLM: {e}"

@app.get("/")
async def health_check():
    """Simple health-check endpoint useful for monitoring & load-balancers."""
    return {"status": "ok"}

# Allow running with `python main.py` locally
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

def generate_prompt(user_input, code="", language=""):
    base = """
You are DevBot — an expert AI assistant for developers.

Capabilities:
- Explain and debug code (Python, C, C++, Java, R)
- Teach AI/ML concepts interactively
- Compile and run code on server
- Provide expert-level assistance with examples

Instructions:
- Respond clearly and professionally.
- Format code in Markdown.
- If code is provided, explain it and give expected output.

Query:
"""
    code_block = f"\n\nCode in {language}:\n```{language}\n{code}\n```" if code else ""
    return base + user_input + code_block
