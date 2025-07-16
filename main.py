from fastapi import FastAPI
from pydantic import BaseModel
import openai, os, requests
from fastapi.middleware.cors import CORSMiddleware

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JUDGE0_API = "https://judge0-ce.p.rapidapi.com/submissions?base64_encoded=false&wait=true"
JUDGE0_HEADERS = {
    "X-RapidAPI-Key": os.getenv("RAPIDAPI_KEY"),
    "X-RapidAPI-Host": "judge0-ce.p.rapidapi.com",
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
        response = requests.post(JUDGE0_API, json=payload, headers=JUDGE0_HEADERS)
        output = response.json().get("stdout", "") or response.json().get("stderr", "Error running code.")

    prompt = generate_prompt(req.message, req.code, req.language)
    if output:
        prompt += f"\n\nExpected output:\n{output}"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    reply = response.choices[0].message.content.strip()
    return {"reply": reply, "output": output}

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
