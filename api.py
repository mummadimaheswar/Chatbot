# api.py
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np
from chatbot_model import build_seq2seq, translate_sentence, MAX_LEN, VOCAB_SIZE

# Placeholder model (replace with trained weights)
model, encoder_model, decoder_model = build_seq2seq(VOCAB_SIZE, 100, 300)

app = FastAPI()

class Query(BaseModel):
    text: str

@app.post("/chat")
def chat(query: Query):
    # Dummy example (replace with actual tokenizer & preprocessing)
    seq = np.zeros((1, MAX_LEN))
    response = translate_sentence(seq, encoder_model, decoder_model, {1:"<bos>", 2:"<eos>"}, {1:"Hi", 2:"Bye"})
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
