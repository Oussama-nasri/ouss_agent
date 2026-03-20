"""
Minimal FastAPI server — bridges the HTML frontend to the agent.

Run:
    pip install fastapi uvicorn
    python server.py
Then open index.html in your browser.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from llm.ollama    import OllamaLLM
from agent.core    import Agent
from agent.prompts import build_system_prompt

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared agent instance
llm   = OllamaLLM()
agent = Agent(llm=llm, system=build_system_prompt())


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
def chat(req: ChatRequest):
    response = agent(req.message)
    return {"response": response}


@app.post("/clear")
def clear():
    agent.memory.clear()
    return {"status": "memory cleared"}


@app.get("/history")
def history():
    messages = agent.memory.get_all()
    return {"messages": [m for m in messages if m["role"] != "system"]}


@app.get("/health")
def health():
    return {"status": "ok", "model": agent.llm.model}


