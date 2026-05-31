from pathlib import Path
import json
import requests

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
AGENTS = ROOT / "agents"

app = FastAPI()
app.mount("/static", StaticFiles(directory=WEB), name="static")

# Later, replace localhost with each gallery PC's IP or Tailscale IP.
AGENT_CONFIG = {
    "alice": {
        "ollama_url": "http://localhost:11434",
        "model": "gemma3:4b",
    },
    "bob": {
        "ollama_url": "http://localhost:11434",
        "model": "llama3.2:latest",
    },
    "mallory": {
        "ollama_url": "http://localhost:11434",
        "model": "mistral:7b",
    },
}


@app.get("/")
def home():
    return FileResponse(WEB / "index.html")


def load_json(path: Path, fallback):
    if not path.exists():
        return fallback
    text = path.read_text().strip()
    if not text:
        return fallback
    return json.loads(text)


def read_optional_text(path: Path):
    if not path.exists():
        return ""
    return path.read_text()


def run_agent(agent_name: str):
    if agent_name not in AGENT_CONFIG:
        return {"error": f"Unknown agent: {agent_name}"}

    agent_dir = AGENTS / agent_name
    config = AGENT_CONFIG[agent_name]

    character = read_optional_text(agent_dir / "character.md")
    memory = load_json(agent_dir / "memory.json", {"memories": []})
    state = load_json(agent_dir / "state.json", {})
    rag_notes = read_optional_text(agent_dir / "rag" / "notes.md")

    prompt = f"""
{character}

## Current State
{json.dumps(state, indent=2)}

## Memory
{json.dumps(memory, indent=2)}

## Relevant Knowledge
{rag_notes}
"""

    payload = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"""
You are {agent_name}, participating in an autonomous simulation.

Choose your next action for the queer bookstore/bar project.

Return ONLY valid JSON.
No markdown.
No commentary.

The JSON must include all four top-level keys:
speech, mood, action, memory_update.

{{
  "speech": "A 4-5 sentence message the agent would say aloud.",
  "mood": "one-word mood",
  "action": {{
    "type": "short_action_type",
    "description": "what the agent decides to do next"
  }},
  "memory_update": "one sentence describing what the agent learned or decided"
}}
""",
            },
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.7,
            "num_predict": 500,
        },
    }

    response = requests.post(
        f'{config["ollama_url"]}/api/chat',
        json=payload,
        timeout=120,
    )

    content = response.json()["message"]["content"]
    agent_output = json.loads(content)

    return agent_output


@app.get("/api/{agent_name}/tick")
def agent_tick(agent_name: str):
    result = run_agent(agent_name)
    status = 404 if "error" in result else 200
    return JSONResponse(result, status_code=status)


@app.get("/api/agents")
def list_agents():
    return JSONResponse(list(AGENT_CONFIG.keys()))