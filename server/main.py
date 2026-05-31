from pathlib import Path
import json
import requests

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
AGENTS = ROOT / "agents"
WORLD = ROOT / "world"
CONVERSATION_LOG = WORLD / "conversation.json"

TURN_ORDER = ["alice", "bob", "mallory"]

app = FastAPI()
app.mount("/static", StaticFiles(directory=WEB), name="static")

AGENT_CONFIG = {
    "alice": {
        "ollama_url": "http://localhost:11434",
        "model": "gemma3:4b",
        "temperature": 0.8,
        "num_predict": 500,
        "min_response_length": 15,
    },
    "bob": {
        "ollama_url": "http://localhost:11434",
        "model": "llama3.2:latest",
        "temperature": 0.55,
        "num_predict": 500,
        "min_response_length": 10,
    },
    "mallory": {
        "ollama_url": "http://localhost:11434",
        "model": "mistral:7b",
        "temperature": 0.95,
        "num_predict": 700,
        "min_response_length": 21,
    },
}


@app.get("/")
def home():
    return FileResponse(WEB / "index.html")


@app.get("/sim")
def simulation_page():
    return FileResponse(WEB / "sim.html")


def load_json(path: Path, fallback):
    if not path.exists():
        return fallback

    text = path.read_text().strip()

    if not text:
        return fallback

    return json.loads(text)


def save_json(path: Path, data):
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def read_optional_text(path: Path):
    if not path.exists():
        return ""

    return path.read_text()


def load_conversation():
    return load_json(CONVERSATION_LOG, [])


def save_conversation(conversation):
    save_json(CONVERSATION_LOG, conversation)


def get_next_agent(conversation):
    return TURN_ORDER[len(conversation) % len(TURN_ORDER)]


def format_recent_conversation(conversation, limit=12):
    recent = conversation[-limit:]

    if not recent:
        return "No one has spoken yet. You are helping begin the simulation."

    lines = []

    for entry in recent:
        speaker = entry.get("speaker", "unknown").title()
        speech = entry.get("speech", "")
        mood = entry.get("mood", "")
        action = entry.get("action", {})

        lines.append(
            f"{speaker} said: {speech}\n"
            f"Mood: {mood}\n"
            f"Action: {json.dumps(action)}"
        )

    return "\n\n".join(lines)


def run_agent(agent_name: str, conversation=None):
    if agent_name not in AGENT_CONFIG:
        return {"error": f"Unknown agent: {agent_name}"}

    conversation = conversation or []

    agent_dir = AGENTS / agent_name
    config = AGENT_CONFIG[agent_name]

    character = read_optional_text(agent_dir / "character.md")
    memory = load_json(agent_dir / "memory.json", {"memories": []})
    state = load_json(agent_dir / "state.json", {})
    rag_notes = read_optional_text(agent_dir / "rag" / "notes.md")

    recent_conversation = format_recent_conversation(conversation)

    prompt = f"""
{character}

## Current State
{json.dumps(state, indent=2)}

## Memory
{json.dumps(memory, indent=2)}

## Relevant Knowledge
{rag_notes}

## Recent Conversation
{recent_conversation}
"""

    user_prompt = f"""
You are {agent_name}, participating in an autonomous multi-agent simulation.

You are not speaking in isolation.

You are responding to the recent conversation between Alice, Bob, and Mallory.

Advance the shared simulation for the queer bookstore/bar project.

React to what the others have said. 

You should respect other agents, but you should not automatically agree.
If another agent's proposal conflicts with your core goals, challenge it.
Offer alternatives or tradeoffs.
 
Your reply should be 1-2 sentences of dialogue unless you are Mallory, then you are more verbose.
Your speech should be at least {config["min_response_length"]} words.

Return ONLY valid JSON.
No markdown.
No commentary.

The JSON must include all four top-level keys:
speech, mood, action, memory_update.

{{
  "speech": "What the agent says aloud in response to the group conversation.",
  "mood": "one-word mood",
  "action": {{
    "type": "short_action_type",
    "description": "what the agent decides to do next"
  }},
  "memory_update": "one sentence describing what the agent learned, noticed, or decided"
}}
"""

    payload = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": config["temperature"],
            "num_predict": config["num_predict"],
        },
    }

    response = requests.post(
        f'{config["ollama_url"]}/api/chat',
        json=payload,
        timeout=120,
    )

    response.raise_for_status()

    content = response.json()["message"]["content"]
    agent_output = json.loads(content)

    return agent_output


@app.get("/api/{agent_name}/tick")
def agent_tick(agent_name: str):
    conversation = load_conversation()
    result = run_agent(agent_name, conversation)
    status = 404 if "error" in result else 200
    return JSONResponse(result, status_code=status)


@app.post("/api/sim/tick")
def simulation_tick():
    conversation = load_conversation()
    agent_name = get_next_agent(conversation)

    result = run_agent(agent_name, conversation)

    if "error" not in result:
        entry = {
            "turn": len(conversation) + 1,
            "speaker": agent_name,
            "speech": result.get("speech", ""),
            "mood": result.get("mood", ""),
            "action": result.get("action", {}),
            "memory_update": result.get("memory_update", ""),
        }

        conversation.append(entry)
        save_conversation(conversation)

    return JSONResponse(
        {
            "agent": agent_name,
            "result": result,
            "conversation": conversation,
        }
    )


@app.get("/api/sim/conversation")
def get_conversation():
    return JSONResponse(load_conversation())


@app.post("/api/sim/reset")
def reset_conversation():
    save_conversation([])
    return JSONResponse({"status": "reset", "conversation": []})


@app.get("/api/agents")
def list_agents():
    return JSONResponse(list(AGENT_CONFIG.keys()))