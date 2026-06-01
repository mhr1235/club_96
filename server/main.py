from pathlib import Path
import json
import requests
import random


from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
AGENTS = ROOT / "agents"
WORLD = ROOT / "world"

CONVERSATION_LOG = WORLD / "conversation.json"
TASKS_LOG = WORLD / "tasks.json"
WORLD_STATE_LOG = WORLD / "worldState.json"

TURN_ORDER = ["alice", "bob", "mallory"]

WEAK_MEMORY_PHRASES = [
    "noted potential benefits",
    "considered potential benefits",
    "may be worth exploring",
    "exploring in more depth",
    "open to the idea",
]

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


def normalize_text(value):
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, dict) or isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)

    return str(value).strip()


def load_conversation():
    return load_json(CONVERSATION_LOG, [])


def save_conversation(conversation):
    save_json(CONVERSATION_LOG, conversation)


def get_next_agent(conversation):
    return random.choice(TURN_ORDER)


def format_recent_conversation(conversation, limit=8):
    recent = conversation[-limit:]

    if not recent:
        return "No one has spoken yet. You are helping begin the simulation."

    lines = []

    for entry in recent:
        speaker = entry.get("speaker", "unknown").title()
        speech = entry.get("speech", "")
        mood = entry.get("mood", "")
        action = entry.get("action", {})
        task_update = entry.get("task_update", {})

        lines.append(
            f"{speaker} said: {speech}\n"
            f"Mood: {mood}\n"
            f"Action: {json.dumps(action)}\n"
            f"Task Update: {json.dumps(task_update)}"
        )

    return "\n\n".join(lines)


def is_useful_memory(memory_update):
    memory_text = normalize_text(memory_update)

    if not memory_text:
        return False

    lowered = memory_text.lower()

    return not any(phrase in lowered for phrase in WEAK_MEMORY_PHRASES)


def update_agent_memory(agent_name: str, memory_update):
    memory_text = normalize_text(memory_update)

    if not is_useful_memory(memory_text):
        return

    memory_path = AGENTS / agent_name / "memory.json"
    memory = load_json(memory_path, {"memories": []})

    memories = memory.get("memories", [])

    if memory_text not in memories:
        memories.append(memory_text)

    memory["memories"] = memories[-25:]

    save_json(memory_path, memory)


def update_tasks(task_update):
    if not isinstance(task_update, dict):
        return

    action = task_update.get("action", "none")

    if action == "none":
        return

    title = normalize_text(task_update.get("title", ""))

    if not title:
        return

    tasks_data = load_json(TASKS_LOG, {"tasks": []})
    tasks = tasks_data.get("tasks", [])

    existing = None

    for task in tasks:
        if task.get("title", "").lower() == title.lower():
            existing = task
            break

    if action == "create":
        if existing is None:
            tasks.append(
                {
                    "title": title,
                    "owner": normalize_text(task_update.get("owner", "")),
                    "status": normalize_text(task_update.get("status", "open")),
                }
            )

    elif action == "update":
        if existing:
            existing["owner"] = normalize_text(
                task_update.get("owner", existing.get("owner", ""))
            )
            existing["status"] = normalize_text(
                task_update.get("status", existing.get("status", "open"))
            )

    elif action == "complete":
        if existing:
            existing["status"] = "completed"

    tasks_data["tasks"] = tasks
    save_json(TASKS_LOG, tasks_data)


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
    tasks = load_json(TASKS_LOG, {"tasks": []})
    world_state = load_json(WORLD_STATE_LOG, {})

    recent_conversation = format_recent_conversation(conversation)

    prompt = f"""
{character}

## Personal State
{json.dumps(state, indent=2)}

## Shared World State
{json.dumps(world_state, indent=2)}

## Memory
{json.dumps(memory, indent=2)}

## Shared Tasks
{json.dumps(tasks, indent=2)}

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

You are living inside the world described by the Shared World State.

React to the location, atmosphere, visitors, music, events, and problems happening around you.

Not every response needs to create a task.

You may:
- react to visitors
- discuss art
- discuss nightlife
- discuss community dynamics
- tell stories
- observe strange happenings
- remember past experiences
- disagree with other agents
- suggest events, performances, exhibitions, or social activities

The simulation is not only about planning.

React to what the others have said.

You should respect other agents, but you should not automatically agree.
If another agent's proposal conflicts with your core goals, challenge it.
Offer alternatives or tradeoffs.

Your reply should be 1-2 sentences of dialogue unless you are Mallory, then you are more verbose.
Your speech should be at least {config["min_response_length"]} words.

Use the Shared Tasks list to continue existing work.
If there is an open or in-progress task relevant to the conversation, continue it instead of inventing a brand new topic.
If a topic has circled for too long, turn it into a concrete task, mark it completed, or move the group forward.

Avoid circling. If the group has already discussed an idea, either:
1. make it more concrete,
2. assign a next step,
3. raise a new objection,
4. mark it as decided,
5. try to convince one of the other agents of your perspective,
6. or move to a new topic.

Do not merely say that something is worth exploring again.

Return ONLY valid JSON.
No markdown.
No commentary.

The JSON must include all five top-level keys:
speech, mood, action, memory_update, task_update.

{{
  "speech": "What the agent says aloud in response to the group conversation.",
  "mood": "one-word mood",
  "action": {{
    "type": "short_action_type",
    "description": "what the agent decides to do next"
  }},
  "memory_update": "one concrete memory as a string. Prefer decisions, commitments, objections, or completed work. Do not restate vague possibilities already discussed.",
  "task_update": {{
    "action": "create|update|complete|none",
    "title": "short task title",
    "owner": "alice|bob|mallory",
    "status": "open|in_progress|completed"
  }}
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
            "speech": normalize_text(result.get("speech", "")),
            "mood": normalize_text(result.get("mood", "")),
            "action": result.get("action", {}),
            "memory_update": normalize_text(result.get("memory_update", "")),
            "task_update": result.get("task_update", {}),
        }

        conversation.append(entry)
        save_conversation(conversation)

        update_agent_memory(agent_name, result.get("memory_update", ""))
        update_tasks(result.get("task_update", {}))

    return JSONResponse(
        {
            "agent": agent_name,
            "result": result,
            "conversation": conversation,
            "tasks": load_json(TASKS_LOG, {"tasks": []}),
        }
    )


@app.get("/api/sim/conversation")
def get_conversation():
    return JSONResponse(load_conversation())


@app.get("/api/sim/tasks")
def get_tasks():
    return JSONResponse(load_json(TASKS_LOG, {"tasks": []}))


@app.post("/api/sim/reset")
def reset_conversation():
    save_conversation([])
    return JSONResponse({"status": "reset", "conversation": []})


@app.post("/api/sim/reset-tasks")
def reset_tasks():
    save_json(TASKS_LOG, {"tasks": []})
    return JSONResponse({"status": "reset", "tasks": {"tasks": []}})


@app.get("/api/agents")
def list_agents():
    return JSONResponse(list(AGENT_CONFIG.keys()))