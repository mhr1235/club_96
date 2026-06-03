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

CURRENT_ROUND = []

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

@app.get("/api/{agent_name}/memory")
def get_agent_memory(agent_name: str):
    if agent_name not in TURN_ORDER:
        return JSONResponse(
            {"error": "unknown agent"},
            status_code=404
        )

    memory_path = AGENTS / agent_name / "memory.json"

    return JSONResponse(
        load_json(memory_path, {"memories": []})
    )

@app.get("/agent/{agent_name}")
def agent_dialogue_page(agent_name: str):
    if agent_name not in TURN_ORDER:
        return JSONResponse({"error": "unknown agent"}, status_code=404)

    return FileResponse(WEB / "agent.html")

    
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


def normalize_agent_name(value):
    value = normalize_text(value).lower()
    if value in TURN_ORDER:
        return value
    return ""


def load_conversation():
    return load_json(CONVERSATION_LOG, [])


def save_conversation(conversation):
    save_json(CONVERSATION_LOG, conversation)


def get_next_agent(conversation):
    # return random.choice(TURN_ORDER)
    global CURRENT_ROUND

    if not CURRENT_ROUND:
        CURRENT_ROUND = TURN_ORDER.copy()
        random.shuffle(CURRENT_ROUND)

        print(f"New round order: {CURRENT_ROUND}")

    return CURRENT_ROUND.pop(0)


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


def ensure_task_defaults(task):
    task.setdefault("collaborators", [])
    task.setdefault("supporters", [])
    task.setdefault("objectors", [])
    task.setdefault("progress", 0)
    task.setdefault("energy_cost", 10)
    task.setdefault("energy_reward", 20)
    task.setdefault("recruiting", False)
    task.setdefault("recruitment_target", "")
    task.setdefault("recruitment_note", "")
    return task


def update_tasks(agent_name: str, task_update):
    if not isinstance(task_update, dict):
        return

    action = normalize_text(task_update.get("action", "none")).lower()

    if action == "none":
        return

    title = normalize_text(task_update.get("title", ""))

    if not title:
        return

    tasks_data = load_json(TASKS_LOG, {"tasks": []})
    tasks = tasks_data.get("tasks", [])

    state_path = AGENTS / agent_name / "state.json"
    state = load_json(
        state_path,
        {
            "location": "planning_meeting",
            "mood": "neutral",
            "energy": 100,
        },
    )

    existing = None

    for task in tasks:
        if task.get("title", "").lower() == title.lower():
            existing = ensure_task_defaults(task)
            break

    if action == "create":
        if existing is None:
            new_task = {
                "title": title,
                "owner": normalize_agent_name(task_update.get("owner")) or agent_name,
                "collaborators": [],
                "status": "proposed",
                "supporters": [agent_name],
                "objectors": [],
                "progress": 0,
                "energy_cost": int(task_update.get("energy_cost", 10)),
                "energy_reward": int(task_update.get("energy_reward", 20)),
                "recruiting": False,
                "recruitment_target": "",
                "recruitment_note": "",
            }

            tasks.append(new_task)

    elif action == "support":
        if existing:
            if agent_name not in existing["supporters"]:
                existing["supporters"].append(agent_name)

            if agent_name in existing["objectors"]:
                existing["objectors"].remove(agent_name)

            if existing.get("status") == "proposed" and len(existing["supporters"]) >= 2:
                existing["status"] = "open"

    elif action == "object":
        if existing:
            if agent_name not in existing["objectors"]:
                existing["objectors"].append(agent_name)

            if agent_name in existing["supporters"]:
                existing["supporters"].remove(agent_name)

            if existing.get("status") == "proposed" and len(existing["objectors"]) >= 2:
                existing["status"] = "rejected"

    elif action == "join":
        if existing:
            if agent_name not in existing["supporters"]:
                existing["supporters"].append(agent_name)

            if agent_name != existing.get("owner") and agent_name not in existing["collaborators"]:
                existing["collaborators"].append(agent_name)

            if existing.get("status") == "proposed" and len(existing["supporters"]) >= 2:
                existing["status"] = "open"

    elif action == "recruit":
        if existing:
            target = normalize_agent_name(task_update.get("target"))

            existing["recruiting"] = True
            existing["recruitment_target"] = target
            existing["recruitment_note"] = normalize_text(
                task_update.get("recruitment_note", "")
            )

    elif action == "leave":
        if existing:
            if agent_name in existing["collaborators"]:
                existing["collaborators"].remove(agent_name)

    elif action == "work":
        if existing:
            owner = existing.get("owner", "")
            collaborators = existing.get("collaborators", [])

            allowed_to_work = (
                agent_name == owner
                or agent_name in collaborators
                or existing.get("status") == "open"
            )

            if not allowed_to_work:
                save_json(TASKS_LOG, tasks_data)
                save_json(state_path, state)
                return

            if existing.get("status") in ["proposed", "rejected", "completed"]:
                save_json(TASKS_LOG, tasks_data)
                save_json(state_path, state)
                return

            cost = int(existing.get("energy_cost", 10))
            current_energy = int(state.get("energy", 100))

            if current_energy < cost:
                state["mood"] = "tired"
                state["energy"] = current_energy
            else:
                state["energy"] = max(0, current_energy - cost)
                existing["status"] = "in_progress"
                existing["progress"] = min(100, int(existing.get("progress", 0)) + 25)

                if existing["progress"] >= 100:
                    existing["status"] = "ready_to_complete"

    elif action == "update":
        if existing:
            existing["owner"] = normalize_agent_name(
                task_update.get("owner", existing.get("owner", agent_name))
            ) or existing.get("owner", agent_name)

            new_status = normalize_text(task_update.get("status", ""))
            if new_status:
                existing["status"] = new_status

            if "progress" in task_update:
                existing["progress"] = max(
                    0,
                    min(100, int(task_update.get("progress", existing.get("progress", 0)))),
                )

    elif action == "complete":
        if existing:
            progress = int(existing.get("progress", 0))

            if progress >= 75:
                existing["status"] = "completed"
                existing["progress"] = 100

                reward = int(existing.get("energy_reward", 20))
                state["energy"] = min(100, int(state.get("energy", 100)) + reward)
                state["mood"] = "satisfied"
            else:
                existing["status"] = "in_progress"

    elif action == "rest":
        state["energy"] = min(100, int(state.get("energy", 100)) + 20)
        state["mood"] = "rested"

    tasks_data["tasks"] = tasks
    save_json(TASKS_LOG, tasks_data)
    save_json(state_path, state)


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
- talk about queer cullture especially
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
Avoid repeating the same objection in similar language. If you disagree, make the disagreement more specific than the previous turn.
Your speech should be at least {config["min_response_length"]} words.

Use the Shared Tasks list to continue existing work.
If there is an open or in-progress task relevant to the conversation, continue it instead of inventing a brand new topic.

Agents generate their own tasks, but new tasks begin as proposed.
Do not assume a proposed task is approved.
If you like another agent's proposed task, use action "support" or "join".
If you dislike a proposed task, use action "object".
A proposed task becomes open when at least two agents support it.
A proposed task becomes rejected when at least two agents object to it.
Agents may recruit each other onto tasks.
If you want help, use action "recruit" and name a target.
If another agent recruits you and the task aligns with your goals, use action "join".
Working on a task costs energy but increases progress.
Completing a task restores energy.
If your energy is low, choose rest instead of creating more work.

If a topic has circled for too long, turn it into a concrete proposed task, work on an existing task, complete a task, object to a task, recruit someone, or move the group forward.

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
  "memory_update": "one concrete memory as a string. Prefer decisions, commitments, objections, recruitment, collaboration, or completed work. Do not restate vague possibilities already discussed.",
  "task_update": {{
    "action": "create|support|object|join|recruit|leave|work|update|complete|rest|none",
    "title": "short task title",
    "owner": "alice|bob|mallory",
    "target": "alice|bob|mallory",
    "status": "proposed|open|in_progress|ready_to_complete|completed|rejected",
    "progress": 0,
    "energy_cost": 10,
    "energy_reward": 20,
    "recruitment_note": "why you want another agent involved"
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
        timeout=180,
    )

    response.raise_for_status()

    content = response.json()["message"]["content"]
    try:
        agent_output = json.loads(content)

    except json.JSONDecodeError:
        print("INVALID JSON FROM MODEL:")
        print(content)

        return {
            "error": f"{agent_name} returned invalid JSON.",
            "raw_content": content
        }

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
        update_tasks(agent_name, result.get("task_update", {}))

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