# Club 96

## Start Here: Runtime Profile

Club 96 uses one codebase on every machine. Choose a runtime profile when you start the coordinator server:

### Stable / local profile

Use this on smaller machines such as the GPD Pocket 4, or whenever Alice's homepage/custom prompt tester gets flaky.

```bash
python -m uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

This is the default. Homepage custom prompts ask the model for plain speech text and the server wraps it for the UI, while the simulation still uses the full agent turn format.

### Rich profile

Use this on machines with more headroom, such as the M5 Max, when you want fuller custom prompt answers from the homepage tester.

```bash
CLUB96_RUNTIME_PROFILE=rich python -m uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

On Windows PowerShell:

```powershell
$env:CLUB96_RUNTIME_PROFILE="rich"
python -m uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

Open the app at:

```text
http://127.0.0.1:8000
```

For another machine on your Tailscale network, use the coordinator machine's Tailscale IP:

```text
http://TAILSCALE-IP:8000
```

Club 96 is an experimental multi-agent simulation exploring how AI characters with distinct identities, memories, and goals interact within a shared social space.

The project combines:

* FastAPI backend
* Multiple AI agents with individual character definitions
* Retrieval-Augmented Generation (RAG) knowledge files
* Simple browser-based frontend
* Persistent world-building and agent-to-agent interaction

## Agents

### Alice

Creative organizer focused on building and maintaining the community space.

### Bob

Practical collaborator concerned with operations, logistics, and sustainability.

### Mallory

A trans-futurist journalist from the 2070s who introduces outside perspectives, future knowledge, and critical questions into the simulation.

Each agent has:

* Character definition (`character.md`)
* Knowledge base (`rag/notes.md`)
* Independent reasoning and memory

## Project Structure

```text
club_96/
├── agents/
│   ├── alice/
│   ├── bob/
│   └── mallory/
├── server/
│   └── main.py
├── web/
│   └── index.html
└── .gitignore
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional tools for chunking PDFs, images, HTML, and scanned source files:

```bash
pip install -r requirements-dev.txt
```

Some source-processing workflows also need system tools:

```bash
brew install poppler tesseract ffmpeg
```

## Run

Start the FastAPI server:

```bash
python -m uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

The server will be available at:

```text
http://127.0.0.1:8000
```

## Goals

This project explores:

* Multi-agent interaction
* Persistent simulation worlds
* AI roleplay and social dynamics
* Emergent narrative behavior
* Experimental approaches to digital community building

The project is intended as both a technical experiment and an artistic exploration of synthetic social spaces.
