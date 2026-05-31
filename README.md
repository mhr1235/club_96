# Club 96

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
source .venv/bin/activate.fish
```

Install dependencies:

```bash
pip install fastapi uvicorn requests
```

## Run

Start the FastAPI server:

```bash
python -m uvicorn server.main:app --reload
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
