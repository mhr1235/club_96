from pathlib import Path
import argparse
import re

import chromadb
import requests


ROOT = Path(__file__).resolve().parent.parent
AGENTS = ROOT / "agents"
CHROMA_PATH = ROOT / "world" / "rag_chroma"

COLLECTION_NAME = "club96_rag"
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"

CHUNK_HEADING = re.compile(r"^## Chunk:\s*(.+?)\s*$", re.MULTILINE)


def normalize_text(value):
    return value.strip()


def parse_metadata_line(line, label):
    prefix = f"{label}:"
    if line.startswith(prefix):
        return line[len(prefix):].strip()
    return ""


def parse_chunk_body(raw_body):
    lines = [line.rstrip() for line in raw_body.strip().splitlines()]
    metadata = {
        "source": "",
        "url": "",
        "tags": "",
    }
    content_lines = []

    for line in lines:
        source = parse_metadata_line(line, "Source")
        url = parse_metadata_line(line, "URL")
        tags = parse_metadata_line(line, "Tags")

        if source:
            metadata["source"] = source
            continue

        if url:
            metadata["url"] = url
            continue

        if tags:
            metadata["tags"] = tags
            continue

        content_lines.append(line)

    return metadata, normalize_text("\n".join(content_lines))


def parse_chunk_file(agent_name, path):
    text = path.read_text()
    matches = list(CHUNK_HEADING.finditer(text))
    chunks = []

    for index, match in enumerate(matches):
        chunk_name = match.group(1).strip()
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        raw_body = text[body_start:body_end]
        metadata, content = parse_chunk_body(raw_body)

        if not content:
            continue

        chunk_id = f"{agent_name}_{path.stem}_{chunk_name}".lower().replace(" ", "_")

        chunks.append(
            {
                "id": chunk_id,
                "agent": agent_name,
                "chunk_name": chunk_name,
                "source_file": str(path.relative_to(ROOT)),
                "source": metadata["source"],
                "url": metadata["url"],
                "tags": metadata["tags"],
                "text": content,
            }
        )

    return chunks


def load_chunks():
    chunks = []

    for path in sorted(AGENTS.glob("*/rag/chunks/*.md")):
        agent_name = path.relative_to(AGENTS).parts[0]
        chunks.extend(parse_chunk_file(agent_name, path))

    return chunks


def embed_text(text):
    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={
            "model": EMBED_MODEL,
            "prompt": text,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["embedding"]


def rebuild_collection(chunks):
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for chunk in chunks:
        print(f"Embedding {chunk['id']}")
        ids.append(chunk["id"])
        documents.append(chunk["text"])
        embeddings.append(embed_text(chunk["text"]))
        metadatas.append(
            {
                "agent": chunk["agent"],
                "chunk_name": chunk["chunk_name"],
                "source_file": chunk["source_file"],
                "source": chunk["source"],
                "url": chunk["url"],
                "tags": chunk["tags"],
            }
        )

    if ids:
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    return collection


def main():
    parser = argparse.ArgumentParser(description="Build the Club 96 Chroma RAG index.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse chunks and print what would be indexed without embedding or writing Chroma.",
    )
    args = parser.parse_args()

    chunks = load_chunks()

    if args.dry_run:
        print(f"Found {len(chunks)} chunks.")
        for chunk in chunks:
            print(f"- {chunk['id']} ({chunk['agent']}): {chunk['tags']}")
        return

    collection = rebuild_collection(chunks)
    print(f"Indexed {collection.count()} chunks into {CHROMA_PATH}")


if __name__ == "__main__":
    main()
