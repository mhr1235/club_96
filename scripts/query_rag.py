from pathlib import Path
import argparse

import chromadb
import requests


ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = ROOT / "world" / "rag_chroma"

COLLECTION_NAME = "club96_rag"
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"


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


def query_rag(agent_name, query, n_results):
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection(name=COLLECTION_NAME)
    query_embedding = embed_text(query)

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"agent": agent_name},
        include=["documents", "metadatas", "distances"],
    )


def print_results(results):
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        print("No matching chunks found.")
        return

    for index, document in enumerate(documents, start=1):
        metadata = metadatas[index - 1]
        distance = distances[index - 1]
        print(f"\nResult {index}")
        print(f"Chunk: {metadata.get('chunk_name', '')}")
        print(f"Agent: {metadata.get('agent', '')}")
        print(f"Source: {metadata.get('source', '')}")
        print(f"URL: {metadata.get('url', '')}")
        print(f"Tags: {metadata.get('tags', '')}")
        print(f"Distance: {distance:.4f}")
        print("Text:")
        print(document)


def main():
    parser = argparse.ArgumentParser(description="Query the Club 96 Chroma RAG index.")
    parser.add_argument("agent", help="Agent name to filter by, such as bob.")
    parser.add_argument("query", help="Natural language query to retrieve chunks for.")
    parser.add_argument(
        "-n",
        "--n-results",
        type=int,
        default=3,
        help="Number of chunks to return.",
    )
    args = parser.parse_args()

    results = query_rag(args.agent, args.query, args.n_results)
    print_results(results)


if __name__ == "__main__":
    main()
