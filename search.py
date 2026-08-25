import json
import os

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


# -----------------------------
# Configuration
# -----------------------------

MODEL_NAME = "BAAI/bge-small-en-v1.5"

INDEX_PATH = "data/vectorstore/index.faiss"
METADATA_PATH = "data/vectorstore/metadata.json"

TOP_K = 5


# -----------------------------
# Load vector database
# -----------------------------

print("Loading vector database...")

index = faiss.read_index(INDEX_PATH)

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8"
) as file:
    metadata = json.load(file)


print(f"Vectors available: {index.ntotal}")


# -----------------------------
# Load embedding model
# -----------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Embedding device: {device}")

model = SentenceTransformer(
    MODEL_NAME,
    device=device
)


# -----------------------------
# Search function
# -----------------------------

def search(query, top_k=TOP_K):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, index_id in zip(
        scores[0],
        indices[0]
    ):

        if index_id == -1:
            continue

        result = metadata[index_id].copy()

        result["score"] = float(score)

        results.append(result)

    return results


# -----------------------------
# Interactive search
# -----------------------------

while True:

    query = input(
        "\nAsk something about the document "
        "(type 'exit' to quit): "
    )

    if query.lower() == "exit":
        break

    results = search(query)

    print("\n" + "=" * 70)
    print("SEARCH RESULTS")
    print("=" * 70)

    for i, result in enumerate(
        results,
        start=1
    ):

        print(f"\nResult {i}")
        print(f"Score : {result['score']:.4f}")
        print(f"Page  : {result['page']}")
        print(f"Chunk : {result['chunk_id']}")
        print(f"Source: {result['source']}")

        print("\nText:")
        print(result["text"][:700])