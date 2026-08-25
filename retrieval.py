import json
import os
import sys

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

VECTORSTORE_DIR = "data/vectorstore"

INDEX_PATH = os.path.join(
    VECTORSTORE_DIR,
    "index.faiss"
)

METADATA_PATH = os.path.join(
    VECTORSTORE_DIR,
    "metadata.json"
)

EMBEDDING_MAP_PATH = os.path.join(
    VECTORSTORE_DIR,
    "embedding_map.json"
)

# IMPORTANT:
# This MUST be exactly the same model used to create
# embeddings.npy.
MODEL_NAME = "BAAI/bge-small-en-v1.5"

TOP_K = 5

# Because we are using normalized vectors + IndexFlatIP,
# the FAISS score is cosine similarity.
MIN_SCORE = 0.15


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("SEMANTIC RETRIEVAL ENGINE")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    INDEX_PATH,
    METADATA_PATH,
    EMBEDDING_MAP_PATH
]

for path in required_files:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}\n"
        )


# ============================================================
# LOAD FAISS INDEX
# ============================================================

print("\nLoading FAISS index...")

index = faiss.read_index(
    INDEX_PATH
)

print(
    f"FAISS vectors : {index.ntotal}"
)

print(
    f"Dimensions    : {index.d}"
)


# ============================================================
# LOAD CHUNKS / METADATA
# ============================================================

print("\nLoading semantic chunk metadata...")

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8"
) as f:

    chunks = json.load(f)


if not isinstance(
    chunks,
    list
):

    raise ValueError(
        "metadata.json must contain a list."
    )


print(
    f"Semantic chunks: {len(chunks)}"
)


# ============================================================
# LOAD EMBEDDING MAP
# ============================================================

print("\nLoading embedding map...")

with open(
    EMBEDDING_MAP_PATH,
    "r",
    encoding="utf-8"
) as f:

    embedding_map = json.load(f)


if not isinstance(
    embedding_map,
    list
):

    raise ValueError(
        "embedding_map.json must contain a list."
    )


print(
    f"Embedding mappings: "
    f"{len(embedding_map)}"
)


# ============================================================
# BUILD EMBEDDING INDEX → CHUNK LOOKUP
# ============================================================

print("\nBuilding vector lookup...")

vector_to_chunk = {}

for item in embedding_map:

    if not isinstance(
        item,
        dict
    ):
        continue

    embedding_index = item.get(
        "embedding_index"
    )

    chunk_id = item.get(
        "chunk_id"
    )

    if embedding_index is None:
        continue

    if chunk_id is None:
        continue

    vector_to_chunk[
        int(embedding_index)
    ] = chunk_id


print(
    f"Vector lookup entries: "
    f"{len(vector_to_chunk)}"
)


# ============================================================
# BUILD CHUNK ID → CHUNK LOOKUP
# ============================================================

print("\nBuilding chunk lookup...")

chunk_by_id = {}

for chunk in chunks:

    if not isinstance(
        chunk,
        dict
    ):
        continue

    chunk_id = chunk.get(
        "chunk_id"
    )

    if chunk_id is None:
        continue

    chunk_by_id[
        chunk_id
    ] = chunk


print(
    f"Chunk lookup entries: "
    f"{len(chunk_by_id)}"
)


# ============================================================
# VALIDATION
# ============================================================

if index.ntotal != len(
    embedding_map
):

    print("\nWARNING")
    print("-" * 70)

    print(
        f"FAISS vectors       : "
        f"{index.ntotal}"
    )

    print(
        f"Embedding mappings  : "
        f"{len(embedding_map)}"
    )

    print(
        "FAISS and embedding_map "
        "have different sizes."
    )


if index.ntotal != len(
    vector_to_chunk
):

    print("\nWARNING")
    print("-" * 70)

    print(
        f"FAISS vectors       : "
        f"{index.ntotal}"
    )

    print(
        f"Vector lookup       : "
        f"{len(vector_to_chunk)}"
    )


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

print(
    f"Model: {MODEL_NAME}"
)

model = SentenceTransformer(
    MODEL_NAME
)

print(
    "Embedding model loaded."
)


# ============================================================
# VERIFY DIMENSION
# ============================================================

print("\nChecking embedding dimension...")

test_embedding = model.encode(
    ["dimension test"],
    normalize_embeddings=True,
    convert_to_numpy=True
)

test_embedding = np.asarray(
    test_embedding,
    dtype="float32"
)

query_dimension = test_embedding.shape[1]

print(
    f"Query dimension : {query_dimension}"
)

print(
    f"FAISS dimension : {index.d}"
)

if query_dimension != index.d:

    raise RuntimeError(
        "\nEmbedding dimension mismatch!\n"
        f"Query model: {query_dimension}\n"
        f"FAISS: {index.d}\n"
        "\nThe retrieval model must be the same model "
        "used to create embeddings.npy."
    )


# ============================================================
# QUERY EMBEDDING
# ============================================================

def embed_query(
    query
):

    # --------------------------------------------------------
    # BGE models generally benefit from an instruction prefix
    # for retrieval queries.
    # --------------------------------------------------------

    query_text = (
        "Represent this sentence for searching relevant "
        "passages: "
        + query
    )

    embedding = model.encode(
        [query_text],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    embedding = np.asarray(
        embedding,
        dtype="float32"
    )

    return embedding


# ============================================================
# GET TEXT
# ============================================================

def get_text(
    chunk
):

    if not isinstance(
        chunk,
        dict
    ):

        return str(
            chunk
        )

    for key in (
        "text",
        "content",
        "chunk_text",
        "page_content"
    ):

        value = chunk.get(
            key
        )

        if value:

            return str(
                value
            )

    return ""


# ============================================================
# RETRIEVE
# ============================================================

def retrieve(
    query,
    top_k=TOP_K
):

    query = query.strip()

    if not query:

        return []

    # --------------------------------------------------------
    # Create query embedding
    # --------------------------------------------------------

    query_vector = embed_query(
        query
    )

    # --------------------------------------------------------
    # Search FAISS
    # --------------------------------------------------------

    scores, indices = index.search(
        query_vector,
        top_k
    )

    results = []

    for rank, (
        score,
        vector_index
    ) in enumerate(
        zip(
            scores[0],
            indices[0]
        ),
        start=1
    ):

        vector_index = int(
            vector_index
        )

        score = float(
            score
        )

        # ----------------------------------------------------
        # Invalid FAISS result
        # ----------------------------------------------------

        if vector_index < 0:
            continue

        # ----------------------------------------------------
        # Ignore extremely weak matches
        # ----------------------------------------------------

        if score < MIN_SCORE:
            continue

        # ----------------------------------------------------
        # Vector → chunk ID
        # ----------------------------------------------------

        chunk_id = vector_to_chunk.get(
            vector_index
        )

        if chunk_id is None:

            print(
                f"\nWARNING: No chunk mapping "
                f"for vector {vector_index}"
            )

            continue

        # ----------------------------------------------------
        # Chunk ID → actual chunk
        # ----------------------------------------------------

        chunk = chunk_by_id.get(
            chunk_id
        )

        if chunk is None:

            print(
                f"\nWARNING: Chunk ID "
                f"{chunk_id} not found "
                f"in metadata."
            )

            continue

        # ----------------------------------------------------
        # Create result
        # ----------------------------------------------------

        result = {
            "rank": rank,
            "score": score,
            "vector_index": vector_index,
            "chunk_id": chunk_id,
            "source": chunk.get(
                "source"
            ),
            "page": chunk.get(
                "page"
            ),
            "region": chunk.get(
                "region"
            ),
            "content_type": chunk.get(
                "content_type"
            ),
            "text": get_text(
                chunk
            )
        }

        # ----------------------------------------------------
        # Preserve OCR information if available
        # ----------------------------------------------------

        for key in (
            "ocr",
            "ocr_confidence",
            "ocr_quality",
            "ocr_engine"
        ):

            if key in chunk:

                result[key] = chunk[
                    key
                ]

        results.append(
            result
        )

    return results


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    query,
    results
):

    print("\n")
    print("=" * 70)
    print("RETRIEVAL RESULTS")
    print("=" * 70)

    print(
        f"\nQuery:\n{query}"
    )

    if not results:

        print(
            "\nNo relevant chunks found."
        )

        return

    for result in results:

        print("\n")
        print("-" * 70)

        print(
            f"Rank         : "
            f"{result['rank']}"
        )

        print(
            f"Similarity   : "
            f"{result['score']:.4f}"
        )

        print(
            f"Vector index : "
            f"{result['vector_index']}"
        )

        print(
            f"Chunk ID     : "
            f"{result['chunk_id']}"
        )

        print(
            f"Source       : "
            f"{result['source']}"
        )

        print(
            f"Page         : "
            f"{result['page']}"
        )

        print(
            f"Region       : "
            f"{result['region']}"
        )

        print(
            f"Content type : "
            f"{result['content_type']}"
        )

        if "ocr" in result:

            print(
                f"OCR          : "
                f"{result['ocr']}"
            )

        if "ocr_engine" in result:

            print(
                f"OCR engine   : "
                f"{result['ocr_engine']}"
            )

        if "ocr_confidence" in result:

            print(
                f"OCR conf.    : "
                f"{result['ocr_confidence']}"
            )

        if "ocr_quality" in result:

            print(
                f"OCR quality  : "
                f"{result['ocr_quality']}"
            )

        print("\nText:")
        print(result["text"])


# ============================================================
# INTERACTIVE MODE
# ============================================================

def interactive_mode():

    print("\n")
    print("=" * 70)
    print("INTERACTIVE RESEARCH RETRIEVAL")
    print("=" * 70)

    print(
        "\nAsk a question about your documents."
    )

    print(
        "Type 'exit' to quit."
    )

    while True:

        try:

            query = input(
                "\nQuery > "
            )

        except (
            KeyboardInterrupt,
            EOFError
        ):

            print(
                "\nExiting."
            )

            break

        query = query.strip()

        if not query:
            continue

        if query.lower() in (
            "exit",
            "quit",
            "q"
        ):

            print(
                "Exiting."
            )

            break

        results = retrieve(
            query,
            TOP_K
        )

        print_results(
            query,
            results
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Command-line query
    #
    # Example:
    #
    # python retrieval.py "What is the main topic?"
    # --------------------------------------------------------

    if len(sys.argv) > 1:

        query = " ".join(
            sys.argv[1:]
        )

        results = retrieve(
            query,
            TOP_K
        )

        print_results(
            query,
            results
        )

    else:

        interactive_mode()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()