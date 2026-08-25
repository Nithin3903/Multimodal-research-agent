import json
import os

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

CHUNKS_PATH = "data/processed/semantic_chunks.json"

EMBEDDINGS_PATH = "data/processed/embeddings.npy"

MODEL_NAME = "BAAI/bge-small-en-v1.5"

BATCH_SIZE = 16


# ============================================================
# EMBEDDING SETTINGS
# ============================================================

# Text chunks and table chunks are embedded.
#
# Image chunks are NOT embedded as text because their text is
# only metadata such as:
#
# "Embedded image 1 (jpeg), dimensions 2514x3832"
#
# That would pollute semantic retrieval.
#
# The image/page information remains inside semantic_chunks.json
# and can later be passed to a vision model.


EMBEDDABLE_CONTENT_TYPES = {
    "text",
    "table"
}


# ============================================================
# START
# ============================================================

print("=" * 60)

print(
    "MULTIMODAL SEMANTIC EMBEDDING GENERATOR"
)

print("=" * 60)


# ============================================================
# CHECK INPUT
# ============================================================

if not os.path.exists(
    CHUNKS_PATH
):

    raise FileNotFoundError(
        "\nSemantic chunks file was not found:\n"
        f"{CHUNKS_PATH}\n\n"
        "Run semantic_chunker.py first."
    )


# ============================================================
# LOAD CHUNKS
# ============================================================

print(
    "\nLoading semantic chunks..."
)


with open(
    CHUNKS_PATH,
    "r",
    encoding="utf-8"
) as f:

    chunks = json.load(
        f
    )


if not isinstance(
    chunks,
    list
):

    raise ValueError(
        "semantic_chunks.json must contain a list of chunks."
    )


print(
    f"Total chunks loaded: {len(chunks)}"
)


# ============================================================
# FILTER EMBEDDABLE CHUNKS
# ============================================================

embeddable_chunks = []

skipped_chunks = []


for chunk in chunks:

    content_type = chunk.get(
        "content_type",
        "text"
    )


    text = chunk.get(
        "text",
        ""
    )


    if not isinstance(
        text,
        str
    ):

        text = str(
            text
        )


    text = text.strip()


    # --------------------------------------------------------
    # Skip image-only metadata
    # --------------------------------------------------------

    if content_type not in (
        EMBEDDABLE_CONTENT_TYPES
    ):

        skipped_chunks.append(
            chunk
        )

        continue


    # --------------------------------------------------------
    # Skip empty text
    # --------------------------------------------------------

    if not text:

        skipped_chunks.append(
            chunk
        )

        continue


    embeddable_chunks.append(
        chunk
    )


# ============================================================
# EMBEDDING STATISTICS
# ============================================================

text_chunk_count = sum(

    1

    for chunk in embeddable_chunks

    if chunk.get(
        "content_type",
        "text"
    ) == "text"
)


table_chunk_count = sum(

    1

    for chunk in embeddable_chunks

    if chunk.get(
        "content_type"
    ) == "table"
)


image_chunk_count = sum(

    1

    for chunk in skipped_chunks

    if chunk.get(
        "content_type"
    ) == "image"
)


visual_page_count = sum(

    1

    for chunk in skipped_chunks

    if chunk.get(
        "content_type"
    ) == "visual_page"
)


print()

print(
    "EMBEDDING FILTER"
)

print(
    "-" * 60
)

print(
    f"Text chunks embedded   : "
    f"{text_chunk_count}"
)

print(
    f"Table chunks embedded   : "
    f"{table_chunk_count}"
)

print(
    f"Image chunks skipped    : "
    f"{image_chunk_count}"
)

print(
    f"Visual pages skipped    : "
    f"{visual_page_count}"
)

print(
    f"Total skipped           : "
    f"{len(skipped_chunks)}"
)

print(
    f"Total embedded          : "
    f"{len(embeddable_chunks)}"
)


if not embeddable_chunks:

    raise ValueError(
        "No text or table chunks are available for embedding."
    )


# ============================================================
# PREPARE TEXT
# ============================================================

texts = [

    chunk.get(
        "text",
        ""
    ).strip()

    for chunk in embeddable_chunks
]


# ============================================================
# DEVICE
# ============================================================

device = (

    "cuda"

    if torch.cuda.is_available()

    else "cpu"
)


print()

print(
    f"Device: {device}"
)


if device == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA version:",
        torch.version.cuda
    )


# ============================================================
# LOAD MODEL
# ============================================================

print()

print(
    "Loading embedding model..."
)


model = SentenceTransformer(
    MODEL_NAME,
    device=device
)


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

print()

print(
    "Generating embeddings..."
)


embeddings = model.encode(

    texts,

    batch_size=BATCH_SIZE,

    show_progress_bar=True,

    normalize_embeddings=True,

    convert_to_numpy=True
)


# ============================================================
# VALIDATE EMBEDDINGS
# ============================================================

if len(embeddings) != len(
    embeddable_chunks
):

    raise RuntimeError(
        "\nEmbedding count does not match "
        "the number of embedded chunks.\n"
        f"Chunks: {len(embeddable_chunks)}\n"
        f"Embeddings: {len(embeddings)}"
    )


if embeddings.ndim != 2:

    raise RuntimeError(
        "Unexpected embedding shape: "
        f"{embeddings.shape}"
    )


# ============================================================
# SAVE EMBEDDINGS
# ============================================================

os.makedirs(

    os.path.dirname(
        EMBEDDINGS_PATH
    ),

    exist_ok=True
)


np.save(
    EMBEDDINGS_PATH,
    embeddings
)


# ============================================================
# SAVE EMBEDDING INDEX MAP
# ============================================================
#
# IMPORTANT:
#
# Because image chunks are now skipped, embeddings.npy no
# longer has a simple:
#
# embedding index == semantic_chunks.json index
#
# relationship.
#
# This mapping tells hybrid_search exactly which chunk each
# embedding belongs to.
#
# ============================================================

EMBEDDING_MAP_PATH = (
    "data/processed/embedding_map.json"
)


embedding_map = []


for embedding_index, chunk in enumerate(
    embeddable_chunks
):

    embedding_map.append({

        "embedding_index":
            embedding_index,

        "chunk_id":
            chunk.get(
                "chunk_id"
            ),

        "source":
            chunk.get(
                "source"
            ),

        "page":
            chunk.get(
                "page"
            ),

        "content_type":
            chunk.get(
                "content_type",
                "text"
            )
    })


with open(
    EMBEDDING_MAP_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        embedding_map,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# VERIFY
# ============================================================

print()

print(
    "=" * 60
)

print(
    "EMBEDDING COMPLETE"
)

print(
    "=" * 60
)


print(
    "Embeddings generated:",
    len(embeddings)
)


print(
    "Embedding dimensions:",
    embeddings.shape[1]
)


print(
    "Embedding dtype:",
    embeddings.dtype
)


print(
    "Saved embeddings to:",
    EMBEDDINGS_PATH
)


print(
    "Saved embedding map to:",
    EMBEDDING_MAP_PATH
)


# ============================================================
# SANITY CHECK
# ============================================================

print()

print(
    "SANITY CHECK"
)

print(
    "-" * 60
)


first_chunk = embeddable_chunks[0]


print(
    "First embedding index:",
    0
)


print(
    "First chunk ID:",
    first_chunk.get(
        "chunk_id"
    )
)


print(
    "First source:",
    first_chunk.get(
        "source"
    )
)


print(
    "First page:",
    first_chunk.get(
        "page"
    )
)


print(
    "First region:",
    first_chunk.get(
        "region"
    )
)


print(
    "First content type:",
    first_chunk.get(
        "content_type"
    )
)


print(
    "First embedding norm:",
    np.linalg.norm(
        embeddings[0]
    )
)


# ============================================================
# NORMALIZED EMBEDDING CHECK
# ============================================================

norms = np.linalg.norm(
    embeddings,
    axis=1
)


print(
    "Average embedding norm:",
    float(
        np.mean(norms)
    )
)


print(
    "Minimum embedding norm:",
    float(
        np.min(norms)
    )
)


print(
    "Maximum embedding norm:",
    float(
        np.max(norms)
    )
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print()

print(
    "Ready for FAISS."
)