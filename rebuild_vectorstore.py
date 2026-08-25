import json
import os

import faiss
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

CHUNKS_PATH = (
    "data/processed/semantic_chunks.json"
)

EMBEDDINGS_PATH = (
    "data/processed/embeddings.npy"
)

EMBEDDING_MAP_PATH = (
    "data/processed/embedding_map.json"
)

VECTORSTORE_DIR = (
    "data/vectorstore"
)

INDEX_PATH = os.path.join(
    VECTORSTORE_DIR,
    "index.faiss"
)

METADATA_PATH = os.path.join(
    VECTORSTORE_DIR,
    "metadata.json"
)

VECTOR_MAP_PATH = os.path.join(
    VECTORSTORE_DIR,
    "embedding_map.json"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 60)

print(
    "MULTIMODAL FAISS VECTOR STORE BUILDER"
)

print("=" * 60)


# ============================================================
# LOAD SEMANTIC CHUNKS
# ============================================================

print(
    "\nLoading semantic chunks..."
)


with open(
    CHUNKS_PATH,
    "r",
    encoding="utf-8"
) as file:

    chunks = json.load(
        file
    )


if not isinstance(
    chunks,
    list
):

    raise ValueError(
        "semantic_chunks.json must contain a list."
    )


print(
    f"Chunks loaded: {len(chunks)}"
)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print(
    "\nLoading embeddings..."
)


embeddings = np.load(
    EMBEDDINGS_PATH
)


embeddings = np.asarray(
    embeddings,
    dtype="float32"
)


print(
    f"Embeddings shape: "
    f"{embeddings.shape}"
)


# ============================================================
# LOAD EMBEDDING MAP
# ============================================================

print(
    "\nLoading embedding map..."
)


with open(
    EMBEDDING_MAP_PATH,
    "r",
    encoding="utf-8"
) as file:

    embedding_map = json.load(
        file
    )


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
# BASIC VALIDATION
# ============================================================

if embeddings.ndim != 2:

    raise ValueError(
        "Embeddings must be a 2D matrix."
    )


if len(embeddings) != len(
    embedding_map
):

    raise ValueError(
        "\nEmbedding/map mismatch!\n"
        f"Embeddings : {len(embeddings)}\n"
        f"Map        : {len(embedding_map)}\n"
    )


# ============================================================
# BUILD CHUNK LOOKUP
# ============================================================

print(
    "\nBuilding chunk lookup..."
)


chunk_lookup = {}


for chunk_index, chunk in enumerate(
    chunks
):

    chunk_id = chunk.get(
        "chunk_id"
    )

    source = chunk.get(
        "source"
    )

    page = chunk.get(
        "page"
    )


    key = (
        source,
        page,
        chunk_id
    )


    if key in chunk_lookup:

        raise ValueError(
            "\nDuplicate chunk identifier found:\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Chunk: {chunk_id}\n"
        )


    chunk_lookup[key] = (
        chunk_index
    )


# ============================================================
# CREATE VECTOR METADATA
# ============================================================

print(
    "Creating vector metadata..."
)


vector_metadata = []


for embedding_index, mapping in enumerate(
    embedding_map
):

    mapped_embedding_index = mapping.get(
        "embedding_index"
    )


    if mapped_embedding_index != (
        embedding_index
    ):

        raise ValueError(
            "\nEmbedding map is not ordered correctly.\n"
            f"Expected index: {embedding_index}\n"
            f"Found index: {mapped_embedding_index}"
        )


    source = mapping.get(
        "source"
    )

    page = mapping.get(
        "page"
    )

    chunk_id = mapping.get(
        "chunk_id"
    )


    key = (
        source,
        page,
        chunk_id
    )


    if key not in chunk_lookup:

        raise ValueError(
            "\nEmbedding map references "
            "a chunk that does not exist.\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Chunk: {chunk_id}"
        )


    chunk_index = (
        chunk_lookup[key]
    )


    chunk = chunks[
        chunk_index
    ]


    vector_metadata.append({

        "embedding_index":
            embedding_index,

        "chunk_index":
            chunk_index,

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

        "region":
            chunk.get(
                "region"
            ),

        "section":
            chunk.get(
                "section"
            ),

        "subsection":
            chunk.get(
                "subsection"
            ),

        "content_type":
            chunk.get(
                "content_type",
                "text"
            ),

        "ocr_used":
            chunk.get(
                "ocr_used",
                False
            ),

        "text":
            chunk.get(
                "text",
                ""
            )
    })


# ============================================================
# CREATE FAISS INDEX
# ============================================================

print(
    "\nCreating FAISS index..."
)


dimension = (
    embeddings.shape[1]
)


index = faiss.IndexFlatIP(
    dimension
)


# ============================================================
# ADD EMBEDDINGS
# ============================================================

index.add(
    embeddings
)


print(
    f"FAISS vectors: "
    f"{index.ntotal}"
)


print(
    f"Vector dimension: "
    f"{dimension}"
)


# ============================================================
# VALIDATE FAISS COUNT
# ============================================================

if index.ntotal != len(
    embeddings
):

    raise RuntimeError(
        "\nFAISS vector count mismatch!\n"
        f"FAISS: {index.ntotal}\n"
        f"Embeddings: {len(embeddings)}"
    )


# ============================================================
# CREATE VECTORSTORE DIRECTORY
# ============================================================

os.makedirs(
    VECTORSTORE_DIR,
    exist_ok=True
)


# ============================================================
# SAVE FAISS INDEX
# ============================================================

print(
    "\nSaving FAISS index..."
)


faiss.write_index(
    index,
    INDEX_PATH
)


# ============================================================
# SAVE METADATA
# ============================================================

print(
    "Saving vector metadata..."
)


with open(
    METADATA_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        vector_metadata,
        file,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# SAVE EMBEDDING MAP
# ============================================================

print(
    "Saving embedding map..."
)


with open(
    VECTOR_MAP_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        embedding_map,
        file,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# CONTENT TYPE STATISTICS
# ============================================================

content_type_counts = {}


ocr_vectors = 0


for metadata in vector_metadata:

    content_type = metadata.get(
        "content_type",
        "text"
    )


    content_type_counts[
        content_type
    ] = (
        content_type_counts.get(
            content_type,
            0
        )
        + 1
    )


    if metadata.get(
        "ocr_used",
        False
    ):

        ocr_vectors += 1


# ============================================================
# SOURCE STATISTICS
# ============================================================

source_counts = {}


for metadata in vector_metadata:

    source = metadata.get(
        "source",
        "Unknown"
    )


    source_counts[source] = (
        source_counts.get(
            source,
            0
        )
        + 1
    )


# ============================================================
# VALIDATION
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "VECTOR STORE REBUILD COMPLETE"
)

print(
    "=" * 60
)


print(
    f"Semantic chunks : "
    f"{len(chunks)}"
)


print(
    f"Embeddings      : "
    f"{len(embeddings)}"
)


print(
    f"Embedding map   : "
    f"{len(embedding_map)}"
)


print(
    f"FAISS vectors   : "
    f"{index.ntotal}"
)


print(
    f"Dimensions      : "
    f"{dimension}"
)


print(
    f"OCR vectors     : "
    f"{ocr_vectors}"
)


print(
    f"Index           : "
    f"{INDEX_PATH}"
)


print(
    f"Metadata        : "
    f"{METADATA_PATH}"
)


print(
    f"Vector map      : "
    f"{VECTOR_MAP_PATH}"
)


# ============================================================
# CONTENT TYPE SUMMARY
# ============================================================

print(
    "\nCONTENT TYPES IN FAISS:"
)


for content_type, count in (
    content_type_counts.items()
):

    print(
        f"  {content_type}: "
        f"{count}"
    )


# ============================================================
# SOURCE SUMMARY
# ============================================================

print(
    "\nVECTORS BY SOURCE:"
)


for source, count in (
    source_counts.items()
):

    print(
        f"  {source}: "
        f"{count}"
    )


# ============================================================
# SANITY CHECK
# ============================================================

print(
    "\nSANITY CHECK"
)

print(
    "-" * 60
)


first_metadata = (
    vector_metadata[0]
)


print(
    "First embedding index:",
    first_metadata[
        "embedding_index"
    ]
)


print(
    "First chunk index:",
    first_metadata[
        "chunk_index"
    ]
)


print(
    "First chunk ID:",
    first_metadata[
        "chunk_id"
    ]
)


print(
    "First source:",
    first_metadata[
        "source"
    ]
)


print(
    "First page:",
    first_metadata[
        "page"
    ]
)


print(
    "First region:",
    first_metadata[
        "region"
    ]
)


print(
    "First content type:",
    first_metadata[
        "content_type"
    ]
)


print(
    "First vector norm:",
    np.linalg.norm(
        embeddings[0]
    )
)


# ============================================================
# FINAL VALIDATION
# ============================================================

if (
    index.ntotal
    != len(embedding_map)
    or
    index.ntotal
    != len(embeddings)
):

    raise RuntimeError(
        "\nFINAL VALIDATION FAILED.\n"
        "FAISS, embeddings and embedding map "
        "must contain the same number of vectors."
    )


print(
    "\nFAISS is ready."
)