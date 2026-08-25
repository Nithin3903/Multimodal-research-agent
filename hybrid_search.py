import json
import re

import faiss
import numpy as np
import torch

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

CHUNKS_PATH = "data/processed/semantic_chunks.json"

INDEX_PATH = "data/vectorstore/index.faiss"

EMBEDDING_MAP_PATH = (
    "data/processed/embedding_map.json"
)

MODEL_NAME = "BAAI/bge-small-en-v1.5"

TOP_K = 5

SEMANTIC_WEIGHT = 0.55
BM25_WEIGHT = 0.30
KEYWORD_WEIGHT = 0.15


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print("QUERY-AWARE MULTIMODAL HYBRID SEARCH")
print("=" * 70)


# ============================================================
# LOAD SEMANTIC CHUNKS
# ============================================================

print()
print("Loading semantic chunks...")

with open(
    CHUNKS_PATH,
    "r",
    encoding="utf-8"
) as file:

    chunks = json.load(file)


print(
    f"Loaded chunks: {len(chunks)}"
)


# ============================================================
# LOAD FAISS INDEX
# ============================================================

print()
print("Loading FAISS index...")

index = faiss.read_index(
    INDEX_PATH
)

print(
    f"FAISS vectors: {index.ntotal}"
)


# ============================================================
# LOAD EMBEDDING MAP
# ============================================================
#
# IMPORTANT
#
# Not every semantic chunk is necessarily embedded.
#
# For example:
#
# semantic chunks = 59
# FAISS vectors   = 55
#
# Therefore:
#
# FAISS index 0 does NOT necessarily mean
# semantic chunk 0.
#
# embedding_map.json gives us:
#
# FAISS embedding index
#          ↓
# chunk_id
#          ↓
# semantic_chunks.json position
#
# ============================================================

print()
print("Loading embedding map...")

with open(
    EMBEDDING_MAP_PATH,
    "r",
    encoding="utf-8"
) as file:

    embedding_map = json.load(file)


# ============================================================
# VALIDATE EMBEDDING MAP
# ============================================================

if len(embedding_map) != index.ntotal:

    raise ValueError(
        "\n"
        "Embedding map / FAISS mismatch!\n"
        f"FAISS vectors : {index.ntotal}\n"
        f"Embedding map : {len(embedding_map)}\n"
    )


# ============================================================
# CREATE CHUNK ID → CHUNK INDEX MAP
# ============================================================

chunk_index_by_id = {}


for i, chunk in enumerate(chunks):

    chunk_id = chunk.get(
        "chunk_id"
    )

    if chunk_id is not None:

        chunk_index_by_id[
            str(chunk_id)
        ] = i


# ============================================================
# CREATE FAISS INDEX → CHUNK INDEX MAP
# ============================================================

faiss_to_chunk_index = {}


for item in embedding_map:

    embedding_index = int(
        item["embedding_index"]
    )

    chunk_id = str(
        item["chunk_id"]
    )

    if chunk_id not in chunk_index_by_id:

        raise ValueError(
            f"\nChunk ID {chunk_id} "
            "from embedding_map.json "
            "does not exist in "
            "semantic_chunks.json."
        )

    chunk_index = (
        chunk_index_by_id[
            chunk_id
        ]
    )

    faiss_to_chunk_index[
        embedding_index
    ] = chunk_index


# ============================================================
# FINAL MAP VALIDATION
# ============================================================

if (
    len(faiss_to_chunk_index)
    != index.ntotal
):

    raise ValueError(
        "\nNot every FAISS vector "
        "has a valid chunk mapping."
    )


print(
    "Embedding map validated."
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print()
print("Loading embedding model...")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    f"Embedding device: {device}"
)


if device == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


model = SentenceTransformer(
    MODEL_NAME,
    device=device
)


print(
    "Embedding model ready."
)


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text):

    text = text.lower()

    return re.findall(
        r"[a-zA-Z0-9]+(?:[.,][a-zA-Z0-9]+)*",
        text
    )


# ============================================================
# BUILD BM25
# ============================================================

print()
print("Building BM25 index...")


tokenized_documents = [

    tokenize(
        chunk.get(
            "text",
            ""
        )
    )

    for chunk in chunks

]


bm25 = BM25Okapi(
    tokenized_documents
)


print(
    "BM25 ready."
)


# ============================================================
# NORMALIZE SCORES
# ============================================================

def normalize_scores(scores):

    scores = np.asarray(
        scores,
        dtype=np.float32
    )

    minimum = scores.min()

    maximum = scores.max()


    if (
        maximum - minimum
        < 1e-8
    ):

        return np.zeros_like(
            scores
        )


    return (
        scores - minimum
    ) / (
        maximum - minimum
    )


# ============================================================
# QUERY INTENT
# ============================================================

def detect_query_intent(query):

    q = query.lower()


    # --------------------------------------------------------
    # AUTHOR / HEADER
    # --------------------------------------------------------

    author_words = [

        "author",
        "authors",
        "who wrote",
        "researcher",
        "researchers",
        "written by"

    ]


    # --------------------------------------------------------
    # METADATA / HEADER
    # --------------------------------------------------------

    metadata_words = [

        "title",
        "paper title",
        "department",
        "affiliation",
        "email"

    ]


    # --------------------------------------------------------
    # ABSTRACT
    # --------------------------------------------------------

    abstract_words = [

        "abstract",
        "summary",
        "main objective",
        "objective",
        "purpose",
        "aim",
        "goal",
        "what is the paper about"

    ]


    # --------------------------------------------------------
    # TECHNICAL
    # --------------------------------------------------------

    technical_words = [

        "architecture",
        "method",
        "methodology",
        "implementation",
        "experiment",
        "experimental",
        "model",
        "algorithm",
        "classification",
        "prediction",
        "result",
        "results",
        "performance",
        "dataset",
        "sensor",
        "sensors",
        "system",
        "how does",
        "which model",
        "why",
        "score",
        "accuracy",
        "cross-validation",
        "cross validation",
        "xgboost",
        "random forest"

    ]


    # --------------------------------------------------------
    # DETECTION
    # --------------------------------------------------------

    if any(
        word in q
        for word in author_words
    ):

        return "header"


    if any(
        word in q
        for word in metadata_words
    ):

        return "header"


    if any(
        word in q
        for word in abstract_words
    ):

        return "abstract"


    if any(
        word in q
        for word in technical_words
    ):

        return "body"


    return "general"


# ============================================================
# OBJECTIVE QUERY DETECTION
# ============================================================

def is_objective_query(query):

    q = query.lower()


    objective_phrases = [

        "main objective",
        "objective",
        "purpose",
        "aim",
        "goal",
        "what does the system",
        "why was the system developed",
        "why was the system proposed",
        "what is the purpose"

    ]


    return any(
        phrase in q
        for phrase in objective_phrases
    )


# ============================================================
# QUERY EXPANSION
# ============================================================

def expand_query(query):

    q = query.lower()

    expanded_query = query


    # --------------------------------------------------------
    # HAISAS
    # --------------------------------------------------------

    if "haisas" in q:

        expanded_query += (
            " Hybrid AI IoT "
            "Smart Aquaponics System"
        )


    # --------------------------------------------------------
    # OBJECTIVE
    # --------------------------------------------------------

    if is_objective_query(
        query
    ):

        expanded_query += (

            " objective "
            "purpose "
            "aim "
            "goal "
            "real-time monitoring "
            "fish health prediction "
            "automated control "
            "integrated framework "
            "aquaponics"

        )


    # --------------------------------------------------------
    # RANDOM FOREST SCORE
    # --------------------------------------------------------

    if (

        "0.9593" in q

        or
        "cross-validation" in q

        or
        "cross validation" in q

    ):

        expanded_query += (

            " Random Forest "
            "RFC "
            "highest cross-validation "
            "score "
            "0.9593"

        )


    # --------------------------------------------------------
    # XGBOOST
    # --------------------------------------------------------

    if "xgboost" in q:

        expanded_query += (

            " XGBoost "
            "test accuracy "
            "macro F1 "
            "98% "
            "0.94"

        )


    return expanded_query


# ============================================================
# REGION ADJUSTMENT
# ============================================================

def get_region_adjustment(
    region,
    intent
):


    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    if intent == "header":

        if region == "header":
            return 0.08

        if region == "abstract":
            return -0.02

        if region == "body":
            return -0.03

        return 0.0


    # --------------------------------------------------------
    # ABSTRACT
    # --------------------------------------------------------

    if intent == "abstract":

        if region == "abstract":
            return 0.04

        if region == "body":
            return 0.01

        if region == "header":
            return -0.08

        return 0.0


    # --------------------------------------------------------
    # BODY
    # --------------------------------------------------------

    if intent == "body":

        if region == "body":
            return 0.03

        if region == "abstract":
            return 0.01

        if region == "header":
            return -0.08

        return 0.0


    # --------------------------------------------------------
    # GENERAL
    # --------------------------------------------------------

    if region == "header":
        return -0.05

    if region == "abstract":
        return 0.01

    if region == "body":
        return 0.02

    return 0.0


# ============================================================
# KEYWORD SCORE
# ============================================================

def calculate_keyword_scores(
    query,
    chunks
):

    query_tokens = set(
        tokenize(query)
    )


    scores = []


    for chunk in chunks:

        text_tokens = set(
            tokenize(
                chunk.get(
                    "text",
                    ""
                )
            )
        )


        if not query_tokens:

            scores.append(
                0.0
            )

            continue


        overlap = (

            len(
                query_tokens
                &
                text_tokens
            )

            /

            len(
                query_tokens
            )

        )


        scores.append(
            overlap
        )


    return np.asarray(
        scores,
        dtype=np.float32
    )


# ============================================================
# OBJECTIVE SCORE
# ============================================================

def calculate_objective_scores(
    query,
    chunks
):

    scores = np.zeros(
        len(chunks),
        dtype=np.float32
    )


    if not is_objective_query(
        query
    ):

        return scores


    objective_terms = [

        "objective",
        "purpose",
        "aim",
        "goal",
        "integrat",
        "framework",
        "monitoring",
        "prediction",
        "automated control",
        "fish health",
        "aquaponics"

    ]


    for i, chunk in enumerate(
        chunks
    ):

        text = chunk.get(
            "text",
            ""
        ).lower()


        score = 0.0


        # ----------------------------------------------------
        # TERM MATCHING
        # ----------------------------------------------------

        for term in objective_terms:

            if term in text:

                score += 0.05


        # ----------------------------------------------------
        # EXPLICIT OBJECTIVE LANGUAGE
        # ----------------------------------------------------

        if (

            "this paper proposes"
            in text

            or

            "this paper presents"
            in text

            or

            "the main objective"
            in text

            or

            "objective of"
            in text

            or

            "aims to"
            in text

            or

            "aim of"
            in text

        ):

            score += 0.25


        # ----------------------------------------------------
        # INTEGRATION BONUS
        # ----------------------------------------------------

        if (

            "integrat" in text

            and

            "monitor" in text

            and

            "predict" in text

        ):

            score += 0.20


        scores[i] = score


    return scores


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(
    query,
    top_k=TOP_K
):


    if not query or not query.strip():

        return []


    # ========================================================
    # QUERY EXPANSION
    # ========================================================

    expanded_query = expand_query(
        query
    )


    # ========================================================
    # QUERY EMBEDDING
    # ========================================================

    query_embedding = model.encode(

        [expanded_query],

        normalize_embeddings=True,

        convert_to_numpy=True

    )


    query_embedding = np.asarray(

        query_embedding,

        dtype=np.float32

    )


    # ========================================================
    # FAISS SEARCH
    # ========================================================

    search_k = min(

        index.ntotal,

        len(chunks)

    )


    (
        semantic_scores_raw,
        semantic_indices
    ) = index.search(

        query_embedding,

        search_k

    )


    semantic_scores_raw = (
        semantic_scores_raw[0]
    )


    semantic_indices = (
        semantic_indices[0]
    )


    # ========================================================
    # MAP FAISS SCORES TO CORRECT CHUNKS
    # ========================================================
    #
    # THIS IS THE IMPORTANT FIX.
    #
    # Do NOT do:
    #
    # semantic_scores[idx] = score
    #
    # because FAISS index and semantic chunk index
    # can be different.
    #
    # ========================================================

    semantic_scores = np.zeros(

        len(chunks),

        dtype=np.float32

    )


    for (

        score,

        faiss_index

    ) in zip(

        semantic_scores_raw,

        semantic_indices

    ):


        if faiss_index < 0:

            continue


        chunk_index = (
            faiss_to_chunk_index.get(
                int(faiss_index)
            )
        )


        if chunk_index is not None:

            semantic_scores[
                chunk_index
            ] = float(score)


    # ========================================================
    # BM25 SEARCH
    # ========================================================

    query_tokens = tokenize(
        expanded_query
    )


    bm25_scores = bm25.get_scores(
        query_tokens
    )


    # ========================================================
    # NORMALIZE SEMANTIC
    # ========================================================

    semantic_normalized = (
        normalize_scores(
            semantic_scores
        )
    )


    # ========================================================
    # NORMALIZE BM25
    # ========================================================

    bm25_normalized = (
        normalize_scores(
            bm25_scores
        )
    )


    # ========================================================
    # KEYWORD SEARCH
    # ========================================================

    keyword_scores = (
        calculate_keyword_scores(

            expanded_query,

            chunks

        )
    )


    keyword_normalized = (
        normalize_scores(
            keyword_scores
        )
    )


    # ========================================================
    # HYBRID SCORE
    # ========================================================

    hybrid_scores = (

        SEMANTIC_WEIGHT
        *
        semantic_normalized

    ) + (

        BM25_WEIGHT
        *
        bm25_normalized

    ) + (

        KEYWORD_WEIGHT
        *
        keyword_normalized

    )


    # ========================================================
    # QUERY INTENT
    # ========================================================

    intent = detect_query_intent(
        query
    )


    # ========================================================
    # REGION ADJUSTMENT
    # ========================================================

    final_scores = (
        hybrid_scores.copy()
    )


    adjustments = np.zeros(

        len(chunks),

        dtype=np.float32

    )


    for i, chunk in enumerate(
        chunks
    ):

        adjustment = (
            get_region_adjustment(

                chunk.get(
                    "region",
                    "body"
                ),

                intent

            )
        )


        adjustments[i] = (
            adjustment
        )


        final_scores[i] += (
            adjustment
        )


    # ========================================================
    # OBJECTIVE BOOST
    # ========================================================

    objective_scores = (
        calculate_objective_scores(

            query,

            chunks

        )
    )


    final_scores += (
        objective_scores
    )


    # ========================================================
    # RANK RESULTS
    # ========================================================

    ranked_indices = np.argsort(

        final_scores

    )[::-1]


    # ========================================================
    # BUILD RESULTS
    # ========================================================

    results = []


    for (

        rank,

        chunk_index

    ) in enumerate(

        ranked_indices[:top_k],

        start=1

    ):


        result = (
            chunks[
                chunk_index
            ].copy()
        )


        result["rank"] = (
            rank
        )


        result["intent"] = (
            intent
        )


        result["semantic_score"] = float(

            semantic_normalized[
                chunk_index
            ]

        )


        result["bm25_score"] = float(

            bm25_normalized[
                chunk_index
            ]

        )


        result["keyword_score"] = float(

            keyword_normalized[
                chunk_index
            ]

        )


        result["hybrid_score"] = float(

            hybrid_scores[
                chunk_index
            ]

        )


        result["region_adjustment"] = float(

            adjustments[
                chunk_index
            ]

        )


        result["objective_score"] = float(

            objective_scores[
                chunk_index
            ]

        )


        result["final_score"] = float(

            final_scores[
                chunk_index
            ]

        )


        results.append(
            result
        )


    return results


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(results):


    if not results:

        print(
            "No results found."
        )

        return


    print()
    print("=" * 70)

    print(
        "Detected intent:",
        results[0]["intent"]
    )

    print("=" * 70)


    for result in results:


        print()
        print(
            f"Rank: {result['rank']}"
        )


        print(
            f"Final score: "
            f"{result['final_score']:.4f}"
        )


        print(
            f"Hybrid: "
            f"{result['hybrid_score']:.4f}"
        )


        print(
            f"Semantic: "
            f"{result['semantic_score']:.4f}"
        )


        print(
            f"BM25: "
            f"{result['bm25_score']:.4f}"
        )


        print(
            f"Keyword: "
            f"{result['keyword_score']:.4f}"
        )


        print(
            f"Objective: "
            f"{result['objective_score']:.4f}"
        )


        print(
            f"Region adjustment: "
            f"{result['region_adjustment']:+.4f}"
        )


        print(
            "Region:",
            result.get(
                "region",
                "unknown"
            )
        )


        print(
            "Page:",
            result.get(
                "page",
                "unknown"
            )
        )


        print(
            "Chunk:",
            result.get(
                "chunk_id",
                "unknown"
            )
        )


        print(
            "Section:",
            result.get(
                "section",
                "unknown"
            )
        )


        print(
            "Subsection:",
            result.get(
                "subsection",
                "unknown"
            )
        )


        print(
            "Content type:",
            result.get(
                "content_type",
                "text"
            )
        )


        print()
        print("Text:")


        print(
            result.get(
                "text",
                ""
            )[:700]
        )


        print(
            "-" * 70
        )


# ============================================================
# INTERACTIVE TEST
# ============================================================

if __name__ == "__main__":


    print()
    print(
        f"Semantic chunks : {len(chunks)}"
    )

    print(
        f"FAISS vectors   : {index.ntotal}"
    )

    print(
        f"Embedding map   : {len(embedding_map)}"
    )


    print()
    print(
        "Ask a question."
    )

    print(
        "Type 'exit' to quit."
    )

    print()


    while True:


        query = input(
            "Query: "
        ).strip()


        if query.lower() in [

            "exit",

            "quit"

        ]:

            break


        if not query:

            continue


        print()
        print(
            "Searching..."
        )


        results = hybrid_search(
            query
        )


        print_results(
            results
        )