import json
import os
import re

from hybrid_search import hybrid_search

try:
    from ollama import chat
except ImportError:
    raise ImportError(
        "Ollama package is missing. Run: pip install ollama"
    )

try:
    from PIL import Image
except ImportError:
    raise ImportError(
        "Pillow is missing. Run: pip install pillow"
    )


# ============================================================
# CONFIGURATION
# ============================================================

LLM_MODEL = "qwen2.5vl:7b"

TOP_K = 5

# Keep text evidence small when working with vision.
MULTIMODAL_TEXT_TOP_K = 2

# Maximum number of pages to inspect separately.
MAX_VISUAL_PAGES = 2

CHUNKS_PATH = "data/processed/semantic_chunks.json"

MAX_IMAGE_WIDTH = 850
MAX_IMAGE_HEIGHT = 1100

RESIZED_IMAGE_DIR = "data/processed/vision_cache"

# Keep prompts/evidence small enough for Qwen's 4096 context.
MAX_TEXT_PER_SOURCE = 1800
MAX_VISUAL_ANALYSIS = 1800


# ============================================================
# CONVERSATION MEMORY
# ============================================================

MAX_MEMORY_TURNS = 4
MAX_MEMORY_CHARS = 3500

conversation_history = []

# Stores the most recent generated answer so the interactive
# loop can add it to conversation memory.
last_answer = ""
last_api_result = {}


def build_memory_context():

    if not conversation_history:
        return ""

    parts = []

    for turn in conversation_history[-MAX_MEMORY_TURNS:]:

        parts.append(
            f"User: {turn['question']}\n"
            f"Assistant: {turn['answer'][:1000]}"
        )

    return "\n\n".join(parts)[:MAX_MEMORY_CHARS]


def build_retrieval_query(question):

    memory = build_memory_context()

    if not memory:
        return question

    return f"""
Previous conversation:
{memory}

Current question:
{question}

Use the previous conversation only to resolve references
such as 'it', 'this', 'that', 'the model', or 'the result'.
Retrieve evidence for the current question.
"""


def add_memory_turn(question, answer):

    if not answer:
        return

    conversation_history.append(
        {
            "question": question,
            "answer": answer
        }
    )

    if len(conversation_history) > MAX_MEMORY_TURNS:

        del conversation_history[
            :-MAX_MEMORY_TURNS
        ]


# ============================================================
# STARTUP — Load chunks if available
# ============================================================

print()
print("=" * 70)
print("MULTIMODAL RESEARCH AGENT")
print("=" * 70)

print(f"Reasoning model : {LLM_MODEL}")
print("Retriever       : Hybrid FAISS + BM25 + Keyword")
print(f"Top-K           : {TOP_K}")
print(f"Multimodal text : {MULTIMODAL_TEXT_TOP_K}")
print(f"Visual pages    : {MAX_VISUAL_PAGES}")

print()

# --------------------------------------------------------
# Gracefully handle missing chunks file (no document
# uploaded yet on fresh install).
# --------------------------------------------------------

if os.path.exists(CHUNKS_PATH):

    print("Loading semantic chunks...")

    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        chunks = json.load(f)

    print(f"Loaded chunks: {len(chunks)}")

else:

    print(
        "No semantic chunks found. "
        "Upload a document via /upload to begin."
    )

    chunks = []


# ============================================================
# TEXT CONTEXT
# ============================================================

def build_context(results):

    """
    Convert retrieved chunks into compact grounded context.
    """

    if not results:

        return (
            "No relevant text evidence was retrieved."
        )

    context_parts = []

    for number, result in enumerate(
        results,
        start=1
    ):

        text = result.get(
            "text",
            ""
        )

        text = text[:MAX_TEXT_PER_SOURCE]

        context_parts.append(
            f"""
[Source {number}]
File: {result.get("source", "Unknown")}
Page: {result.get("page", "Unknown")}
Chunk: {result.get("chunk_id", "Unknown")}
Region: {result.get("region", "Unknown")}
Content type: {result.get("content_type", "text")}

Evidence:
{text}
"""
        )

    return "\n".join(
        context_parts
    )


# ============================================================
# TEXT LLM
# ============================================================

def ask_text_llm(
    question,
    context
):

    memory = build_memory_context()

    memory_section = ""

    if memory:

        memory_section = f"""
CONVERSATION HISTORY
====================

{memory}

Use this history only to resolve references in the
current question. The current question remains primary.
"""

    prompt = f"""
You are a careful research assistant answering
questions about one or more uploaded documents.

Use ONLY the supplied document evidence.

Rules:

1. Do not use outside knowledge.
2. Do not invent facts.
3. Answer the exact question.
4. Be concise but complete.
5. Cite factual claims using [Source N].
6. Only use source numbers that actually exist.

{memory_section}

RETRIEVED EVIDENCE
==================

{context}

QUESTION
========

{question}

ANSWER:
"""

    response = chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.message.content.strip()


# ============================================================
# QUERY DETECTION
# ============================================================

def is_table_query(question):

    words = [
        "table",
        "tables",
        "tabular",
        "row",
        "rows",
        "column",
        "columns",
        "precision",
        "recall",
        "f1-score",
        "f1 score",
        "confusion matrix",
        "confusion matrices"
    ]

    q = question.lower()

    return any(
        word in q
        for word in words
    )


def is_visual_query(question):

    words = [
        "image",
        "images",
        "picture",
        "pictures",
        "photo",
        "photos",
        "figure",
        "figures",
        "diagram",
        "diagrams",
        "chart",
        "charts",
        "graph",
        "graphs",
        "visual",
        "shown",
        "shows",
        "visible",
        "look",
        "see",
        "handwritten",
        "illustration"
    ]

    q = question.lower()

    return any(
        word in q
        for word in words
    )


def is_fusion_query(question):

    """
    Fusion is allowed even when the user does NOT
    explicitly mention an image or table.

    This handles questions such as:

    "Why was XGBoost selected even though Random Forest
     had a higher cross-validation score?"

    Such questions require evidence from multiple
    parts/pages of the paper.
    """

    fusion_words = [
        "compare",
        "comparison",
        "compared",
        "best",
        "better",
        "worst",
        "performance",
        "perform",
        "why",
        "how",
        "explain",
        "according to the document",
        "according to document",
        "discuss",
        "relate",
        "relationship",
        "difference",
        "differences",
        "which model",
        "which models",
        "what does this mean",
        "what does it show",
        "what does the table show",
        "how does",
        "impact",
        "result",
        "results",
        "even though",
        "despite",
        "selected",
        "selection",
        "deployment",
        "deploy",
        "cross-validation",
        "cross validation",
        "independent test",
        "test set",
        "accuracy",
        "macro f1",
        "macro f1-score"
    ]

    q = question.lower()

    return any(
        word in q
        for word in fusion_words
    )


# ============================================================
# PAGE NUMBER
# ============================================================

def extract_requested_page(question):

    patterns = [
        r"\bpage\s*(\d+)\b",
        r"\bp\.\s*(\d+)\b",
        r"\bpg\.?\s*(\d+)\b"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            question,
            flags=re.IGNORECASE
        )

        if match:

            return int(
                match.group(1)
            )

    return None


# ============================================================
# TABLE NUMBER
# ============================================================

def extract_table_number(question):

    patterns = [
        r"\btable\s*([IVXLCDM]+)\b",
        r"\btable\s*(\d+)\b"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            question,
            flags=re.IGNORECASE
        )

        if match:

            return match.group(1).upper()

    return None


# ============================================================
# IMAGE CHUNKS
# ============================================================

def get_image_chunks():

    image_chunks = []

    for chunk in chunks:

        if chunk.get(
            "content_type"
        ) != "image":

            continue

        images = chunk.get(
            "images",
            []
        )

        for image in images:

            path = image.get(
                "path"
            )

            if not path:
                continue

            path = os.path.normpath(
                path
            )

            if not os.path.exists(path):
                continue

            image_chunks.append(
                {
                    "chunk_id": chunk.get(
                        "chunk_id"
                    ),
                    "source": chunk.get(
                        "source"
                    ),
                    "page": chunk.get(
                        "page"
                    ),
                    "path": path
                }
            )

    return image_chunks


# ============================================================
# PAGE IMAGE
# ============================================================

def get_page_image(page_number):

    """
    Get a visual representation of a page.

    Preference:
    1. Embedded image
    2. Rendered page image
    """

    # --------------------------------------------------------
    # Embedded image
    # --------------------------------------------------------

    for image in get_image_chunks():

        if image.get(
            "page"
        ) == page_number:

            return image

    # --------------------------------------------------------
    # Rendered page image
    # --------------------------------------------------------

    for chunk in chunks:

        if chunk.get(
            "page"
        ) != page_number:

            continue

        page_image = chunk.get(
            "page_image",
            {}
        )

        path = page_image.get(
            "path"
        )

        if not path:
            continue

        path = os.path.normpath(
            path
        )

        if os.path.exists(path):

            return {
                "chunk_id": chunk.get(
                    "chunk_id"
                ),
                "source": chunk.get(
                    "source"
                ),
                "page": page_number,
                "path": path
            }

    return None


# ============================================================
# ALL AVAILABLE PAGE IMAGES
# ============================================================

def get_available_pages():

    pages = {}

    for chunk in chunks:

        page = chunk.get(
            "page"
        )

        if page is None:
            continue

        # ----------------------------------------------------
        # Embedded images
        # ----------------------------------------------------

        images = chunk.get(
            "images",
            []
        )

        for image in images:

            path = image.get(
                "path"
            )

            if not path:
                continue

            path = os.path.normpath(
                path
            )

            if os.path.exists(path):

                pages[page] = {
                    "chunk_id": chunk.get(
                        "chunk_id"
                    ),
                    "source": chunk.get(
                        "source"
                    ),
                    "page": page,
                    "path": path
                }

                break

        # ----------------------------------------------------
        # Rendered page image
        # ----------------------------------------------------

        if page not in pages:

            page_image = chunk.get(
                "page_image",
                {}
            )

            path = page_image.get(
                "path"
            )

            if path:

                path = os.path.normpath(
                    path
                )

                if os.path.exists(path):

                    pages[page] = {
                        "chunk_id": chunk.get(
                            "chunk_id"
                        ),
                        "source": chunk.get(
                            "source"
                        ),
                        "page": page,
                        "path": path
                    }

    return pages


# ============================================================
# FIND TABLE PAGES DYNAMICALLY
# ============================================================

def find_table_pages():

    """
    Dynamically discover which pages contain table chunks,
    instead of using hardcoded page numbers.
    """

    table_pages = []

    for chunk in chunks:

        if chunk.get(
            "content_type"
        ) == "table":

            page = chunk.get("page")

            if page is not None and page not in table_pages:

                table_pages.append(page)

    return sorted(table_pages)


# ============================================================
# SELECT VISUAL SOURCES
# ============================================================

def select_visual_sources(
    question,
    text_results=None
):

    if text_results is None:

        text_results = []

    pages = get_available_pages()

    selected = []

    # --------------------------------------------------------
    # Explicit page wins.
    # --------------------------------------------------------

    requested_page = (
        extract_requested_page(
            question
        )
    )

    if requested_page is not None:

        image = pages.get(
            requested_page
        )

        if image:

            return [image]

    # --------------------------------------------------------
    # Explicit table — use dynamic table page discovery.
    # --------------------------------------------------------

    table_number = (
        extract_table_number(
            question
        )
    )

    if table_number:

        table_pages = find_table_pages()

        for tp in table_pages[:MAX_VISUAL_PAGES]:

            image = pages.get(tp)

            if image:

                already = any(
                    item["page"] == tp
                    for item in selected
                )

                if not already:

                    selected.append(image)

    # --------------------------------------------------------
    # Pages from retrieved evidence.
    # --------------------------------------------------------

    for result in text_results:

        page = result.get(
            "page"
        )

        if page is None:
            continue

        image = pages.get(
            page
        )

        if not image:
            continue

        already = any(
            item["page"] == page
            for item in selected
        )

        if not already:

            selected.append(
                image
            )

    # --------------------------------------------------------
    # Comparative questions — use pages from top evidence.
    # --------------------------------------------------------

    q = question.lower()

    comparative = any(
        word in q
        for word in [
            "compare",
            "comparison",
            "even though",
            "despite",
            "why was",
            "why were",
            "why did",
            "performance",
            "selected",
            "selection",
            "deployment",
            "cross-validation",
            "cross validation",
            "independent test",
            "which model",
            "best model"
        ]
    )

    if comparative:

        # Add pages from retrieved results (up to MAX_VISUAL_PAGES)
        for result in text_results:

            page = result.get("page")

            if page is None:
                continue

            image = pages.get(page)

            if not image:
                continue

            already = any(
                item["page"] == page
                for item in selected
            )

            if not already:

                selected.append(image)

            if len(selected) >= MAX_VISUAL_PAGES:
                break

    # --------------------------------------------------------
    # Generic table.
    # --------------------------------------------------------

    if (
        not selected
        and is_table_query(question)
    ):

        table_pages = find_table_pages()

        for tp in table_pages:

            image = pages.get(tp)

            if image:

                selected.append(image)

                break

    # --------------------------------------------------------
    # Generic visual question.
    # --------------------------------------------------------

    if (
        not selected
        and is_visual_query(question)
    ):

        for page in sorted(
            pages.keys()
        ):

            selected.append(
                pages[page]
            )

            if len(selected) >= MAX_VISUAL_PAGES:

                break

    # --------------------------------------------------------
    # Limit.
    # --------------------------------------------------------

    return selected[
        :MAX_VISUAL_PAGES
    ]


# ============================================================
# RESIZE IMAGE
# ============================================================

def prepare_image_for_qwen(
    image_path,
    index=1
):

    os.makedirs(
        RESIZED_IMAGE_DIR,
        exist_ok=True
    )

    filename = os.path.basename(
        image_path
    )

    output_path = os.path.join(
        RESIZED_IMAGE_DIR,
        f"qwen_{index}_{filename}"
    )

    try:

        image = Image.open(
            image_path
        )

        image = image.convert(
            "RGB"
        )

        width, height = image.size

        scale = min(
            MAX_IMAGE_WIDTH / width,
            MAX_IMAGE_HEIGHT / height,
            1.0
        )

        new_width = int(
            width * scale
        )

        new_height = int(
            height * scale
        )

        if scale < 1.0:

            image = image.resize(
                (
                    new_width,
                    new_height
                ),
                Image.Resampling.LANCZOS
            )

        image.save(
            output_path,
            "JPEG",
            quality=75
        )

        print(
            f"Image prepared: "
            f"{new_width}x{new_height}"
        )

        return output_path

    except Exception as error:

        print(
            "Image preparation warning:",
            error
        )

        return image_path


# ============================================================
# VISUAL ANALYSIS PROMPT
# ============================================================

def build_visual_analysis_prompt(
    question,
    page,
    mode
):

    table_number = (
        extract_table_number(
            question
        )
    )

    if mode == "table":

        if table_number:

            focus = (
                f"Focus specifically on Table "
                f"{table_number}."
            )

        else:

            focus = (
                "Identify the table relevant "
                "to the question."
            )

        return f"""
You are inspecting page {page} of a document.

{focus}

Question:
{question}

Analyze ONLY the supplied page image.

For the requested table:

1. Locate the correct table.
2. Identify the table title.
3. Identify the models/rows.
4. Identify the columns.
5. Read the relevant numerical values.
6. Determine the relevant comparison.
7. If a value is unclear, say "unclear".
8. Never invent numbers.
9. Do not discuss unrelated tables.

Return a concise factual visual analysis.
"""

    return f"""
You are inspecting page {page} of a document.

Question:
{question}

Describe only the visual evidence on this page
that helps answer the question.

Read visible text carefully.

Do not invent information.

Return a concise factual visual analysis.
"""


# ============================================================
# ANALYZE ONE IMAGE
# ============================================================

def analyze_single_image(
    question,
    image,
    mode,
    index
):

    prepared_image = (
        prepare_image_for_qwen(
            image["path"],
            index
        )
    )

    prompt = build_visual_analysis_prompt(
        question,
        image["page"],
        mode
    )

    response = chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [
                    prepared_image
                ]
            }
        ]
    )

    answer = (
        response.message.content
        .strip()
    )

    return answer[
        :MAX_VISUAL_ANALYSIS
    ]


# ============================================================
# MULTI-PAGE VISUAL ANALYSIS
# ============================================================

def analyze_visual_sources(
    question,
    images,
    mode
):

    analyses = []

    print()

    for index, image in enumerate(
        images,
        start=1
    ):

        print(
            f"Analyzing visual source "
            f"{index}/{len(images)}..."
        )

        print(
            f"Page: {image['page']}"
        )

        try:

            analysis = analyze_single_image(
                question,
                image,
                mode,
                index
            )

            analyses.append(
                {
                    "number": index,
                    "page": image["page"],
                    "source": image["source"],
                    "chunk_id": image["chunk_id"],
                    "analysis": analysis
                }
            )

            print(
                f"Visual analysis complete "
                f"for page {image['page']}."
            )

        except Exception as error:

            print(
                f"Visual analysis failed "
                f"for page {image['page']}:"
            )

            print(
                type(error).__name__,
                str(error)
            )

    return analyses


# ============================================================
# BUILD VISUAL CONTEXT
# ============================================================

def build_visual_context(
    analyses
):

    if not analyses:

        return (
            "No visual evidence was successfully analyzed."
        )

    parts = []

    for item in analyses:

        parts.append(
            f"""
[Visual Source {item['number']}]
Page: {item['page']}
File: {item['source']}
Chunk: {item['chunk_id']}

Visual findings:
{item['analysis']}
"""
        )

    return "\n".join(parts)


# ============================================================
# FINAL FUSION LLM
# ============================================================

def ask_fusion_llm(
    question,
    text_context,
    visual_context
):

    memory = build_memory_context()

    memory_section = ""

    if memory:

        memory_section = f"""
CONVERSATION HISTORY
====================

{memory}

Use this history only to resolve references in the
current question. The current question remains primary.
"""

    prompt = f"""
You are the final reasoning component of a
multimodal assistant.

Answer the user's question using ONLY:

1. Retrieved textual evidence.
2. Visual analyses extracted from document pages.

Do not use outside knowledge.

IMPORTANT:

- If the question asks about a table, use the
  visual table evidence for the actual table comparison.
- Use textual evidence for what the document says.
- Use visual evidence for what the document shows.
- Do not invent numerical values.
- If evidence conflicts, explicitly explain the
  conflict instead of silently choosing.
- Cite textual evidence as [Source N].
- Cite visual evidence as [Visual Source N].
- Keep the final answer concise.

{memory_section}

TEXTUAL EVIDENCE
================

{text_context}

VISUAL EVIDENCE
===============

{visual_context}

USER QUESTION
=============

{question}

Now answer the question.

ANSWER:
"""

    response = chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.message.content.strip()


# ============================================================
# CITATION HELPERS
# ============================================================

def get_valid_text_citations(
    answer,
    number_of_sources
):

    matches = re.findall(
        r"\[Source\s+(\d+)\]",
        answer,
        flags=re.IGNORECASE
    )

    valid = []

    for match in matches:

        number = int(match)

        if (
            1 <= number
            <= number_of_sources
        ):

            valid.append(
                number
            )

    return sorted(
        set(valid)
    )


def get_valid_visual_citations(
    answer,
    number_of_sources
):

    matches = re.findall(
        r"\[Visual Source\s+(\d+)\]",
        answer,
        flags=re.IGNORECASE
    )

    valid = []

    for match in matches:

        number = int(match)

        if (
            1 <= number
            <= number_of_sources
        ):

            valid.append(
                number
            )

    return sorted(
        set(valid)
    )


# ============================================================
# PRINT RETRIEVAL RESULTS
# ============================================================

def print_retrieval_results(
    results
):

    print()
    print("=" * 70)
    print("RETRIEVED EVIDENCE")
    print("=" * 70)

    for number, result in enumerate(
        results,
        start=1
    ):

        print()
        print(
            f"[Source {number}]"
        )

        print(
            "Rank:",
            result.get(
                "rank",
                number
            )
        )

        print(
            "Final score:",
            f"{result.get('final_score', 0):.4f}"
        )

        print(
            "Semantic:",
            f"{result.get('semantic_score', 0):.4f}"
        )

        print(
            "BM25:",
            f"{result.get('bm25_score', 0):.4f}"
        )

        print(
            "Keyword:",
            f"{result.get('keyword_score', 0):.4f}"
        )

        print(
            "Objective:",
            f"{result.get('objective_score', 0):.4f}"
        )

        print(
            "File:",
            result.get(
                "source",
                "Unknown"
            )
        )

        print(
            "Page:",
            result.get(
                "page",
                "Unknown"
            )
        )

        print(
            "Chunk:",
            result.get(
                "chunk_id",
                "Unknown"
            )
        )

        print(
            "Content type:",
            result.get(
                "content_type",
                "text"
            )
        )

        print("Text:")

        print(
            result.get(
                "text",
                ""
            )[:500]
        )

        print(
            "-" * 70
        )


# ============================================================
# NORMAL TEXT QUESTION
# ============================================================

def process_text_question(
    question
):

    global last_answer, last_api_result

    last_answer = ""

    print()
    print(
        "Searching documents..."
    )

    retrieval_query = (
        build_retrieval_query(
            question
        )
    )

    results = hybrid_search(
        retrieval_query,
        top_k=TOP_K
    )

    if not results:

        print(
            "No relevant documents found."
        )

        last_api_result = {
            "answer": (
                "I could not find relevant information "
                "in the uploaded documents to answer "
                "your question. Please try rephrasing."
            ),
            "sources": []
        }

        return

    print_retrieval_results(
        results
    )

    print()
    print(
        "Building grounded context..."
    )

    context = build_context(
        results
    )

    print(
        f"Generating answer with "
        f"{LLM_MODEL}..."
    )

    answer = ask_text_llm(
        question,
        context
    )

    last_answer = answer

    citations = (
        get_valid_text_citations(
            answer,
            len(results)
        )
    )

    if not citations:

        count = min(
            2,
            len(results)
        )

        answer += (
            "\n\n" +
            "".join(
                f"[Source {i}]"
                for i in range(
                    1,
                    count + 1
                )
            )
        )

        citations = list(
            range(
                1,
                count + 1
            )
        )

    print()
    print("=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(answer)

    print()
    print("=" * 70)
    print("SOURCES USED")
    print("=" * 70)

    sources_list = []

    for number in citations:

        result = results[
            number - 1
        ]

        sources_list.append(result)

        print(
            f"[Source {number}] "
            f"{result.get('source', 'Unknown')} "
            f"— Page "
            f"{result.get('page', 'Unknown')} "
            f"— Chunk "
            f"{result.get('chunk_id', 'Unknown')}"
        )

    # Store for API response.
    last_api_result = {
        "answer": answer,
        "sources": sources_list
    }


# ============================================================
# VISUAL-ONLY QUESTION
# ============================================================

def process_visual_question(
    question,
    mode
):

    global last_answer, last_api_result

    last_answer = ""

    if mode == "table":

        print()
        print(
            "Table question detected."
        )

        table_number = (
            extract_table_number(
                question
            )
        )

        if table_number:

            print(
                f"Requested table: "
                f"{table_number}"
            )

    else:

        print()
        print(
            "Visual question detected."
        )

    print(
        "Selecting relevant document image..."
    )

    retrieval_query = (
        build_retrieval_query(
            question
        )
    )

    results = hybrid_search(
        retrieval_query,
        top_k=TOP_K
    )

    images = select_visual_sources(
        question,
        results[:MULTIMODAL_TEXT_TOP_K]
    )

    if not images:

        print(
            "No suitable document image found."
        )

        last_api_result = {
            "answer": (
                "No visual content was found in the "
                "uploaded document to answer your question."
            ),
            "sources": []
        }

        return

    print(
        f"Selected pages: "
        f"{len(images)}"
    )

    for index, image in enumerate(
        images,
        start=1
    ):

        print(
            f"  Visual Source {index}: "
            f"Page {image['page']}"
        )

    print()
    print(
        "Analyzing selected visual evidence..."
    )

    analyses = analyze_visual_sources(
        question,
        images,
        mode
    )

    if not analyses:

        print(
            "No visual analysis was produced."
        )

        last_api_result = {
            "answer": (
                "Visual analysis could not be completed "
                "for the document pages."
            ),
            "sources": []
        }

        return

    text_results = results[
        :MULTIMODAL_TEXT_TOP_K
    ]

    text_context = build_context(
        text_results
    )

    visual_context = build_visual_context(
        analyses
    )

    print()
    print(
        "Generating grounded answer..."
    )

    answer = ask_fusion_llm(
        question,
        text_context,
        visual_context
    )

    last_answer = answer

    text_citations = (
        get_valid_text_citations(
            answer,
            len(text_results)
        )
    )

    visual_citations = (
        get_valid_visual_citations(
            answer,
            len(analyses)
        )
    )

    if not visual_citations:

        answer += (
            "\n\n[Visual Source 1]"
        )

        visual_citations = [1]

    print()
    print("=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(answer)

    print()
    print("=" * 70)
    print("SOURCES USED")
    print("=" * 70)

    sources_list = []

    for number in text_citations:

        result = text_results[
            number - 1
        ]

        sources_list.append(result)

        print(
            f"[Source {number}] "
            f"{result.get('source', 'Unknown')} "
            f"— Page "
            f"{result.get('page', 'Unknown')} "
            f"— Chunk "
            f"{result.get('chunk_id', 'Unknown')}"
        )

    for number in visual_citations:

        image = analyses[
            number - 1
        ]

        sources_list.append(image)

        print(
            f"[Visual Source {number}] "
            f"{image['source']} "
            f"— Page "
            f"{image['page']} "
            f"— Chunk "
            f"{image['chunk_id']}"
        )

    # Store for API response.
    last_api_result = {
        "answer": answer,
        "sources": sources_list
    }


# ============================================================
# MULTI-PAGE FUSION QUESTION
# ============================================================

def process_fusion_question(
    question
):

    global last_answer, last_api_result

    last_answer = ""

    print()
    print(
        "Multimodal question detected."
    )

    # --------------------------------------------------------
    # TEXT RETRIEVAL
    # --------------------------------------------------------

    print(
        "Retrieving text evidence..."
    )

    retrieval_query = (
        build_retrieval_query(
            question
        )
    )

    results = hybrid_search(
        retrieval_query,
        top_k=TOP_K
    )

    if results:

        text_results = results[
            :MULTIMODAL_TEXT_TOP_K
        ]

        print_retrieval_results(
            text_results
        )

        text_context = build_context(
            text_results
        )

    else:

        text_results = []

        text_context = (
            "No relevant text evidence "
            "was retrieved."
        )

    # --------------------------------------------------------
    # VISUAL SELECTION
    # --------------------------------------------------------

    print()
    print(
        "Selecting visual evidence..."
    )

    images = select_visual_sources(
        question,
        text_results
    )

    if not images:

        print(
            "No visual evidence found."
        )

        if results:

            process_text_question(
                question
            )

        else:

            last_api_result = {
                "answer": (
                    "No relevant information found "
                    "in the uploaded documents."
                ),
                "sources": []
            }

        return

    print(
        f"Selected visual pages: "
        f"{len(images)}"
    )

    for index, image in enumerate(
        images,
        start=1
    ):

        print(
            f"  Visual Source {index}: "
            f"Page {image['page']}"
        )

    table_number = (
        extract_table_number(
            question
        )
    )

    if table_number:

        mode = "table"

        print(
            f"Table focus: "
            f"{table_number}"
        )

    else:

        mode = "visual"

    # --------------------------------------------------------
    # Analyze each image separately.
    # This prevents context-size errors.
    # --------------------------------------------------------

    print()
    print(
        "Analyzing visual evidence "
        "page-by-page..."
    )

    analyses = analyze_visual_sources(
        question,
        images,
        mode
    )

    if not analyses:

        print(
            "Visual analysis failed."
        )

        if text_results:

            print(
                "Falling back to text evidence..."
            )

            answer = ask_text_llm(
                question,
                text_context
            )

            last_answer = answer

            print()
            print("=" * 70)
            print("ANSWER")
            print("=" * 70)

            print(answer)

            last_api_result = {
                "answer": answer,
                "sources": text_results
            }

        else:

            last_api_result = {
                "answer": "Visual analysis failed and no text evidence was found.",
                "sources": []
            }

        return

    visual_context = build_visual_context(
        analyses
    )

    # --------------------------------------------------------
    # FINAL FUSION
    # --------------------------------------------------------

    print()
    print(
        "Combining text + visual evidence..."
    )

    answer = ask_fusion_llm(
        question,
        text_context,
        visual_context
    )

    last_answer = answer

    # --------------------------------------------------------
    # CITATIONS
    # --------------------------------------------------------

    text_citations = (
        get_valid_text_citations(
            answer,
            len(text_results)
        )
    )

    visual_citations = (
        get_valid_visual_citations(
            answer,
            len(analyses)
        )
    )

    if (
        not text_citations
        and text_results
    ):

        answer += (
            "\n\n[Source 1]"
        )

        text_citations = [1]

    if not visual_citations:

        answer += (
            "\n[Visual Source 1]"
        )

        visual_citations = [1]

    # --------------------------------------------------------
    # API RESULT
    # --------------------------------------------------------

    sources_list = []

    for number in text_citations:
        if number - 1 < len(text_results):
            sources_list.append(text_results[number - 1])

    for number in visual_citations:
        if number - 1 < len(analyses):
            sources_list.append(analyses[number - 1])

    last_api_result = {
        "answer": answer,
        "sources": sources_list
    }

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(answer)

    print()
    print("=" * 70)
    print("SOURCES USED")
    print("=" * 70)

    for number in text_citations:

        result = text_results[
            number - 1
        ]

        print(
            f"[Source {number}] "
            f"{result.get('source', 'Unknown')} "
            f"— Page "
            f"{result.get('page', 'Unknown')} "
            f"— Chunk "
            f"{result.get('chunk_id', 'Unknown')}"
        )

    for number in visual_citations:

        image = analyses[
            number - 1
        ]

        print(
            f"[Visual Source {number}] "
            f"{image['source']} "
            f"— Page "
            f"{image['page']} "
            f"— Chunk "
            f"{image['chunk_id']}"
        )


# ============================================================
# DOCUMENT COMPARISON
# ============================================================

def is_document_comparison_query(
    question
):

    phrases = [
        "compare the documents",
        "compare these documents",
        "compare the two documents",
        "compare documents",
        "what documents are available",
        "which documents are available",
        "what is each document about",
        "what are the documents about",
        "what is each pdf about",
        "what are the pdfs about",
        "compare the pdfs",
        "compare the two pdfs",
        "compare the documents",
        "compare these files",
        "difference between the documents",
        "differences between the documents",
        "difference between the two documents",
        "differences between the two documents",
    ]

    q = question.lower().strip()

    return any(
        phrase in q
        for phrase in phrases
    )


def get_document_sources():

    sources = []

    for chunk in chunks:

        source = chunk.get(
            "source"
        )

        if not source:
            continue

        if source not in sources:

            sources.append(
                source
            )

    return sources


def get_document_text_evidence(
    source,
    max_chunks=4
):

    candidates = []

    for chunk in chunks:

        if chunk.get(
            "source"
        ) != source:
            continue

        if chunk.get(
            "content_type"
        ) != "text":
            continue

        text = str(
            chunk.get(
                "text",
                ""
            )
        ).strip()

        if not text:
            continue

        # Ignore tiny placeholders.
        if len(text) < 80:
            continue

        candidates.append(
            {
                "page": chunk.get(
                    "page"
                ),
                "chunk_id": chunk.get(
                    "chunk_id"
                ),
                "text": text
            }
        )

    # Prefer chunks containing more actual information.
    candidates.sort(
        key=lambda item: len(item["text"]),
        reverse=True
    )

    return candidates[
        :max_chunks
    ]


def get_document_image_evidence(
    source
):

    # Use the first available embedded image belonging
    # to this source. This is mainly useful for the
    # scanned document.
    for item in get_image_chunks():

        if item.get(
            "source"
        ) == source:

            return item

    return None


def analyze_document_image(
    question,
    image
):

    prepared = prepare_image_for_qwen(
        image["path"],
        1
    )

    prompt = f"""
You are identifying the type and subject of a document.

Question:
{question}

Inspect the supplied document page.

Identify:
1. What type of document/content it is.
2. The main subject/topic.
3. Any clearly visible title, heading, or structure.
4. Any useful language or educational/business context.

Do not invent information.
Do not infer details that are not visible.

Return a concise description.
"""

    response = chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [
                    prepared
                ]
            }
        ]
    )

    return response.message.content.strip()[:1600]


def process_document_comparison(
    question
):

    global last_answer, last_api_result

    last_answer = ""

    print()
    print(
        "Document comparison detected."
    )

    sources = get_document_sources()

    if not sources:

        print(
            "No documents found."
        )

        last_api_result = {
            "answer": "No documents are currently loaded.",
            "sources": []
        }

        return

    print()
    print(
        "Documents found:"
    )

    for index, source in enumerate(
        sources,
        start=1
    ):

        print(
            f"  [{index}] {source}"
        )

    document_evidence = []

    for index, source in enumerate(
        sources,
        start=1
    ):

        print()
        print(
            f"Collecting evidence for document "
            f"{index}/{len(sources)}..."
        )

        text_chunks = (
            get_document_text_evidence(
                source
            )
        )

        text_parts = []

        for chunk in text_chunks:

            text_parts.append(
                f"Page {chunk['page']} "
                f"(Chunk {chunk['chunk_id']}):\n"
                f"{chunk['text'][:1400]}"
            )

        text_evidence = (
            "\n\n".join(
                text_parts
            )
        )

        visual_evidence = ""

        image = (
            get_document_image_evidence(
                source
            )
        )

        # If the source is primarily visual/scanned,
        # let Qwen inspect one representative image.
        if image:

            print(
                f"  Visual evidence: page "
                f"{image['page']}"
            )

            try:

                visual_evidence = (
                    analyze_document_image(
                        question,
                        image
                    )
                )

            except Exception as error:

                print(
                    "  Visual analysis skipped:",
                    error
                )

        if not text_evidence:

            text_evidence = (
                "No usable text chunks were available."
            )

        if not visual_evidence:

            visual_evidence = (
                "No additional visual evidence was available."
            )

        document_evidence.append(
            {
                "number": index,
                "source": source,
                "text": text_evidence[:4500],
                "visual": visual_evidence[:1600]
            }
        )

    # --------------------------------------------------------
    # Final comparison prompt.
    # --------------------------------------------------------

    context_parts = []

    for item in document_evidence:

        context_parts.append(
            f"""
[Document {item['number']}]
File:
{item['source']}

Text evidence:
{item['text']}

Visual evidence:
{item['visual']}
"""
        )

    document_context = "\n".join(
        context_parts
    )

    memory = build_memory_context()

    memory_section = ""

    if memory:

        memory_section = f"""
CONVERSATION HISTORY
====================

{memory}

Use this only to resolve references.
"""

    prompt = f"""
You are comparing documents in a local
document collection.

DOCUMENT EVIDENCE
=================

{document_context}

INSTRUCTIONS:

1. Identify every distinct document.
2. Describe what each document is about separately.
3. Never merge facts from one document into another.
4. State the document type when supported by evidence.
5. If one document is scanned or handwritten, say so
   only when supported by the visual evidence.
6. Explain the major difference in subject/content
   between the documents.
7. Do not invent information.
8. Cite each document using [Document N].
9. Keep the answer concise.

ANSWER:
"""

    response = chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = (
        response.message.content
        .strip()
    )

    last_answer = answer

    last_api_result = {
        "answer": answer,
        "sources": document_evidence
    }

    print()
    print("=" * 70)
    print("DOCUMENT COMPARISON")
    print("=" * 70)

    print(
        answer
    )

    print()
    print("=" * 70)
    print("DOCUMENTS USED")
    print("=" * 70)

    for item in document_evidence:

        print(
            f"[Document {item['number']}] "
            f"{item['source']}"
        )


# ============================================================
# MAIN ROUTER
# ============================================================

def process_question(
    question
):

    # --------------------------------------------------------
    # Priority 0:
    # Document comparison.
    # --------------------------------------------------------

    if is_document_comparison_query(
        question
    ):

        process_document_comparison(
            question
        )

        return

    # --------------------------------------------------------
    # Priority 1:
    # Multimodal/fusion questions.
    # --------------------------------------------------------

    if is_fusion_query(
        question
    ):

        process_fusion_question(
            question
        )

        return

    # --------------------------------------------------------
    # Priority 2:
    # Table questions.
    # --------------------------------------------------------

    if is_table_query(
        question
    ):

        process_visual_question(
            question,
            "table"
        )

        return

    # --------------------------------------------------------
    # Priority 3:
    # Image/visual questions.
    # --------------------------------------------------------

    if is_visual_query(
        question
    ):

        process_visual_question(
            question,
            "visual"
        )

        return

    # --------------------------------------------------------
    # Priority 4:
    # Normal text RAG.
    # --------------------------------------------------------

    process_text_question(
        question
    )


# ============================================================
# API ENDPOINT
# ============================================================

def process_question_api(question):

    global last_api_result

    last_api_result = {
        "answer": "No answer generated.",
        "sources": []
    }

    process_question(question)

    # Save this completed turn for follow-up questions in memory.
    add_memory_turn(question, last_answer)

    return last_api_result


# ============================================================
# PIPELINE INFORMATION
# ============================================================

print()
print("Pipeline:")
print(
    "Query -> Query Router -> "
    "Hybrid Retrieval -> "
    "Text / Visual / Table / Fusion -> "
    "Page Selection -> "
    "Page-by-Page Vision -> "
    "Evidence Fusion -> "
    "Qwen2.5-VL -> "
    "Grounded Answer"
)

print("=" * 70)


# ============================================================
# INTERACTIVE LOOP — Only when run directly (not imported)
# ============================================================

if __name__ == "__main__":

    while True:

        try:

            question = input(
                "\nAsk a question "
                "(type 'exit' to quit): "
            ).strip()

        except KeyboardInterrupt:

            print(
                "\n\nExiting..."
            )

            break

        if question.lower() in [
            "exit",
            "quit"
        ]:

            print(
                "\nExiting..."
            )

            break

        if not question:

            continue

        try:

            process_question(
                question
            )

            # Save this completed turn for follow-up questions.
            add_memory_turn(
                question,
                last_answer
            )

        except KeyboardInterrupt:

            print(
                "\n\nOperation cancelled."
            )

        except Exception as error:

            print()
            print("=" * 70)
            print("ERROR")
            print("=" * 70)

            print(
                type(error).__name__
            )

            print(
                str(error)
            )