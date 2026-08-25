import json
import os
import re


# ============================================================
# CONFIGURATION
# ============================================================

DOCUMENT_ELEMENTS_PATH = (
    "data/processed/document_elements.json"
)

LEGACY_INPUT_PATH = (
    "data/processed/chunks.json"
)

OUTPUT_PATH = (
    "data/processed/semantic_chunks.json"
)


# ============================================================
# CHUNKING CONFIGURATION
# ============================================================

TARGET_WORDS = 90

MIN_WORDS = 40

MAX_WORDS = 140


# ============================================================
# MULTIMODAL SETTINGS
# ============================================================

INCLUDE_IMAGE_METADATA = True

INCLUDE_TABLE_CONTENT = True

INCLUDE_EMPTY_VISUAL_CHUNKS = True


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    if not text:
        return ""

    # --------------------------------------------------------
    # Normalize whitespace
    # --------------------------------------------------------

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    # --------------------------------------------------------
    # Fix common PDF hyphenation
    #
    # Example:
    #
    # aquapon-
    # ics
    #
    # becomes:
    #
    # aquaponics
    # --------------------------------------------------------

    text = re.sub(
        r"(\w+)-\s*\n\s*(\w+)",
        r"\1\2",
        text
    )

    # --------------------------------------------------------
    # Normalize line endings
    # --------------------------------------------------------

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    return text.strip()


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_sentences(text):

    text = clean_text(
        text
    )

    if not text:
        return []


    # --------------------------------------------------------
    # Protect decimal numbers
    #
    # Example:
    #
    # 0.9593
    #
    # should not become two sentences.
    # --------------------------------------------------------

    protected = re.sub(
        r"(\d)\.(\d)",
        r"\1<DOT>\2",
        text
    )


    # --------------------------------------------------------
    # Protect common abbreviations
    # --------------------------------------------------------

    protected = re.sub(
        r"\b("
        r"e\.g|"
        r"i\.e|"
        r"etc|"
        r"Dr|"
        r"Mr|"
        r"Mrs|"
        r"Ms|"
        r"Fig|"
        r"Eq|"
        r"approx|"
        r"vs"
        r")\.",
        lambda match:
            match.group(0).replace(
                ".",
                "<DOT>"
            ),
        protected,
        flags=re.IGNORECASE
    )


    # --------------------------------------------------------
    # Split sentences
    #
    # English punctuation is used here, but the original
    # Unicode text is preserved.
    # --------------------------------------------------------

    sentences = re.split(
        r"(?<=[.!?])\s+",
        protected
    )


    # --------------------------------------------------------
    # Restore protected periods
    # --------------------------------------------------------

    sentences = [

        sentence.replace(
            "<DOT>",
            "."
        ).strip()

        for sentence in sentences

        if sentence.strip()
    ]


    return sentences


# ============================================================
# WORD COUNT
# ============================================================

def word_count(text):

    if not text:
        return 0

    return len(
        text.split()
    )


# ============================================================
# SENTENCE CHUNKING
# ============================================================

def make_sentence_chunks(
    sentences,
    target_words=TARGET_WORDS,
    max_words=MAX_WORDS
):

    chunks = []

    current = []

    current_words = 0


    for sentence in sentences:

        sentence_words = word_count(
            sentence
        )


        # ----------------------------------------------------
        # Very long sentence
        #
        # Keep it intact instead of destroying meaning.
        # ----------------------------------------------------

        if (
            sentence_words > max_words
            and not current
        ):

            chunks.append(
                sentence
            )

            continue


        # ----------------------------------------------------
        # Would exceed target?
        # ----------------------------------------------------

        if (
            current
            and
            current_words + sentence_words
            > target_words
        ):

            chunks.append(
                " ".join(
                    current
                )
            )

            current = [
                sentence
            ]

            current_words = (
                sentence_words
            )

            continue


        current.append(
            sentence
        )

        current_words += (
            sentence_words
        )


    # --------------------------------------------------------
    # Remaining chunk
    # --------------------------------------------------------

    if current:

        chunks.append(
            " ".join(
                current
            )
        )


    return chunks


# ============================================================
# MERGE SMALL CHUNKS
# ============================================================

def merge_small_chunks(
    chunks
):

    if not chunks:
        return []


    merged = []


    for chunk in chunks:

        # ----------------------------------------------------
        # Try merging a small chunk into the previous chunk.
        # ----------------------------------------------------

        if (
            merged
            and
            word_count(chunk)
            < MIN_WORDS
        ):

            combined = (
                merged[-1]
                + " "
                + chunk
            )


            if (
                word_count(combined)
                <= MAX_WORDS
            ):

                merged[-1] = (
                    combined
                )

            else:

                merged.append(
                    chunk
                )

        else:

            merged.append(
                chunk
            )


    return merged


# ============================================================
# TABLE TO TEXT
# ============================================================

def table_to_text(
    table
):

    if not table:
        return ""


    rows = table.get(
        "rows",
        []
    )


    if not rows:
        return ""


    lines = []


    for row_index, row in enumerate(
        rows
    ):

        if not row:
            continue


        cells = []


        for cell in row:

            if cell is None:

                cell_text = ""

            else:

                cell_text = clean_text(
                    str(cell)
                )


            cells.append(
                cell_text
            )


        row_text = " | ".join(
            cells
        )


        if row_text.strip():

            lines.append(
                f"Row {row_index + 1}: "
                f"{row_text}"
            )


    return "\n".join(
        lines
    )


# ============================================================
# BUILD TABLE TEXT
# ============================================================

def build_table_context(
    tables
):

    if not INCLUDE_TABLE_CONTENT:
        return ""


    if not tables:
        return ""


    parts = []


    for table_index, table in enumerate(
        tables,
        start=1
    ):

        table_text = table_to_text(
            table
        )


        if not table_text:
            continue


        parts.append(
            f"TABLE {table_index}\n"
            f"{table_text}"
        )


    return "\n\n".join(
        parts
    )


# ============================================================
# BUILD IMAGE CONTEXT
# ============================================================

def build_image_context(
    images
):

    if not INCLUDE_IMAGE_METADATA:
        return ""


    if not images:
        return ""


    parts = []


    for index, image in enumerate(
        images,
        start=1
    ):

        width = image.get(
            "width"
        )

        height = image.get(
            "height"
        )

        extension = image.get(
            "extension",
            "unknown"
        )


        image_description = (
            f"Embedded image {index} "
            f"({extension})"
        )


        if width and height:

            image_description += (
                f", dimensions "
                f"{width}x{height}"
            )


        parts.append(
            image_description
        )


    return "\n".join(
        parts
    )


# ============================================================
# BUILD VISUAL CONTEXT
# ============================================================

def build_visual_context(
    page
):

    parts = []


    # --------------------------------------------------------
    # Embedded images
    # --------------------------------------------------------

    images = page.get(
        "images",
        []
    )


    image_context = (
        build_image_context(
            images
        )
    )


    if image_context:

        parts.append(
            image_context
        )


    # --------------------------------------------------------
    # Tables
    # --------------------------------------------------------

    tables = page.get(
        "tables",
        []
    )


    table_context = (
        build_table_context(
            tables
        )
    )


    if table_context:

        parts.append(
            table_context
        )


    return "\n\n".join(
        parts
    )


# ============================================================
# LOAD DOCUMENT ELEMENTS
# ============================================================

def load_document_elements():

    if os.path.exists(
        DOCUMENT_ELEMENTS_PATH
    ):

        print(
            "Loading multimodal "
            "document elements..."
        )


        with open(
            DOCUMENT_ELEMENTS_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(
                file
            )


        # ----------------------------------------------------
        # New format
        # ----------------------------------------------------

        if isinstance(
            data,
            dict
        ):

            documents = data.get(
                "documents",
                []
            )


            print(
                "Documents loaded: "
                f"{len(documents)}"
            )


            return documents, "multimodal"


        # ----------------------------------------------------
        # Unexpected but usable list
        # ----------------------------------------------------

        if isinstance(
            data,
            list
        ):

            print(
                "Document elements loaded: "
                f"{len(data)}"
            )


            return data, "legacy-list"


    # --------------------------------------------------------
    # Fallback to legacy chunks.json
    # --------------------------------------------------------

    if os.path.exists(
        LEGACY_INPUT_PATH
    ):

        print(
            "document_elements.json "
            "not found."
        )

        print(
            "Falling back to chunks.json..."
        )


        with open(
            LEGACY_INPUT_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            blocks = json.load(
                file
            )


        print(
            "Legacy layout blocks loaded: "
            f"{len(blocks)}"
        )


        return blocks, "legacy"


    raise FileNotFoundError(
        "\nNo input file found.\n\n"
        f"Expected:\n"
        f"  {DOCUMENT_ELEMENTS_PATH}\n"
        f"or\n"
        f"  {LEGACY_INPUT_PATH}"
    )


# ============================================================
# PROCESS LEGACY BLOCK
# ============================================================

def process_legacy_block(
    block,
    chunk_id
):

    text = clean_text(
        block.get(
            "text",
            ""
        )
    )


    if not text:
        return [], chunk_id


    page = block.get(
        "page",
        0
    )


    region = block.get(
        "region",
        "body"
    )


    section = block.get(
        "section",
        "Unknown"
    )


    subsection = block.get(
        "subsection",
        "Unknown"
    )


    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    if region == "header":

        chunk = {

            "chunk_id":
                chunk_id,

            "page":
                page,

            "region":
                region,

            "section":
                section,

            "subsection":
                subsection,

            "source":
                block.get(
                    "source",
                    None
                ),

            "text":
                text,

            "content_type":
                "text",

            "ocr_used":
                False,

            "has_visual_content":
                False
        }


        return [
            chunk
        ], chunk_id + 1


    # --------------------------------------------------------
    # Normal text
    # --------------------------------------------------------

    sentences = split_sentences(
        text
    )


    if not sentences:

        return [], chunk_id


    semantic_chunks = (
        make_sentence_chunks(
            sentences
        )
    )


    semantic_chunks = (
        merge_small_chunks(
            semantic_chunks
        )
    )


    output = []


    for chunk_text in semantic_chunks:

        if not chunk_text.strip():
            continue


        output.append({

            "chunk_id":
                chunk_id,

            "page":
                page,

            "region":
                region,

            "section":
                section,

            "subsection":
                subsection,

            "source":
                block.get(
                    "source",
                    None
                ),

            "text":
                chunk_text,

            "content_type":
                "text",

            "ocr_used":
                False,

            "has_visual_content":
                False
        })


        chunk_id += 1


    return output, chunk_id


# ============================================================
# PROCESS MULTIMODAL PAGE
# ============================================================

def process_multimodal_page(
    page,
    document_source,
    chunk_id
):

    page_number = page.get(
        "page",
        0
    )


    source = page.get(
        "source",
        document_source
    )


    text = clean_text(
        page.get(
            "text",
            ""
        )
    )


    ocr_used = bool(
        page.get(
            "ocr_used",
            False
        )
    )


    ocr_confidence = page.get(
        "ocr_confidence",
        0.0
    )


    ocr_quality = page.get(
        "ocr_quality",
        0.0
    )


    page_image = page.get(
        "page_image"
    )


    images = page.get(
        "images",
        []
    )


    tables = page.get(
        "tables",
        []
    )


    has_visual_content = bool(
        page.get(
            "has_visual_content",
            False
        )
    )


    # ========================================================
    # TEXT CHUNKING
    # ========================================================

    output = []


    if text:

        sentences = split_sentences(
            text
        )


        if sentences:

            semantic_chunks = (
                make_sentence_chunks(
                    sentences
                )
            )


            semantic_chunks = (
                merge_small_chunks(
                    semantic_chunks
                )
            )


            for chunk_text in semantic_chunks:

                if not chunk_text.strip():
                    continue


                output.append({

                    "chunk_id":
                        chunk_id,

                    "source":
                        source,

                    "page":
                        page_number,

                    "region":
                        "body",

                    "section":
                        "Unknown",

                    "subsection":
                        "Unknown",

                    "text":
                        chunk_text,

                    "content_type":
                        "text",

                    "ocr_used":
                        ocr_used,

                    "ocr_languages":
                        (
                            page.get(
                                "ocr_languages"
                            )
                            if ocr_used
                            else None
                        ),

                    "ocr_confidence":
                        ocr_confidence,

                    "ocr_quality":
                        ocr_quality,

                    "page_image":
                        page_image,

                    "images":
                        images,

                    "tables":
                        tables,

                    "image_count":
                        len(images),

                    "table_count":
                        len(tables),

                    "has_visual_content":
                        has_visual_content
                })


                chunk_id += 1


    # ========================================================
    # TABLE CHUNK
    # ========================================================

    table_context = (
        build_table_context(
            tables
        )
    )


    if table_context:

        output.append({

            "chunk_id":
                chunk_id,

            "source":
                source,

            "page":
                page_number,

            "region":
                "table",

            "section":
                "Unknown",

            "subsection":
                "Unknown",

            "text":
                table_context,

            "content_type":
                "table",

            "ocr_used":
                ocr_used,

            "ocr_languages":
                (
                    page.get(
                        "ocr_languages"
                    )
                    if ocr_used
                    else None
                ),

            "ocr_confidence":
                ocr_confidence,

            "ocr_quality":
                ocr_quality,

            "page_image":
                page_image,

            "images":
                images,

            "tables":
                tables,

            "image_count":
                len(images),

            "table_count":
                len(tables),

            "has_visual_content":
                True
        })


        chunk_id += 1


    # ========================================================
    # IMAGE / VISUAL CHUNK
    # ========================================================
    #
    # We don't invent descriptions for images.
    #
    # We only preserve their existence and metadata.
    #
    # Later, a vision model can generate descriptions from
    # the saved page/image.
    # ========================================================

    image_context = (
        build_image_context(
            images
        )
    )


    if (
        image_context
        and
        INCLUDE_EMPTY_VISUAL_CHUNKS
    ):

        visual_text = (
            "Visual content on page "
            f"{page_number}.\n"
            f"{image_context}"
        )


        output.append({

            "chunk_id":
                chunk_id,

            "source":
                source,

            "page":
                page_number,

            "region":
                "image",

            "section":
                "Unknown",

            "subsection":
                "Unknown",

            "text":
                visual_text,

            "content_type":
                "image",

            "ocr_used":
                ocr_used,

            "ocr_languages":
                (
                    page.get(
                        "ocr_languages"
                    )
                    if ocr_used
                    else None
                ),

            "ocr_confidence":
                ocr_confidence,

            "ocr_quality":
                ocr_quality,

            "page_image":
                page_image,

            "images":
                images,

            "tables":
                tables,

            "image_count":
                len(images),

            "table_count":
                len(tables),

            "has_visual_content":
                True
        })


        chunk_id += 1


    # ========================================================
    # FULL VISUAL PAGE CHUNK
    # ========================================================
    #
    # This is useful for scanned pages where OCR may be poor.
    #
    # We preserve the page image reference so a future vision
    # component can inspect the actual page.
    # ========================================================

    if (
        page_image
        and
        not text
    ):

        output.append({

            "chunk_id":
                chunk_id,

            "source":
                source,

            "page":
                page_number,

            "region":
                "visual_page",

            "section":
                "Unknown",

            "subsection":
                "Unknown",

            "text":
                (
                    f"Visual page {page_number} "
                    f"from {source}. "
                    f"Rendered page image available."
                ),

            "content_type":
                "visual_page",

            "ocr_used":
                ocr_used,

            "ocr_languages":
                (
                    page.get(
                        "ocr_languages"
                    )
                    if ocr_used
                    else None
                ),

            "ocr_confidence":
                ocr_confidence,

            "ocr_quality":
                ocr_quality,

            "page_image":
                page_image,

            "images":
                images,

            "tables":
                tables,

            "image_count":
                len(images),

            "table_count":
                len(tables),

            "has_visual_content":
                True
        })


        chunk_id += 1


    return output, chunk_id


# ============================================================
# PROCESS MULTIMODAL DOCUMENT
# ============================================================

def process_multimodal_document(
    document,
    chunk_id
):

    source = document.get(
        "source",
        "Unknown"
    )


    pages = document.get(
        "pages",
        []
    )


    output = []


    print()
    print(
        f"Processing document: "
        f"{source}"
    )


    for page in pages:

        page_chunks, chunk_id = (
            process_multimodal_page(
                page,
                source,
                chunk_id
            )
        )


        output.extend(
            page_chunks
        )


    return output, chunk_id


# ============================================================
# MAIN PROCESSING
# ============================================================

def main():

    print(
        "=========================================="
    )

    print(
        "MULTIMODAL SEMANTIC CHUNKING"
    )

    print(
        "=========================================="
    )


    # --------------------------------------------------------
    # LOAD INPUT
    # --------------------------------------------------------

    data, input_type = (
        load_document_elements()
    )


    all_chunks = []

    chunk_id = 0


    # ========================================================
    # NEW MULTIMODAL FORMAT
    # ========================================================

    if input_type == "multimodal":

        for document in data:

            document_chunks, chunk_id = (
                process_multimodal_document(
                    document,
                    chunk_id
                )
            )


            all_chunks.extend(
                document_chunks
            )


    # ========================================================
    # LEGACY LIST FORMAT
    # ========================================================

    elif input_type == "legacy-list":

        for block in data:

            block_chunks, chunk_id = (
                process_legacy_block(
                    block,
                    chunk_id
                )
            )


            all_chunks.extend(
                block_chunks
            )


    # ========================================================
    # LEGACY FORMAT
    # ========================================================

    elif input_type == "legacy":

        for block in data:

            block_chunks, chunk_id = (
                process_legacy_block(
                    block,
                    chunk_id
                )
            )


            all_chunks.extend(
                block_chunks
            )


    # ========================================================
    # SAVE
    # ========================================================

    os.makedirs(
        os.path.dirname(
            OUTPUT_PATH
        ),
        exist_ok=True
    )


    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_chunks,
            file,
            ensure_ascii=False,
            indent=2
        )


    # ========================================================
    # STATISTICS
    # ========================================================

    region_counts = {}

    content_type_counts = {}

    source_counts = {}

    ocr_chunk_count = 0

    visual_chunk_count = 0

    table_chunk_count = 0

    image_chunk_count = 0


    for chunk in all_chunks:

        # ----------------------------------------------------
        # Region
        # ----------------------------------------------------

        region = chunk.get(
            "region",
            "Unknown"
        )


        region_counts[region] = (
            region_counts.get(
                region,
                0
            )
            + 1
        )


        # ----------------------------------------------------
        # Content type
        # ----------------------------------------------------

        content_type = chunk.get(
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


        # ----------------------------------------------------
        # Source
        # ----------------------------------------------------

        source = chunk.get(
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


        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        if chunk.get(
            "ocr_used",
            False
        ):

            ocr_chunk_count += 1


        # ----------------------------------------------------
        # Visual
        # ----------------------------------------------------

        if chunk.get(
            "has_visual_content",
            False
        ):

            visual_chunk_count += 1


        # ----------------------------------------------------
        # Table
        # ----------------------------------------------------

        if content_type == "table":

            table_chunk_count += 1


        # ----------------------------------------------------
        # Image
        # ----------------------------------------------------

        if content_type == "image":

            image_chunk_count += 1


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()

    print(
        "=========================================="
    )

    print(
        "SEMANTIC CHUNKING COMPLETE"
    )

    print(
        "=========================================="
    )


    print(
        f"Input type      : "
        f"{input_type}"
    )


    print(
        f"Total chunks    : "
        f"{len(all_chunks)}"
    )


    print(
        f"OCR chunks      : "
        f"{ocr_chunk_count}"
    )


    print(
        f"Visual chunks   : "
        f"{visual_chunk_count}"
    )


    print(
        f"Table chunks    : "
        f"{table_chunk_count}"
    )


    print(
        f"Image chunks    : "
        f"{image_chunk_count}"
    )


    print(
        f"Saved to        : "
        f"{OUTPUT_PATH}"
    )


    # ========================================================
    # CHUNKS BY SOURCE
    # ========================================================

    print()

    print(
        "CHUNKS BY SOURCE:"
    )


    for source, count in (
        source_counts.items()
    ):

        print(
            f"  {source}: "
            f"{count}"
        )


    # ========================================================
    # CHUNKS BY REGION
    # ========================================================

    print()

    print(
        "CHUNKS BY REGION:"
    )


    for region, count in (
        region_counts.items()
    ):

        print(
            f"  {region}: "
            f"{count}"
        )


    # ========================================================
    # CHUNKS BY CONTENT TYPE
    # ========================================================

    print()

    print(
        "CHUNKS BY CONTENT TYPE:"
    )


    for content_type, count in (
        content_type_counts.items()
    ):

        print(
            f"  {content_type}: "
            f"{count}"
        )


    # ========================================================
    # FIRST 10 CHUNKS
    # ========================================================

    print()

    print(
        "FIRST 10 CHUNKS:"
    )


    print(
        "=" * 60
    )


    for chunk in all_chunks[:10]:

        print()

        print(
            f"Chunk ID   : "
            f"{chunk.get('chunk_id')}"
        )


        print(
            f"Source     : "
            f"{chunk.get('source')}"
        )


        print(
            f"Page       : "
            f"{chunk.get('page')}"
        )


        print(
            f"Region     : "
            f"{chunk.get('region')}"
        )


        print(
            f"Section    : "
            f"{chunk.get('section')}"
        )


        print(
            f"Subsection : "
            f"{chunk.get('subsection')}"
        )


        print(
            f"Type       : "
            f"{chunk.get('content_type')}"
        )


        print(
            f"OCR        : "
            f"{chunk.get('ocr_used')}"
        )


        print(
            f"Words      : "
            f"{word_count(chunk.get('text', ''))}"
        )


        print(
            f"Text       : "
            f"{chunk.get('text', '')[:500]}"
        )


        print(
            "-" * 60
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()