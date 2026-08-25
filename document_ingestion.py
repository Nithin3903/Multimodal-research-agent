import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pymupdf

# Pillow is used for OCR image preprocessing.
try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
except ImportError:
    raise ImportError(
        "\nPillow is required for OCR preprocessing.\n"
        "Install it with:\n\n"
        "pip install pillow\n"
    )


# ============================================================
# CONFIGURATION
# ============================================================

DOCUMENTS_DIR = Path(
    "data/documents"
)

OUTPUT_DIR = Path(
    "data/processed"
)

OUTPUT_PATH = (
    OUTPUT_DIR /
    "document_elements.json"
)

IMAGE_DIR = (
    OUTPUT_DIR /
    "images"
)

PAGE_IMAGE_DIR = (
    OUTPUT_DIR /
    "page_images"
)


# ============================================================
# OCR CONFIGURATION
# ============================================================

OCR_LANGUAGES = "eng+hin+kan"

TESSERACT_EXE = Path(
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

TESSDATA_PATH = Path(
    r"C:\Program Files\Tesseract-OCR\tessdata"
)


# ============================================================
# PROCESSING CONFIGURATION
# ============================================================

# If native PDF extraction contains less than this amount
# of useful text, OCR is attempted.
MIN_TEXT_LENGTH_FOR_OCR = 40


# Normal page images stored for later visual processing.
PAGE_RENDER_DPI = 150


# OCR rendering resolution.
#
# 400 DPI gives Tesseract considerably more information than
# the previous 300 DPI setting, especially for small Kannada
# and Devanagari characters.
OCR_RENDER_DPI = 400


# OCR segmentation modes.
#
# 3  = automatic page segmentation
# 6  = single uniform block
# 11 = sparse text
#
# 12 = sparse text with OSD
OCR_PSM_MODES = [
    3,
    6,
    11
]


# Maximum number of preprocessing variants.
OCR_IMAGE_VARIANTS = [
    "original",
    "autocontrast",
    "contrast",
    "sharpen",
    "threshold"
]


# Minimum OCR text length considered meaningful.
MIN_MEANINGFUL_OCR_CHARS = 50


# ============================================================
# UTF-8 TERMINAL
# ============================================================

try:

    if hasattr(
        sys.stdout,
        "reconfigure"
    ):

        sys.stdout.reconfigure(
            encoding="utf-8",
            errors="replace"
        )


    if hasattr(
        sys.stderr,
        "reconfigure"
    ):

        sys.stderr.reconfigure(
            encoding="utf-8",
            errors="replace"
        )

except Exception:
    pass


# ============================================================
# VALIDATE TESSERACT
# ============================================================

if not TESSERACT_EXE.exists():

    raise FileNotFoundError(
        "\n"
        "Tesseract executable was not found.\n\n"
        "Expected location:\n"
        f"{TESSERACT_EXE}\n\n"
        "Check your Tesseract installation."
    )


# ============================================================
# VALIDATE TESSDATA
# ============================================================

if not TESSDATA_PATH.exists():

    raise FileNotFoundError(
        "\n"
        "Tesseract tessdata directory was not found.\n\n"
        "Expected location:\n"
        f"{TESSDATA_PATH}"
    )


# ============================================================
# VALIDATE LANGUAGE FILES
# ============================================================

required_languages = [
    "eng",
    "hin",
    "kan"
]


missing_languages = []


for language in required_languages:

    trained_data = (
        TESSDATA_PATH /
        f"{language}.traineddata"
    )


    if not trained_data.exists():

        missing_languages.append(
            f"{language}.traineddata"
        )


if missing_languages:

    raise FileNotFoundError(
        "\n"
        "Missing Tesseract language files:\n"
        +
        "\n".join(
            f"  - {language}"
            for language in missing_languages
        )
        +
        "\n\nExpected location:\n"
        f"{TESSDATA_PATH}"
    )


# ============================================================
# CREATE DIRECTORIES
# ============================================================

DOCUMENTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PAGE_IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if not text:

        return ""


    # Remove carriage returns.

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )


    # Join words broken by PDF/OCR line wrapping.

    text = re.sub(
        r"(\w)-\s*\n\s*(\w)",
        r"\1\2",
        text,
        flags=re.UNICODE
    )


    # Normalize excessive whitespace.

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


    return text.strip()


# ============================================================
# NORMAL TEXT EXTRACTION
# ============================================================

def extract_normal_text(page):

    try:

        text = page.get_text(
            "text"
        )


        return clean_text(
            text
        )


    except Exception as error:

        print(
            "      Text extraction failed: "
            f"{error}"
        )


        return ""


# ============================================================
# RENDER PIXMAP
# ============================================================

def render_pixmap(
    page,
    dpi
):

    zoom = (
        dpi /
        72.0
    )


    matrix = pymupdf.Matrix(
        zoom,
        zoom
    )


    return page.get_pixmap(
        matrix=matrix,
        alpha=False
    )


# ============================================================
# PIXMAP -> PIL IMAGE
# ============================================================

def pixmap_to_pil(
    pixmap
):

    image_bytes = pixmap.tobytes(
        "png"
    )


    from io import BytesIO


    image = Image.open(
        BytesIO(
            image_bytes
        )
    )


    return image.convert(
        "RGB"
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(
    image,
    variant
):

    # --------------------------------------------------------
    # Convert to grayscale.
    #
    # This reduces the amount of irrelevant colour information
    # in photographs/scans.
    # --------------------------------------------------------

    gray = ImageOps.grayscale(
        image
    )


    # --------------------------------------------------------
    # Upscale small images.
    #
    # We do not blindly upscale every page.
    # --------------------------------------------------------

    width, height = (
        gray.size
    )


    minimum_width = 2500


    if width < minimum_width:

        scale = (
            minimum_width /
            float(width)
        )


        new_width = int(
            width * scale
        )

        new_height = int(
            height * scale
        )


        gray = gray.resize(
            (
                new_width,
                new_height
            ),
            Image.Resampling.LANCZOS
        )


    # --------------------------------------------------------
    # ORIGINAL
    # --------------------------------------------------------

    if variant == "original":

        return gray


    # --------------------------------------------------------
    # AUTOCONTRAST
    # --------------------------------------------------------

    if variant == "autocontrast":

        return ImageOps.autocontrast(
            gray,
            cutoff=1
        )


    # --------------------------------------------------------
    # CONTRAST
    # --------------------------------------------------------

    if variant == "contrast":

        enhanced = ImageOps.autocontrast(
            gray,
            cutoff=1
        )


        enhancer = (
            ImageEnhance.Contrast(
                enhanced
            )
        )


        return enhancer.enhance(
            1.8
        )


    # --------------------------------------------------------
    # SHARPEN
    # --------------------------------------------------------

    if variant == "sharpen":

        enhanced = ImageOps.autocontrast(
            gray,
            cutoff=1
        )


        enhanced = enhanced.filter(
            ImageFilter.MedianFilter(
                size=3
            )
        )


        enhanced = enhanced.filter(
            ImageFilter.SHARPEN
        )


        enhanced = ImageEnhance.Contrast(
            enhanced
        ).enhance(
            1.5
        )


        return enhanced


    # --------------------------------------------------------
    # THRESHOLD
    # --------------------------------------------------------

    if variant == "threshold":

        enhanced = ImageOps.autocontrast(
            gray,
            cutoff=1
        )


        # Mild contrast enhancement first.

        enhanced = ImageEnhance.Contrast(
            enhanced
        ).enhance(
            1.5
        )


        # Adaptive-ish global threshold.
        #
        # We intentionally avoid an aggressive threshold because
        # Kannada and Devanagari have fine strokes that can
        # disappear when thresholding is too strong.

        threshold = 185


        return enhanced.point(
            lambda pixel:
                255
                if pixel > threshold
                else 0
        )


    return gray


# ============================================================
# SAVE TEMP OCR IMAGE
# ============================================================

def save_temp_image(
    image
):

    temp_file = tempfile.NamedTemporaryFile(
        suffix=".png",
        delete=False
    )


    temp_path = Path(
        temp_file.name
    )


    temp_file.close()


    image.save(
        str(temp_path),
        format="PNG"
    )


    return temp_path


# ============================================================
# OCR QUALITY
# ============================================================

def calculate_text_quality(
    text,
    confidence
):

    if not text:

        return 0.0


    stripped = text.strip()


    if not stripped:

        return 0.0


    character_count = len(
        stripped
    )


    words = stripped.split()


    word_count = len(
        words
    )


    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence_score = (
        confidence /
        100.0
    )


    # --------------------------------------------------------
    # Useful character ratio
    #
    # Unicode-aware.
    #
    # This is important for:
    # Hindi
    # Kannada
    # English
    # --------------------------------------------------------

    useful_characters = sum(

        1

        for character in stripped

        if (
            character.isalnum()
            or
            character.isspace()
            or
            character in
            ".,;:!?()-/%₹$€£"
        )
    )


    useful_ratio = (

        useful_characters /
        character_count

        if character_count

        else 0.0
    )


    # --------------------------------------------------------
    # Length
    # --------------------------------------------------------

    length_score = min(
        character_count /
        500.0,
        1.0
    )


    # --------------------------------------------------------
    # Word count
    # --------------------------------------------------------

    word_score = min(
        word_count /
        80.0,
        1.0
    )


    # --------------------------------------------------------
    # Short-result penalty
    # --------------------------------------------------------

    if character_count < 20:

        short_result_penalty = 0.10

    elif character_count < 50:

        short_result_penalty = 0.45

    elif character_count < 100:

        short_result_penalty = 0.75

    else:

        short_result_penalty = 1.0


    # --------------------------------------------------------
    # Very repetitive / garbage-like output penalty
    # --------------------------------------------------------

    tokens = stripped.split()


    repetition_penalty = 1.0


    if len(tokens) >= 10:

        unique_tokens = len(
            set(tokens)
        )


        unique_ratio = (
            unique_tokens /
            len(tokens)
        )


        if unique_ratio < 0.20:

            repetition_penalty = 0.45

        elif unique_ratio < 0.35:

            repetition_penalty = 0.70


    # --------------------------------------------------------
    # FINAL QUALITY
    # --------------------------------------------------------

    quality = (

        confidence_score * 0.40

        +

        useful_ratio * 0.20

        +

        word_score * 0.15

        +

        length_score * 0.25
    )


    quality *= (
        short_result_penalty
    )


    quality *= (
        repetition_penalty
    )


    return round(
        quality,
        4
    )


# ============================================================
# RUN TESSERACT
# ============================================================

def run_tesseract(
    image_path,
    psm
):

    command = [

        str(
            TESSERACT_EXE
        ),

        str(
            image_path
        ),

        "stdout",

        "--tessdata-dir",

        str(
            TESSDATA_PATH
        ),

        "-l",

        OCR_LANGUAGES,

        "--psm",

        str(
            psm
        ),

        "--oem",

        "1",

        "tsv"
    ]


    try:

        process = subprocess.run(

            command,

            capture_output=True,

            text=True,

            encoding="utf-8",

            errors="replace",

            timeout=120
        )


    except subprocess.TimeoutExpired:

        return {

            "text": "",

            "confidence": 0.0,

            "quality": 0.0,

            "error":
                "Tesseract timed out."
        }


    except Exception as error:

        return {

            "text": "",

            "confidence": 0.0,

            "quality": 0.0,

            "error":
                str(error)
        }


    if process.returncode != 0:

        return {

            "text": "",

            "confidence": 0.0,

            "quality": 0.0,

            "error":
                process.stderr.strip()
        }


    lines = (
        process.stdout.splitlines()
    )


    if len(lines) <= 1:

        return {

            "text": "",

            "confidence": 0.0,

            "quality": 0.0,

            "error": ""
        }


    text_parts = []

    confidences = []


    # ========================================================
    # PARSE TSV
    # ========================================================

    for line in lines[1:]:

        columns = line.split(
            "\t"
        )


        if len(columns) < 12:

            continue


        confidence_text = (
            columns[10]
        )


        word = (
            columns[11]
            .strip()
        )


        try:

            confidence = float(
                confidence_text
            )


        except ValueError:

            continue


        if (
            word
            and
            confidence >= 0
        ):

            text_parts.append(
                word
            )


            confidences.append(
                confidence
            )


    text = clean_text(
        " ".join(
            text_parts
        )
    )


    average_confidence = (

        sum(
            confidences
        )
        /
        len(
            confidences
        )

        if confidences

        else 0.0
    )


    quality = (
        calculate_text_quality(
            text,
            average_confidence
        )
    )


    return {

        "text":
            text,

        "confidence":
            round(
                average_confidence,
                2
            ),

        "quality":
            quality,

        "error":
            ""
    }


# ============================================================
# OCR PAGE
# ============================================================

def extract_ocr_text(
    page
):

    print(
        "      OCR fallback activated..."
    )


    # --------------------------------------------------------
    # Render at high DPI.
    # --------------------------------------------------------

    pixmap = render_pixmap(
        page,
        OCR_RENDER_DPI
    )


    original_image = (
        pixmap_to_pil(
            pixmap
        )
    )


    candidates = []


    # ========================================================
    # TRY MULTIPLE IMAGE VARIANTS
    # ========================================================

    for variant in (
        OCR_IMAGE_VARIANTS
    ):

        print(
            f"      Image variant: "
            f"{variant}"
        )


        try:

            processed_image = (
                preprocess_image(
                    original_image,
                    variant
                )
            )


        except Exception as error:

            print(
                "      Preprocessing failed: "
                f"{error}"
            )

            continue


        temp_path = None


        try:

            temp_path = (
                save_temp_image(
                    processed_image
                )
            )


            # =================================================
            # TRY MULTIPLE PSM MODES
            # =================================================

            for psm in OCR_PSM_MODES:

                print(
                    f"        OCR pass: "
                    f"{variant} / PSM {psm}"
                )


                result = run_tesseract(
                    temp_path,
                    psm
                )


                if result["error"]:

                    print(
                        "        OCR error: "
                        f"{result['error']}"
                    )

                    continue


                if result["text"]:

                    result["variant"] = (
                        variant
                    )

                    result["psm"] = (
                        psm
                    )


                    print(
                        f"        "
                        f"{len(result['text'])} chars | "
                        f"confidence "
                        f"{result['confidence']:.1f}% | "
                        f"quality "
                        f"{result['quality']:.4f}"
                    )


                    candidates.append(
                        result
                    )


        finally:

            if temp_path:

                try:

                    temp_path.unlink(
                        missing_ok=True
                    )

                except Exception:

                    pass


    # ========================================================
    # NO OCR RESULTS
    # ========================================================

    if not candidates:

        return {

            "text": "",

            "confidence": 0.0,

            "quality": 0.0,

            "psm": None,

            "variant": None
        }


    # ========================================================
    # REMOVE TINY RESULTS WHEN A MEANINGFUL RESULT EXISTS
    # ========================================================

    meaningful_candidates = [

        candidate

        for candidate in candidates

        if len(
            candidate["text"].strip()
        ) >= MIN_MEANINGFUL_OCR_CHARS
    ]


    if meaningful_candidates:

        candidates_for_selection = (
            meaningful_candidates
        )

    else:

        candidates_for_selection = (
            candidates
        )


    # ========================================================
    # SELECT BEST OCR
    # ========================================================

    best = max(

        candidates_for_selection,

        key=lambda item:
            item["quality"]
    )


    print()

    print(
        "      BEST OCR RESULT"
    )

    print(
        f"      Variant   : "
        f"{best['variant']}"
    )

    print(
        f"      PSM       : "
        f"{best['psm']}"
    )

    print(
        f"      Characters: "
        f"{len(best['text'])}"
    )

    print(
        f"      Confidence: "
        f"{best['confidence']:.1f}%"
    )

    print(
        f"      Quality   : "
        f"{best['quality']:.4f}"
    )


    return best


# ============================================================
# RENDER NORMAL PAGE IMAGE
# ============================================================

def render_page(
    page,
    source_name,
    page_number
):

    safe_source = re.sub(

        r"[^A-Za-z0-9_.-]",

        "_",

        Path(
            source_name
        ).stem
    )


    filename = (
        f"{safe_source}"
        f"_page_{page_number}.png"
    )


    output_path = (
        PAGE_IMAGE_DIR /
        filename
    )


    try:

        pixmap = render_pixmap(
            page,
            PAGE_RENDER_DPI
        )


        pixmap.save(
            str(
                output_path
            )
        )


        return {

            "path":
                str(
                    output_path
                ),

            "width":
                pixmap.width,

            "height":
                pixmap.height,

            "dpi":
                PAGE_RENDER_DPI
        }


    except Exception as error:

        print(
            "      Page rendering failed: "
            f"{error}"
        )


        return None


# ============================================================
# EXTRACT EMBEDDED IMAGES
# ============================================================

def extract_images(
    document,
    page,
    source_name,
    page_number
):

    images = []


    try:

        page_images = (
            page.get_images(
                full=True
            )
        )


    except Exception:

        page_images = []


    for image_index, image_info in enumerate(
        page_images
    ):

        try:

            xref = image_info[0]


            image_data = (
                document.extract_image(
                    xref
                )
            )


            if not image_data:

                continue


            image_bytes = (
                image_data[
                    "image"
                ]
            )


            extension = (
                image_data.get(
                    "ext",
                    "png"
                )
            )


            safe_source = re.sub(

                r"[^A-Za-z0-9_.-]",

                "_",

                Path(
                    source_name
                ).stem
            )


            image_filename = (

                f"{safe_source}"

                f"_page_{page_number}"

                f"_image_{image_index}"

                f".{extension}"
            )


            image_path = (
                IMAGE_DIR /
                image_filename
            )


            with open(
                image_path,
                "wb"
            ) as image_file:

                image_file.write(
                    image_bytes
                )


            images.append({

                "image_index":
                    image_index,

                "path":
                    str(
                        image_path
                    ),

                "width":
                    image_data.get(
                        "width"
                    ),

                "height":
                    image_data.get(
                        "height"
                    ),

                "extension":
                    extension
            })


        except Exception as error:

            print(
                "      Embedded image "
                "extraction failed: "
                f"{error}"
            )


    return images


# ============================================================
# TABLE EXTRACTION
# ============================================================

def extract_tables(
    page
):

    tables = []


    try:

        table_finder = (
            page.find_tables()
        )


        for table_index, table in enumerate(
            table_finder.tables
        ):

            try:

                extracted = (
                    table.extract()
                )


                if not extracted:

                    continue


                tables.append({

                    "table_index":
                        table_index,

                    "rows":
                        extracted
                })


            except Exception as error:

                print(
                    "      Table extraction "
                    "failed: "
                    f"{error}"
                )


    except Exception:

        pass


    return tables


# ============================================================
# PROCESS PAGE
# ============================================================

def process_page(
    document,
    page,
    source_name,
    page_number
):

    # ========================================================
    # NORMAL TEXT
    # ========================================================

    normal_text = (
        extract_normal_text(
            page
        )
    )


    ocr_used = False

    ocr_confidence = 0.0

    ocr_quality = 0.0

    ocr_psm = None

    ocr_variant = None


    # ========================================================
    # OCR FALLBACK
    # ========================================================

    if len(
        normal_text.strip()
    ) < MIN_TEXT_LENGTH_FOR_OCR:

        ocr_result = (
            extract_ocr_text(
                page
            )
        )


        # ----------------------------------------------------
        # IMPORTANT:
        #
        # We previously accepted OCR only when it was longer.
        #
        # That can reject a shorter but much better OCR result.
        #
        # Now we compare the quality of native text and OCR.
        # ----------------------------------------------------

        if ocr_result["text"]:

            native_quality = (
                calculate_text_quality(
                    normal_text,
                    100.0
                )
            )


            ocr_is_better = (

                ocr_result["quality"]
                >
                native_quality

            )


            if (
                not normal_text
                or
                ocr_is_better
                or
                len(
                    normal_text
                )
                <
                MIN_TEXT_LENGTH_FOR_OCR
            ):

                normal_text = (
                    ocr_result["text"]
                )

                ocr_used = True

                ocr_confidence = (
                    ocr_result[
                        "confidence"
                    ]
                )

                ocr_quality = (
                    ocr_result[
                        "quality"
                    ]
                )

                ocr_psm = (
                    ocr_result[
                        "psm"
                    ]
                )

                ocr_variant = (
                    ocr_result[
                        "variant"
                    ]
                )


    # ========================================================
    # PAGE IMAGE
    # ========================================================

    page_image = (
        render_page(
            page,
            source_name,
            page_number
        )
    )


    # ========================================================
    # EMBEDDED IMAGES
    # ========================================================

    images = (
        extract_images(
            document,
            page,
            source_name,
            page_number
        )
    )


    # ========================================================
    # TABLES
    # ========================================================

    tables = (
        extract_tables(
            page
        )
    )


    # ========================================================
    # VISUAL CONTENT
    # ========================================================

    has_visual_content = bool(

        page_image

        or

        images

        or

        tables
    )


    # ========================================================
    # RESULT
    # ========================================================

    return {

        "source":
            source_name,

        "page":
            page_number,

        "text":
            normal_text,

        "ocr_used":
            ocr_used,

        "ocr_languages":
            OCR_LANGUAGES
            if ocr_used
            else None,

        "ocr_confidence":
            ocr_confidence,

        "ocr_quality":
            ocr_quality,

        "ocr_psm":
            ocr_psm,

        "ocr_variant":
            ocr_variant,

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

        "has_text":
            bool(
                normal_text.strip()
            ),

        "has_visual_content":
            has_visual_content
    }


# ============================================================
# PROCESS PDF
# ============================================================

def process_pdf(
    pdf_path
):

    pdf_path = Path(
        pdf_path
    )


    source_name = (
        pdf_path.name
    )


    print()

    print(
        "=" * 60
    )

    print(
        f"PROCESSING: "
        f"{source_name}"
    )

    print(
        "=" * 60
    )


    try:

        document = pymupdf.open(
            pdf_path
        )


    except Exception as error:

        print(
            "Could not open PDF: "
            f"{error}"
        )


        return None


    pages = []

    total_pages = len(
        document
    )


    for page_number, page in enumerate(
        document,
        start=1
    ):

        print(
            f"Page "
            f"{page_number}/"
            f"{total_pages}..."
        )


        page_data = (
            process_page(
                document,
                page,
                source_name,
                page_number
            )
        )


        pages.append(
            page_data
        )


    document.close()


    return {

        "source":
            source_name,

        "page_count":
            len(pages),

        "pages":
            pages
    }


# ============================================================
# FIND ALL PDFs
# ============================================================

def find_pdfs():

    return sorted(

        [

            path

            for path in
            DOCUMENTS_DIR.iterdir()

            if (
                path.is_file()
                and
                path.suffix.lower()
                == ".pdf"
            )
        ]

    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print(
        "=" * 60
    )

    print(
        "MULTIMODAL PDF INGESTION"
    )

    print(
        "=" * 60
    )


    print(
        f"Tesseract : "
        f"{TESSERACT_EXE}"
    )


    print(
        f"Tessdata  : "
        f"{TESSDATA_PATH}"
    )


    print(
        f"OCR       : "
        f"{OCR_LANGUAGES}"
    )


    print(
        f"OCR DPI   : "
        f"{OCR_RENDER_DPI}"
    )


    print(
        f"Page DPI  : "
        f"{PAGE_RENDER_DPI}"
    )


    print(
        f"OCR modes : "
        f"{OCR_PSM_MODES}"
    )


    print(
        f"Variants  : "
        f"{len(OCR_IMAGE_VARIANTS)}"
    )


    # ========================================================
    # FIND PDFs
    # ========================================================

    pdf_files = find_pdfs()


    if not pdf_files:

        print()

        print(
            "No PDF files found inside:"
        )

        print(
            DOCUMENTS_DIR
        )

        return


    print()

    print(
        f"PDFs found: "
        f"{len(pdf_files)}"
    )


    for pdf in pdf_files:

        print(
            f"- {pdf.name}"
        )


    documents = []


    # ========================================================
    # PROCESS ALL PDFs
    # ========================================================

    for pdf_path in pdf_files:

        result = process_pdf(
            pdf_path
        )


        if result:

            documents.append(
                result
            )


    # ========================================================
    # BUILD OUTPUT
    # ========================================================

    output = {

        "version":
            "3.0",

        "ocr_languages":
            OCR_LANGUAGES,

        "tessdata_path":
            str(
                TESSDATA_PATH
            ),

        "ocr_render_dpi":
            OCR_RENDER_DPI,

        "page_render_dpi":
            PAGE_RENDER_DPI,

        "ocr_psm_modes":
            OCR_PSM_MODES,

        "ocr_image_variants":
            OCR_IMAGE_VARIANTS,

        "document_count":
            len(documents),

        "documents":
            documents
    }


    # ========================================================
    # SAVE JSON
    # ========================================================

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as output_file:

        json.dump(

            output,

            output_file,

            ensure_ascii=False,

            indent=2
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    total_pages = sum(

        document[
            "page_count"
        ]

        for document in
        documents
    )


    total_images = sum(

        page[
            "image_count"
        ]

        for document in
        documents

        for page in
        document[
            "pages"
        ]
    )


    total_tables = sum(

        page[
            "table_count"
        ]

        for document in
        documents

        for page in
        document[
            "pages"
        ]
    )


    ocr_pages = sum(

        1

        for document in
        documents

        for page in
        document[
            "pages"
        ]

        if page[
            "ocr_used"
        ]
    )


    rendered_pages = sum(

        1

        for document in
        documents

        for page in
        document[
            "pages"
        ]

        if page[
            "page_image"
        ]
    )


    confidence_values = [

        page[
            "ocr_confidence"
        ]

        for document in
        documents

        for page in
        document[
            "pages"
        ]

        if (
            page[
                "ocr_used"
            ]
            and
            page[
                "ocr_confidence"
            ] > 0
        )
    ]


    quality_values = [

        page[
            "ocr_quality"
        ]

        for document in
        documents

        for page in
        document[
            "pages"
        ]

        if (
            page[
                "ocr_used"
            ]
            and
            page[
                "ocr_quality"
            ] > 0
        )
    ]


    average_ocr_confidence = (

        sum(
            confidence_values
        )
        /
        len(
            confidence_values
        )

        if confidence_values

        else 0.0
    )


    average_ocr_quality = (

        sum(
            quality_values
        )
        /
        len(
            quality_values
        )

        if quality_values

        else 0.0
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()

    print(
        "=" * 60
    )

    print(
        "INGESTION COMPLETE"
    )

    print(
        "=" * 60
    )


    print(
        f"Documents       : "
        f"{len(documents)}"
    )


    print(
        f"Pages           : "
        f"{total_pages}"
    )


    print(
        f"OCR pages       : "
        f"{ocr_pages}"
    )


    print(
        f"Avg OCR conf.   : "
        f"{average_ocr_confidence:.2f}%"
    )


    print(
        f"Avg OCR quality : "
        f"{average_ocr_quality:.4f}"
    )


    print(
        f"Embedded images : "
        f"{total_images}"
    )


    print(
        f"Tables          : "
        f"{total_tables}"
    )


    print(
        f"Rendered pages  : "
        f"{rendered_pages}"
    )


    print(
        f"Output          : "
        f"{OUTPUT_PATH}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()