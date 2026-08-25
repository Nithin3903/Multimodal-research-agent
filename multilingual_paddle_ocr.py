import sys
import json
import re
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from paddleocr import TextDetection, TextRecognition


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = "gpu:0"

# ------------------------------------------------------------
# Paddle models
# ------------------------------------------------------------

DETECTION_MODEL = "PP-OCRv5_server_det"

RECOGNITION_MODELS = {
    "latin": "latin_PP-OCRv5_mobile_rec",
    "hindi": "devanagari_PP-OCRv5_mobile_rec",
    "kannada": "ka_PP-OCRv3_mobile_rec",
}

# ------------------------------------------------------------
# Tesseract
# ------------------------------------------------------------

TESSERACT_EXE = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

TESSDATA_DIR = (
    r"C:\Program Files\Tesseract-OCR\tessdata"
)

TESS_LANGUAGES = "eng+hin+kan"

# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

OUTPUT_DIR = Path(
    "data/processed/paddle_ocr"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ------------------------------------------------------------
# Layout
# ------------------------------------------------------------

LINE_Y_OVERLAP = 0.35
LINE_X_GAP_FACTOR = 3.0

MIN_BOX_WIDTH = 8
MIN_BOX_HEIGHT = 5

# ------------------------------------------------------------
# OCR selection
# ------------------------------------------------------------

MIN_RECOGNITION_CONFIDENCE = 0.20
MIN_FINAL_QUALITY = 0.42


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    if not text:
        return ""

    text = str(text)

    text = text.replace(
        "\r",
        " "
    )

    text = text.replace(
        "\n",
        " "
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# SCRIPT ANALYSIS
# ============================================================

def script_counts(text):

    counts = {
        "latin": 0,
        "hindi": 0,
        "kannada": 0,
        "digits": 0,
        "other": 0,
    }

    for char in text:

        code = ord(char)

        # Latin
        if (
            0x0041 <= code <= 0x005A
            or
            0x0061 <= code <= 0x007A
        ):
            counts["latin"] += 1

        # Devanagari
        elif 0x0900 <= code <= 0x097F:
            counts["hindi"] += 1

        # Kannada
        elif 0x0C80 <= code <= 0x0CFF:
            counts["kannada"] += 1

        elif char.isdigit():
            counts["digits"] += 1

        elif char.isspace():
            pass

        else:
            counts["other"] += 1

    return counts


def dominant_script(text):

    counts = script_counts(text)

    scripts = {
        "latin": counts["latin"],
        "hindi": counts["hindi"],
        "kannada": counts["kannada"],
    }

    best = max(
        scripts,
        key=scripts.get
    )

    if scripts[best] == 0:

        if counts["digits"] > 0:
            return "latin"

        return "unknown"

    return best


def script_character_count(
    text,
    language
):

    counts = script_counts(text)

    if language == "latin":
        return (
            counts["latin"]
            + counts["digits"]
        )

    if language == "hindi":
        return counts["hindi"]

    if language == "kannada":
        return counts["kannada"]

    return 0


def script_purity(
    text,
    language
):

    counts = script_counts(text)

    total = (
        counts["latin"]
        + counts["hindi"]
        + counts["kannada"]
        + counts["digits"]
        + counts["other"]
    )

    if total == 0:
        return 0.0

    matching = script_character_count(
        text,
        language
    )

    return matching / total


# ============================================================
# TEXT QUALITY
# ============================================================

def quality_score(
    text,
    confidence,
    language
):

    text = normalize_text(text)

    if not text:
        return 0.0

    counts = script_counts(text)

    total = (
        counts["latin"]
        + counts["hindi"]
        + counts["kannada"]
        + counts["digits"]
        + counts["other"]
    )

    if total == 0:
        return 0.0

    meaningful = (
        counts["latin"]
        + counts["hindi"]
        + counts["kannada"]
        + counts["digits"]
    )

    meaningful_ratio = (
        meaningful / total
    )

    symbol_ratio = (
        counts["other"] / total
    )

    symbol_score = max(
        0.0,
        1.0 - symbol_ratio
    )

    length_score = min(
        len(text) / 40.0,
        1.0
    )

    purity = script_purity(
        text,
        language
    )

    # Strongly penalize output from a model
    # that does not match the script it is supposed
    # to recognize.
    if purity < 0.20:
        script_score = 0.0

    elif purity < 0.40:
        script_score = 0.20

    elif purity < 0.60:
        script_score = 0.50

    elif purity < 0.80:
        script_score = 0.80

    else:
        script_score = 1.0

    score = (
        0.40 * float(confidence)
        + 0.20 * meaningful_ratio
        + 0.10 * symbol_score
        + 0.10 * length_score
        + 0.20 * script_score
    )

    return float(score)


def valid_paddle_candidate(
    text,
    confidence,
    language
):

    text = normalize_text(text)

    if not text:
        return False

    if confidence < MIN_RECOGNITION_CONFIDENCE:
        return False

    script_chars = script_character_count(
        text,
        language
    )

    purity = script_purity(
        text,
        language
    )

    # Hindi/Kannada must actually contain
    # characters from the expected script.
    if language in (
        "hindi",
        "kannada"
    ):

        if script_chars < 2:
            return False

        if purity < 0.30:
            return False

    # Latin may contain numbers.
    elif language == "latin":

        if script_chars < 1:
            return False

        if purity < 0.20:
            return False

    return True


# ============================================================
# GEOMETRY
# ============================================================

def polygon_to_bbox(
    polygon
):

    points = np.asarray(
        polygon,
        dtype=np.float32
    )

    return (
        int(np.min(points[:, 0])),
        int(np.min(points[:, 1])),
        int(np.max(points[:, 0])),
        int(np.max(points[:, 1]))
    )


def bbox_height(box):

    return max(
        1,
        box["y2"] - box["y1"]
    )


def vertical_overlap(
    a,
    b
):

    top = max(
        a["y1"],
        b["y1"]
    )

    bottom = min(
        a["y2"],
        b["y2"]
    )

    overlap = max(
        0,
        bottom - top
    )

    minimum_height = min(
        bbox_height(a),
        bbox_height(b)
    )

    return (
        overlap
        /
        max(
            minimum_height,
            1
        )
    )


# ============================================================
# DETECTION CLEANUP
# ============================================================

def clean_detection_boxes(
    polygons
):

    boxes = []

    for index, polygon in enumerate(
        polygons
    ):

        x1, y1, x2, y2 = (
            polygon_to_bbox(
                polygon
            )
        )

        width = x2 - x1
        height = y2 - y1

        if width < MIN_BOX_WIDTH:
            continue

        if height < MIN_BOX_HEIGHT:
            continue

        boxes.append({
            "id": index,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        })

    return boxes


# ============================================================
# GROUP DETECTIONS INTO LINES
# ============================================================

def group_into_lines(
    boxes
):

    if not boxes:
        return []

    boxes = sorted(
        boxes,
        key=lambda box: (
            box["y1"],
            box["x1"]
        )
    )

    lines = []

    for box in boxes:

        best_line = None
        best_score = -1.0

        for line in lines:

            line_box = {
                "x1": min(
                    item["x1"]
                    for item in line
                ),
                "y1": min(
                    item["y1"]
                    for item in line
                ),
                "x2": max(
                    item["x2"]
                    for item in line
                ),
                "y2": max(
                    item["y2"]
                    for item in line
                ),
            }

            overlap = vertical_overlap(
                box,
                line_box
            )

            average_height = np.mean([
                bbox_height(item)
                for item in line
            ])

            horizontal_gap = (
                box["x1"]
                -
                line_box["x2"]
            )

            if horizontal_gap < 0:
                horizontal_gap = 0

            maximum_gap = (
                average_height
                *
                LINE_X_GAP_FACTOR
            )

            if (
                overlap >= LINE_Y_OVERLAP
                and
                horizontal_gap <= maximum_gap
            ):

                if overlap > best_score:

                    best_score = overlap
                    best_line = line

        if best_line is not None:

            best_line.append(
                box
            )

        else:

            lines.append(
                [box]
            )

    for line in lines:

        line.sort(
            key=lambda box:
                box["x1"]
        )

    lines.sort(
        key=lambda line:
            min(
                box["y1"]
                for box in line
            )
    )

    return lines


def line_to_bbox(
    line
):

    return {
        "x1": min(
            box["x1"]
            for box in line
        ),
        "y1": min(
            box["y1"]
            for box in line
        ),
        "x2": max(
            box["x2"]
            for box in line
        ),
        "y2": max(
            box["y2"]
            for box in line
        ),
        "source_boxes": [
            box["id"]
            for box in line
        ],
    }


# ============================================================
# OCR ENGINE
# ============================================================

class HybridOCR:

    def __init__(self):

        print("=" * 70)
        print(
            "HYBRID MULTILINGUAL OCR"
        )
        print("=" * 70)

        print(
            f"Paddle device : {DEVICE}"
        )

        print(
            f"Tesseract     : "
            f"{TESSERACT_EXE}"
        )

        # ----------------------------------------------------
        # Verify Tesseract
        # ----------------------------------------------------

        if not Path(
            TESSERACT_EXE
        ).exists():

            raise FileNotFoundError(
                "\nTesseract executable "
                "was not found:\n"
                f"{TESSERACT_EXE}"
            )

        if not Path(
            TESSDATA_DIR
        ).exists():

            raise FileNotFoundError(
                "\nTessdata directory "
                "was not found:\n"
                f"{TESSDATA_DIR}"
            )

        # ----------------------------------------------------
        # Paddle detector
        # ----------------------------------------------------

        print(
            "\nLoading Paddle detection model..."
        )

        self.detector = TextDetection(
            model_name=DETECTION_MODEL,
            device=DEVICE
        )

        # ----------------------------------------------------
        # Paddle recognizers
        # ----------------------------------------------------

        self.recognizers = {}

        for (
            language,
            model_name
        ) in RECOGNITION_MODELS.items():

            print(
                f"Loading Paddle "
                f"{language} model..."
            )

            self.recognizers[
                language
            ] = TextRecognition(
                model_name=model_name,
                device=DEVICE
            )

        print(
            "\nAll OCR models loaded."
        )

    # ========================================================
    # PADDLE DETECTION
    # ========================================================

    def detect(
        self,
        image_path
    ):

        output = self.detector.predict(
            input=str(image_path)
        )

        result = next(
            iter(output)
        )

        data = result.json

        if isinstance(
            data,
            str
        ):

            data = json.loads(
                data
            )

        if "res" in data:

            data = data["res"]

        return data.get(
            "dt_polys",
            []
        )

    # ========================================================
    # CROP
    # ========================================================

    @staticmethod
    def crop_image(
        image,
        box,
        padding=12
    ):

        x1 = max(
            0,
            box["x1"] - padding
        )

        y1 = max(
            0,
            box["y1"] - padding
        )

        x2 = min(
            image.shape[1],
            box["x2"] + padding
        )

        y2 = min(
            image.shape[0],
            box["y2"] + padding
        )

        crop = image[
            y1:y2,
            x1:x2
        ]

        return crop

    # ========================================================
    # PADDLE RECOGNITION
    # ========================================================

    def paddle_recognize(
        self,
        crop,
        language
    ):

        recognizer = (
            self.recognizers[
                language
            ]
        )

        output = recognizer.predict(
            input=crop,
            batch_size=1
        )

        result = next(
            iter(output)
        )

        data = result.json

        if isinstance(
            data,
            str
        ):

            data = json.loads(
                data
            )

        if "res" in data:

            data = data["res"]

        text = normalize_text(
            data.get(
                "rec_text",
                ""
            )
        )

        confidence = float(
            data.get(
                "rec_score",
                0.0
            )
        )

        return (
            text,
            confidence
        )

    # ========================================================
    # TESSERACT OCR
    # ========================================================

    def tesseract_recognize(
        self,
        crop
    ):

        if crop is None:
            return {
                "text": "",
                "confidence": 0.0,
                "language": "tesseract",
                "quality": 0.0,
                "script": "unknown",
            }

        if crop.size == 0:
            return {
                "text": "",
                "confidence": 0.0,
                "language": "tesseract",
                "quality": 0.0,
                "script": "unknown",
            }

        # ----------------------------------------------------
        # Upscale line for Tesseract
        # ----------------------------------------------------

        height, width = (
            crop.shape[:2]
        )

        scale = 2.0

        if height < 80:

            scale = 3.0

        resized = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

        # ----------------------------------------------------
        # Temporary image
        # ----------------------------------------------------

        temp_path = None

        try:

            with tempfile.NamedTemporaryFile(
                suffix=".png",
                delete=False
            ) as temp:

                temp_path = temp.name

            cv2.imwrite(
                temp_path,
                resized
            )

            command = [
                TESSERACT_EXE,
                temp_path,
                "stdout",
                "--tessdata-dir",
                TESSDATA_DIR,
                "-l",
                TESS_LANGUAGES,
                "--oem",
                "1",
                "--psm",
                "7",
                "tsv",
            ]

            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace"
            )

            if process.returncode != 0:

                return {
                    "text": "",
                    "confidence": 0.0,
                    "language": "tesseract",
                    "quality": 0.0,
                    "script": "unknown",
                }

            lines = (
                process.stdout
                .splitlines()
            )

            words = []
            confidences = []

            # ------------------------------------------------
            # TSV format:
            #
            # level page block par line word
            # left top width height conf text
            # ------------------------------------------------

            for line in lines[1:]:

                parts = line.split(
                    "\t"
                )

                if len(parts) < 12:
                    continue

                text = normalize_text(
                    parts[11]
                )

                try:

                    confidence = float(
                        parts[10]
                    )

                except ValueError:

                    continue

                if not text:
                    continue

                if confidence < 0:
                    continue

                words.append(
                    text
                )

                confidences.append(
                    confidence
                )

            text = normalize_text(
                " ".join(words)
            )

            if confidences:

                confidence = (
                    np.mean(
                        confidences
                    )
                    /
                    100.0
                )

            else:

                confidence = 0.0

            # Tesseract is multilingual, so determine
            # the dominant script from its output.
            script = dominant_script(
                text
            )

            # For quality calculation, use the detected
            # script rather than pretending Tesseract
            # was a single-language recognizer.
            quality_language = script

            if quality_language not in (
                "latin",
                "hindi",
                "kannada"
            ):

                quality_language = "latin"

            quality = quality_score(
                text,
                confidence,
                quality_language
            )

            return {
                "text": text,
                "confidence": float(
                    confidence
                ),
                "language": "tesseract",
                "quality": float(
                    quality
                ),
                "script": script,
            }

        finally:

            if (
                temp_path
                and
                Path(temp_path).exists()
            ):

                try:

                    Path(
                        temp_path
                    ).unlink()

                except Exception:
                    pass

    # ========================================================
    # ONE LINE — PADDLE + TESSERACT
    # ========================================================

    def recognize_line(
        self,
        image,
        line_box
    ):

        crop = self.crop_image(
            image,
            line_box
        )

        if crop is None:
            return None

        if crop.size == 0:
            return None

        candidates = []

        # ----------------------------------------------------
        # Paddle candidates
        # ----------------------------------------------------

        for language in (
            "latin",
            "hindi",
            "kannada"
        ):

            try:

                text, confidence = (
                    self.paddle_recognize(
                        crop,
                        language
                    )
                )

                if not valid_paddle_candidate(
                    text,
                    confidence,
                    language
                ):

                    continue

                quality = quality_score(
                    text,
                    confidence,
                    language
                )

                candidates.append({
                    "engine": "paddle",
                    "language": language,
                    "text": text,
                    "confidence": float(
                        confidence
                    ),
                    "quality": float(
                        quality
                    ),
                    "script": dominant_script(
                        text
                    ),
                    "purity": float(
                        script_purity(
                            text,
                            language
                        )
                    ),
                })

            except Exception as exc:

                print(
                    f"Paddle error "
                    f"[{language}]: "
                    f"{exc}"
                )

        # ----------------------------------------------------
        # Tesseract candidate
        # ----------------------------------------------------

        try:

            tess = (
                self.tesseract_recognize(
                    crop
                )
            )

            if tess["text"]:

                candidates.append({
                    "engine": "tesseract",
                    "language": "multilingual",
                    "text": tess["text"],
                    "confidence": tess[
                        "confidence"
                    ],
                    "quality": tess[
                        "quality"
                    ],
                    "script": tess[
                        "script"
                    ],
                    "purity": 1.0,
                })

        except Exception as exc:

            print(
                f"Tesseract error: "
                f"{exc}"
            )

        # ----------------------------------------------------
        # Nothing
        # ----------------------------------------------------

        if not candidates:

            return {
                "engine": "uncertain",
                "language": "unknown",
                "text": "",
                "confidence": 0.0,
                "quality": 0.0,
                "script": "unknown",
                "purity": 0.0,
                "candidates": [],
            }

        # ----------------------------------------------------
        # Choose candidate
        # ----------------------------------------------------

        best = max(
            candidates,
            key=lambda candidate:
                candidate["quality"]
        )

        # ----------------------------------------------------
        # Keep complete candidate audit trail.
        #
        # Important: create copies so there is NO circular
        # reference.
        # ----------------------------------------------------

        candidate_log = []

        for candidate in candidates:

            candidate_log.append({
                "engine": candidate[
                    "engine"
                ],
                "language": candidate[
                    "language"
                ],
                "text": candidate[
                    "text"
                ],
                "confidence": float(
                    candidate[
                        "confidence"
                    ]
                ),
                "quality": float(
                    candidate[
                        "quality"
                    ]
                ),
                "script": candidate[
                    "script"
                ],
                "purity": float(
                    candidate[
                        "purity"
                    ]
                ),
            })

        if (
            best["quality"]
            < MIN_FINAL_QUALITY
        ):

            return {
                "engine": "uncertain",
                "language": "unknown",
                "text": "",
                "confidence": float(
                    best["confidence"]
                ),
                "quality": float(
                    best["quality"]
                ),
                "script": best[
                    "script"
                ],
                "purity": float(
                    best["purity"]
                ),
                "candidates": candidate_log,
            }

        return {
            "engine": best[
                "engine"
            ],
            "language": best[
                "language"
            ],
            "text": best[
                "text"
            ],
            "confidence": float(
                best["confidence"]
            ),
            "quality": float(
                best["quality"]
            ),
            "script": best[
                "script"
            ],
            "purity": float(
                best["purity"]
            ),
            "candidates": candidate_log,
        }

    # ========================================================
    # PROCESS PAGE
    # ========================================================

    def process_page(
        self,
        image_path
    ):

        image_path = Path(
            image_path
        )

        print(
            "\n"
            + "-"
            * 70
        )

        print(
            f"Processing: "
            f"{image_path.name}"
        )

        print(
            "-"
            * 70
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            raise RuntimeError(
                f"Could not read image:\n"
                f"{image_path}"
            )

        # ----------------------------------------------------
        # Detection
        # ----------------------------------------------------

        print(
            "\nDetecting text regions..."
        )

        polygons = self.detect(
            image_path
        )

        print(
            f"Raw detections: "
            f"{len(polygons)}"
        )

        boxes = clean_detection_boxes(
            polygons
        )

        print(
            f"Usable detections: "
            f"{len(boxes)}"
        )

        # ----------------------------------------------------
        # Group into lines
        # ----------------------------------------------------

        lines = group_into_lines(
            boxes
        )

        print(
            f"Merged text lines: "
            f"{len(lines)}"
        )

        # ----------------------------------------------------
        # OCR each line
        # ----------------------------------------------------

        results = []

        for index, line in enumerate(
            lines
        ):

            line_box = line_to_bbox(
                line
            )

            result = self.recognize_line(
                image,
                line_box
            )

            if result is None:
                continue

            result["line_id"] = index
            result["bbox"] = line_box

            results.append(
                result
            )

            if (
                result["engine"]
                == "uncertain"
            ):

                print(
                    f"[{index:03d}] "
                    f"UNCERTAIN"
                )

            else:

                print(
                    f"[{index:03d}] "
                    f"{result['engine']:10s} "
                    f"{result['language']:11s} "
                    f"conf="
                    f"{result['confidence']:.3f} "
                    f"quality="
                    f"{result['quality']:.3f} "
                    f"{result['text']}"
                )

        # ----------------------------------------------------
        # Reconstruct text
        # ----------------------------------------------------

        text_lines = []

        for result in results:

            if result["text"]:

                text_lines.append(
                    result["text"]
                )

        page_text = "\n".join(
            text_lines
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        accepted = [
            result
            for result in results
            if result["engine"]
            != "uncertain"
        ]

        paddle_selected = sum(
            1
            for result in accepted
            if result["engine"]
            == "paddle"
        )

        tesseract_selected = sum(
            1
            for result in accepted
            if result["engine"]
            == "tesseract"
        )

        uncertain = sum(
            1
            for result in results
            if result["engine"]
            == "uncertain"
        )

        if accepted:

            average_confidence = float(
                np.mean([
                    result[
                        "confidence"
                    ]
                    for result in accepted
                ])
            )

            average_quality = float(
                np.mean([
                    result[
                        "quality"
                    ]
                    for result in accepted
                ])
            )

        else:

            average_confidence = 0.0
            average_quality = 0.0

        return {
            "source": str(
                image_path
            ),
            "text": page_text,
            "lines": results,

            "raw_detection_count": len(
                polygons
            ),

            "usable_detection_count": len(
                boxes
            ),

            "merged_line_count": len(
                lines
            ),

            "ocr_line_count": len(
                results
            ),

            "accepted_line_count": len(
                accepted
            ),

            "paddle_selected": (
                paddle_selected
            ),

            "tesseract_selected": (
                tesseract_selected
            ),

            "uncertain_line_count": (
                uncertain
            ),

            "average_confidence": (
                average_confidence
            ),

            "average_quality": (
                average_quality
            ),
        }


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) < 2:

        print(
            "\nUsage:\n"
            "python multilingual_paddle_ocr.py "
            "\"path_to_page.png\"\n"
        )

        sys.exit(1)

    image_path = sys.argv[1]

    engine = HybridOCR()

    result = engine.process_page(
        image_path
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\n"
    )

    print(
        "=" * 70
    )

    print(
        "HYBRID OCR RESULT"
    )

    print(
        "=" * 70
    )

    print(
        f"Raw detections       : "
        f"{result['raw_detection_count']}"
    )

    print(
        f"Usable detections    : "
        f"{result['usable_detection_count']}"
    )

    print(
        f"Merged text lines    : "
        f"{result['merged_line_count']}"
    )

    print(
        f"OCR lines            : "
        f"{result['ocr_line_count']}"
    )

    print(
        f"Accepted lines       : "
        f"{result['accepted_line_count']}"
    )

    print(
        f"Paddle selected      : "
        f"{result['paddle_selected']}"
    )

    print(
        f"Tesseract selected   : "
        f"{result['tesseract_selected']}"
    )

    print(
        f"Uncertain lines      : "
        f"{result['uncertain_line_count']}"
    )

    print(
        f"Average confidence   : "
        f"{result['average_confidence']:.2%}"
    )

    print(
        f"Average quality      : "
        f"{result['average_quality']:.4f}"
    )

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RECONSTRUCTED TEXT"
    )

    print(
        "=" * 70
    )

    print(
        result["text"]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        /
        f"{Path(image_path).stem}"
        f"_hybrid.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(
        "\nSaved: "
        f"{output_path}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()