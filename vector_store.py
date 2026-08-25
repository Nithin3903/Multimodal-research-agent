import json
import os

import faiss
import numpy as np
import pymupdf
import torch
from sentence_transformers import SentenceTransformer


# -----------------------------
# Configuration
# -----------------------------

MODEL_NAME = "BAAI/bge-small-en-v1.5"

DOCUMENTS_DIR = "data/documents"
VECTORSTORE_DIR = "data/vectorstore"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# -----------------------------
# Find PDF
# -----------------------------

pdf_files = [
    file for file in os.listdir(DOCUMENTS_DIR)
    if file.lower().endswith(".pdf")
]

if not pdf_files:
    raise FileNotFoundError(
        "No PDF found inside data/documents"
    )

pdf_filename = pdf_files[0]
pdf_path = os.path.join(DOCUMENTS_DIR, pdf_filename)

print(f"Using PDF: {pdf_filename}")


# -----------------------------
# Extract PDF pages
# -----------------------------

document = pymupdf.open(pdf_path)

pages = []

for page_number, page in enumerate(document, start=1):

    text = page.get_text().strip()

    if text:
        pages.append({
            "page": page_number,
            "text": text
        })

document.close()

print(f"Pages extracted: {len(pages)}")


# -----------------------------
# Create chunks
# -----------------------------

chunks = []
chunk_id = 0

for page_data in pages:

    text = page_data["text"]
    page_number = page_data["page"]

    start = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk_text = text[start:end].strip()

        if chunk_text:

            chunks.append({
                "chunk_id": chunk_id,
                "page": page_number,
                "source": pdf_filename,
                "text": chunk_text
            })

            chunk_id += 1

        start += CHUNK_SIZE - CHUNK_OVERLAP


print(f"Chunks created: {len(chunks)}")


# -----------------------------
# Load embedding model
# -----------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")

if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


model = SentenceTransformer(
    MODEL_NAME,
    device=device
)


# -----------------------------
# Generate embeddings
# -----------------------------

texts = [
    chunk["text"]
    for chunk in chunks
]

embeddings = model.encode(
    texts,
    batch_size=16,
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = np.asarray(
    embeddings,
    dtype="float32"
)


print(f"Embedding shape: {embeddings.shape}")


# -----------------------------
# Create FAISS index
# -----------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)


print(f"FAISS vectors stored: {index.ntotal}")


# -----------------------------
# Save vector store
# -----------------------------

os.makedirs(
    VECTORSTORE_DIR,
    exist_ok=True
)

index_path = os.path.join(
    VECTORSTORE_DIR,
    "index.faiss"
)

metadata_path = os.path.join(
    VECTORSTORE_DIR,
    "metadata.json"
)


faiss.write_index(
    index,
    index_path
)


with open(
    metadata_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        chunks,
        file,
        ensure_ascii=False,
        indent=2
    )


print("\nVector store created successfully!")

print(f"Index: {index_path}")
print(f"Metadata: {metadata_path}")