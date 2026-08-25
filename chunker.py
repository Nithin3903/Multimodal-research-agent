import fitz


PDF_PATH = "data/documents/New_research_paper.pdf"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def extract_pages(pdf_path):
    document = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text().strip()

        if text:
            pages.append({
                "page": page_number,
                "text": text
            })

    document.close()

    return pages


def create_chunks(pages):
    chunks = []
    chunk_id = 0

    for page_data in pages:

        text = page_data["text"]
        page_number = page_data["page"]

        start = 0

        while start < len(text):

            end = start + CHUNK_SIZE
            chunk_text = text[start:end]

            chunks.append({
                "chunk_id": chunk_id,
                "page": page_number,
                "text": chunk_text
            })

            chunk_id += 1

            start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


pages = extract_pages(PDF_PATH)
chunks = create_chunks(pages)


print(f"Pages extracted: {len(pages)}")
print(f"Chunks created: {len(chunks)}")

print("\n--- FIRST CHUNK ---\n")
print(chunks[0]["text"])

print("\n--- METADATA ---\n")
print("Chunk ID:", chunks[0]["chunk_id"])
print("Page:", chunks[0]["page"])