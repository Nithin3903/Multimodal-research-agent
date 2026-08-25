import fitz


PDF_PATH = "data/documents/New_research_paper.pdf"


def extract_text(pdf_path):
    document = fitz.open(pdf_path)

    full_text = ""

    for page_number, page in enumerate(document, start=1):
        text = page.get_text()

        full_text += f"\n\n--- PAGE {page_number} ---\n\n"
        full_text += text

    document.close()

    return full_text


text = extract_text(PDF_PATH)

print(text)