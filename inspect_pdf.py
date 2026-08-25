import pymupdf


PDF_PATH = "data/documents/New_research_paper.pdf"


document = pymupdf.open(PDF_PATH)

for page_number, page in enumerate(document, start=1):

    print("\n" + "=" * 80)
    print(f"PAGE {page_number}")
    print("=" * 80)

    blocks = page.get_text("blocks")

    for block_number, block in enumerate(blocks):

        text = block[4].strip()

        if not text:
            continue

        print(
            f"\nBLOCK {block_number}"
        )

        print(
            f"Position: "
            f"x={block[0]:.1f}, "
            f"y={block[1]:.1f}"
        )

        print(text[:500])


document.close()