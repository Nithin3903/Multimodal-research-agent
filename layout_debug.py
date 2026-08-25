import sys
sys.stdout.reconfigure(encoding="utf-8")
import pymupdf


PDF_PATH = "data/documents/New_research_paper.pdf"

document = pymupdf.open(PDF_PATH)

page = document[0]

print("PAGE SIZE")
print("Width :", page.rect.width)
print("Height:", page.rect.height)

print("\nBLOCKS")
print("=" * 100)

blocks = page.get_text("blocks", sort=False)

for i, block in enumerate(blocks):

    x0, y0, x1, y1, text = block[:5]

    print(f"\nBLOCK {i}")
    print(f"x0={x0:.1f}, y0={y0:.1f}, x1={x1:.1f}, y1={y1:.1f}")
    print(f"width={x1-x0:.1f}, height={y1-y0:.1f}")
    print("TEXT:")
    print(text.replace("\n", " ")[:300])

document.close()