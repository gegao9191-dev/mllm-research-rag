from pathlib import Path

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf


PAPER_DIR = Path("data/papers")


def main():
    pdf_files = list(PAPER_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in data/papers")
        return

    for pdf_path in pdf_files:
        print("=" * 80)
        print(f"PDF: {pdf_path.name}")

        try:
            doc = pymupdf.open(pdf_path)
            print(f"Pages: {len(doc)}")

            preview_text = []
            for page_id, page in enumerate(doc[:3], start=1):
                text = page.get_text("text", sort=True)
                preview_text.append(f"\n--- Page {page_id} ---\n{text[:1500]}")

            full_preview = "\n".join(preview_text)
            print(full_preview[:3000])

        except Exception as e:
            print(f"Failed to read {pdf_path.name}: {e}")


if __name__ == "__main__":
    main()