import fitz


class PDFInspector:
    """IO‑модуль: определяет, является ли PDF сканом или содержит текст."""

    def __init__(self, min_chars: int = 10):
        self.min_chars = min_chars

    def is_scanned(self, pdf_bytes: bytes) -> bool:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            text = page.get_text().strip()
            if len(text) >= self.min_chars:
                return False
        return True