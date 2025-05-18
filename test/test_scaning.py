import unittest
from src.extractors.pdf_inspector import PDFInspector
import common_test 

class Test_TestInspector(unittest.TestCase):


    def test_scan_pdf(self) -> None:
        """PDF‑скан должен быть определён как скан (True)."""
        inspector = PDFInspector()
        pdf_bytes = common_test.get_pdf_scan()
        assert inspector.is_scanned(pdf_bytes) is True, "Скан не распознан как скан"

    def test_structured_pdf(self) -> None:
        """PDF с текстовым слоем должен быть определён как не‑скан (False)."""
        inspector = PDFInspector()
        pdf_bytes = common_test.get_pdf_structure()
        assert inspector.is_scanned(pdf_bytes) is False, "Структурный PDF ошибочно принят за скан"
