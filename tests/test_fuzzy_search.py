
import unittest
import common_test

from src.PDFExtractor.scan_extractor import ScanExtractor
from src.fuzzy_search import update_cell_text_in_document


class Test_TestFuzzySearch(unittest.TestCase):

    def setUp(self):
        self.extractor_scan = ScanExtractor()

    def test_update_text_to_cell(self):

        pdf_bytes = common_test.get_pdf_scan()
        doc = self.extractor_scan.extract(pdf_bytes)
        new_doc = update_cell_text_in_document(doc, 0, 0, 8, 2, "10 000 000,00")
        new_doc = update_cell_text_in_document(new_doc, 0, 0, 4, 1, "10 000 000,00")
        new_doc = update_cell_text_in_document(new_doc, 0, 0, 4, 2, "10 000 000,00")
        new_doc = update_cell_text_in_document(new_doc, 0, 0, 2, 4, "10 000 000,00")
        new_doc = update_cell_text_in_document(new_doc, 0, 0, 5, 3, "10 000 000,00")
        with open("updated_sample.pdf", "wb") as file:
            file.write(new_doc.pdf_bytes)


if __name__ == '__main__':
    unittest.main()
