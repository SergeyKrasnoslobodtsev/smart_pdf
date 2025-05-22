
import unittest


import common_test

from src.PDFExtractor.scan_extractor import ScanExtractor
from src.PDFExtractor.native_extractor import NativeExtractor
from src.PDFExtractor.base_extractor import Page

class Test_TestPDFExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor_scan = ScanExtractor()
        self.extractor_nativ = NativeExtractor()
    
    def test_extract_from_scan(self):
        pdf_bytes = common_test.get_pdf_scan()
        doc = self.extractor_scan.extract(pdf_bytes)
        images = common_test.convert_to_pil(doc.pdf_bytes)

        for image, page in zip(images, doc.pages):
            self._draw_table(image, page)
    
    def test_extract_from_native(self):
        pdf_bytes = common_test.get_pdf_structure()
        doc =  self.extractor_nativ.extract(pdf_bytes)
        images = common_test.convert_to_pil(pdf_bytes)

        for image, page in zip(images, doc.pages):
            self._draw_table(image, page)


    def _draw_table(self, image, page: Page):
        from PIL import ImageDraw
        
        draw = ImageDraw.Draw(image)
        for id_t, table in enumerate(page.tables):
            x = table.bbox.x1
            y = table.bbox.y1
            w = table.bbox.x2
            h = table.bbox.y2
            
            draw = common_test.draw_label(draw, f'Table: {id_t}', (x, y))
            draw.rectangle([(x, y), (w, h)], outline='blue', width=3) 
            for cell in table.cells:
                cx = cell.bbox.x1
                cy = cell.bbox.y1
                cw = cell.bbox.x2
                ch = cell.bbox.y2
                r = cell.row
                c = cell.col
                draw = common_test.draw_label(draw, F'R{r}:C{c} - {cell.text}', (cx, cy + 35))
                draw.rectangle([(cx, cy), (cw, ch)], outline='green', width=2)
                print(f'R{r}:C{c} - {cell.text}')
        for id_p, parag in enumerate(page.paragraphs):
            x = parag.bbox.x1
            y = parag.bbox.y1
            w = parag.bbox.x2
            h = parag.bbox.y2
            draw = common_test.draw_label(draw, F'{parag.type.name} - {parag.text}', (x, y + 35))
            draw.rectangle([(x, y), (w, h)], outline='red', width=3) 
        image.show()

if __name__ == '__main__':
    unittest.main()
