
from typing import List
import unittest


import common_test

from src.initialize_api import logger_configure
from src.PDFExtractor.scan_extractor import ScanExtractor
from src.PDFExtractor.native_extractor import NativeExtractor
from src.PDFExtractor.base_extractor import Page, Table

class Test_TestPDFExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor_scan = ScanExtractor()
        self.extractor_nativ = NativeExtractor()
        logger_configure()
    
    def test_extract_from_scan(self):
        pdf_bytes = common_test.get_pdf_scan()
        doc = self.extractor_scan.extract(pdf_bytes)
        images = common_test.convert_to_pil(doc.pdf_bytes)
        tables = doc.get_tables()
        for image, page in zip(images, doc.pages):
            self._draw_table(image, page, tables)
    
    def test_extract_from_native(self):
        pdf_bytes = common_test.get_pdf_structure()
        doc =  self.extractor_nativ.extract(pdf_bytes)
        images = common_test.convert_to_pil(pdf_bytes)
        tables = doc.get_tables()
        for image, page in zip(images, doc.pages):
            self._draw_table(image, page, tables)


    def _draw_table(self, image, page: Page, tables: List[Table]):
        from PIL import ImageDraw
        
        draw = ImageDraw.Draw(image)
        current_page_idx = page.num_page 

        for id_t, logical_table_item in enumerate(tables):
            if logical_table_item.start_page_num == current_page_idx:
                x = logical_table_item.bbox.x1
                y = logical_table_item.bbox.y1
                w = logical_table_item.bbox.x2
                h = logical_table_item.bbox.y2
            
                draw = common_test.draw_label(draw, f'Table: {id_t} (Pg {logical_table_item.start_page_num})', (x, y))
                draw.rectangle([(x, y), (w, h)], outline='blue', width=3) 

            for cell in logical_table_item.cells:
                if cell.original_page_num == current_page_idx:
                    cx = cell.bbox.x1
                    cy = cell.bbox.y1
                    cw = cell.bbox.x2
                    ch = cell.bbox.y2
                    r = cell.row 
                    c = cell.col 

                    draw = common_test.draw_label(draw, F'R{r}:C{c}', (cx, cy + 35))
                    draw.rectangle([(cx, cy), (cw, ch)], outline='green', width=2)
                    print(f'Page {current_page_idx} - LogicalTable {id_t} - Cell R{r}:C{c} - {cell.text}')
                    
                    for line_blob in cell.blobs:
                        lx = line_blob.x1
                        ly = line_blob.y1
                        lw = line_blob.x2
                        lh = line_blob.y2
                        draw.rectangle([(lx, ly), (lw, lh)], outline='red', width=1)
        for id_p, parag in enumerate(page.paragraphs):
            x = parag.bbox.x1
            y = parag.bbox.y1
            w = parag.bbox.x2
            h = parag.bbox.y2
            draw = common_test.draw_label(draw, F'{parag.type.name}', (x, y + 35))
            draw.rectangle([(x, y), (w, h)], outline='red', width=3) 
        image.show()

if __name__ == '__main__':
    unittest.main()
