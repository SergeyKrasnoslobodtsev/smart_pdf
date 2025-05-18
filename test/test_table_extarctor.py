from typing import List

import cv2
import numpy as np
import common_test

import unittest

from src.extractors.table_extractor import Table, TableExtractor

class Test_TestExtractor(unittest.TestCase):

    def test_extract_table(self):
   
        images = common_test.load_document_pdf()
        detector = TableExtractor()
        for image in images:
            im_arr = np.array(image)
            gray = cv2.cvtColor(im_arr, cv2.COLOR_RGB2GRAY)
            tables = detector.extract_from_image(gray)
            self._draw_table(image, tables)

    def _draw_table(self, image, tables: List[Table]):
        from PIL import ImageDraw
        draw = ImageDraw.Draw(image)
        for id_t, table in enumerate(tables):
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
        image.show()
        


