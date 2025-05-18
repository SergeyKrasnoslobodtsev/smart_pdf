
import unittest
from unittest.mock import patch
import common_test

from src.extractors.document_processor import DocumentContext, DocumentProcessor
from src.extractors.table_extractor import TableExtractor, Table
from src.extractors.paragraph_extractor import ParagraphExtractor, Paragraph, ParagraphType
from src.extractors.extractor_base import BBox

class TestDocumentProcessor(unittest.TestCase):
    def setUp(self):
        self.proc = DocumentProcessor()

    def test_scan_document_extractor(self):
        pdf_bytes = common_test.get_pdf_scan()
        ctx = self.proc.process(pdf_bytes)
        images = common_test.convert_to_pil(ctx.pdf_bytes)
        for image in images:
            self._draw_table(image, ctx)
        # Должен определить как скан
        self.assertTrue(ctx.is_scanned, "Document should be detected as scanned")
        # Таблицы и параграфы должны быть массивами с данными
        self.assertIsInstance(ctx.tables, list)
        self.assertGreater(len(ctx.tables), 0, "Expected at least one table in scanned document")
        self.assertIsInstance(ctx.paragraphs, list)
        self.assertGreater(len(ctx.paragraphs), 0, "Expected at least one paragraph in scanned document")

        # Должны быть HEADER и FOOTER среди типов параграфов
        types = {p.type for p in ctx.paragraphs}
        self.assertIn(ParagraphType.HEADER, types, "Scanned document should contain HEADER paragraph")
        self.assertIn(ParagraphType.FOOTER, types, "Scanned document should contain FOOTER paragraph")

    @patch.object(TableExtractor, 'extract_from_pdf')
    @patch.object(ParagraphExtractor, 'extract_from_pdf')
    def test_digital_document_extractor(self, mock_para_pdf, mock_table_pdf):
        # Подготавливаем фиктивные значения для цифрового PDF
        dummy_table = Table(bbox=BBox(0, 0, 100, 100), cells=[])
        dummy_para = Paragraph(bbox=BBox(0, 0, 50, 20), type=ParagraphType.NONE, text="dummy text")
        mock_table_pdf.return_value = [dummy_table]
        mock_para_pdf.return_value = [dummy_para]

        pdf_bytes = common_test.get_pdf_structure()
        ctx = self.proc.process(pdf_bytes)

        # Должен определить как цифровой
        self.assertFalse(ctx.is_scanned, "Document should be detected as digital")
        # Методы extract_from_pdf должны быть вызваны
        mock_table_pdf.assert_called_once()
        mock_para_pdf.assert_called_once()
        # Контекст должен содержать наши фиктивные данные
        self.assertEqual(ctx.tables, [dummy_table])
        self.assertEqual(ctx.paragraphs, [dummy_para])

    
    def _draw_table(self, image, ctx: DocumentContext) -> None:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(image)
        for id_t, table in enumerate(ctx.tables):
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

        for id_p, pargraph in enumerate(ctx.paragraphs):
            x = pargraph.bbox.x1
            y = pargraph.bbox.y1
            w = pargraph.bbox.x2
            h = pargraph.bbox.y2
            draw = common_test.draw_label(draw, f'{pargraph.type.name}: {id_p} - {pargraph.text}', (x, y))
            draw.rectangle([(x, y), (w, h)], outline='blue', width=3) 

        image.show()

if __name__ == '__main__':
    unittest.main()