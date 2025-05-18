from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
import fitz  # PyMuPDF
import numpy as np

from .ocr_engine import OcrEngine

from .pdf_inspector import PDFInspector
from .table_extractor import TableExtractor, Table
from .paragraph_extractor import ParagraphExtractor, Paragraph


@dataclass
class DocumentContext:
    pdf_bytes: Optional[bytes] = None
    gray_images: List[np.ndarray] = field(default_factory=list)
    tables: List[Table] = field(default_factory=list)
    paragraphs: List[Paragraph] = field(default_factory=list)
    is_scanned: Optional[bool] = None


class PDFParseStrategy(ABC):
    @abstractmethod
    def render(self, ctx: DocumentContext) -> None:
        """Заполняет ctx.gray_images для последующей обработки."""
        ...

class ScannedPDFStrategy(PDFParseStrategy):
    def render(self, ctx: DocumentContext) -> None:
        doc = fitz.open(stream=ctx.pdf_bytes, filetype="pdf")
        for page in doc:
            pix = page.get_pixmap(dpi=300, colorspace=fitz.csGRAY)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
            ctx.gray_images.append(arr)

class DigitalPDFStrategy(PDFParseStrategy):
    def __init__(self, table_ext: TableExtractor, para_ext: ParagraphExtractor):
        self.table_ext = table_ext
        self.para_ext = para_ext

    def render(self, ctx: DocumentContext) -> None:
        # Прямое извлечение встроенных таблиц и текста
        ctx.tables = self.table_ext.extract_from_pdf()
        ctx.paragraphs = self.para_ext.extract_from_pdf()


class Handler(ABC):
    def __init__(self, nxt: Optional['Handler'] = None):
        self._next = nxt

    def set_next(self, nxt: 'Handler') -> 'Handler':
        self._next = nxt
        return nxt

    def handle(self, ctx: DocumentContext) -> None:
        self._handle(ctx)
        if self._next:
            self._next.handle(ctx)

    @abstractmethod
    def _handle(self, ctx: DocumentContext) -> None:
        ...

class InspectionHandler(Handler):
    def __init__(self, inspector: PDFInspector, nxt=None):
        super().__init__(nxt)
        self.inspector = inspector

    def _handle(self, ctx: DocumentContext) -> None:
        ctx.is_scanned = self.inspector.is_scanned(ctx.pdf_bytes)

class TableHandler(Handler):
    def __init__(self, table_ext: TableExtractor, nxt=None):
        super().__init__(nxt)
        self.table_ext = table_ext

    def _handle(self, ctx: DocumentContext) -> None:
        if ctx.is_scanned and ctx.gray_images:
            for page, gray in enumerate(ctx.gray_images, start=1):
                ctx.tables.extend(self.table_ext.extract_from_image(gray, page))

class ParagraphHandler(Handler):
    def __init__(self, para_ext: ParagraphExtractor, nxt=None):
        super().__init__(nxt)
        self.para_ext = para_ext

    def _handle(self, ctx: DocumentContext) -> None:
        if ctx.is_scanned and ctx.gray_images:
            for page, gray in enumerate(ctx.gray_images, start=1):
                ctx.paragraphs.extend(self.para_ext.extract_from_image(gray, ctx.tables, page))


class DocumentProcessor:
    def __init__(self):
        self.inspector = PDFInspector(min_chars=10)
        self.table_ext = TableExtractor(ocr=OcrEngine.TESSERACT)
        self.para_ext = ParagraphExtractor(ocr=OcrEngine.TESSERACT)
        self.scanned_strat = ScannedPDFStrategy()
        self.digital_strat = DigitalPDFStrategy(self.table_ext, self.para_ext)
        # собираем цепочку: инспекция → извлечение таблиц → абзацев
        self.chain = InspectionHandler(self.inspector)
        self.chain \
            .set_next(TableHandler(self.table_ext)) \
            .set_next(ParagraphHandler(self.para_ext))

    def process(self, pdf_bytes: bytes) -> DocumentContext:
        ctx = DocumentContext(pdf_bytes=pdf_bytes)
        # 1) определяем, скан ли это
        ctx.is_scanned = self.inspector.is_scanned(pdf_bytes)
        # 2) выбираем стратегию
        if ctx.is_scanned:
            self.scanned_strat.render(ctx)
            # 3) запускаем цепочку обработки изображений
            self.chain.handle(ctx)
        else:
            self.digital_strat.render(ctx)
        return ctx
