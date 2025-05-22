from abc import ABC
from dataclasses import dataclass, field
import enum
from typing import List, Tuple, Union

import pymupdf


@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def coords(self) -> Tuple[int, int, int, int]:
        """Вернуть все координаты как кортеж."""
        return self.x1, self.y1, self.x2, self.y2

    @coords.setter
    def coords(self, vals: Tuple[int, int, int, int]):
        """Установить сразу все координаты из кортежа."""
        self.x1, self.y1, self.x2, self.y2 = vals

    @property
    def width(self) -> int:
        """Ширина bbox."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Высота bbox."""
        return self.y2 - self.y1

    def padding(self, pix: int = 5) -> "BBox":
        return BBox(
            x1=self.x1 - pix,
            y1=self.y1 - pix,
            x2=self.x2 + pix,
            y2=self.y2 + pix,
        )
    
    def contains(self, other: 'BBox') -> bool:
        return (self.x1 <= other.x1 and
                self.y1 <= other.y1 and
                self.x2 >= other.x2 and
                self.y2 >= other.y2)

    @classmethod
    def from_rect(
        cls,
        rect: Union[
            "pymupdf.Rect",      # если у вас есть real Rect
            Tuple[float, float, float, float]  # или кортеж (x0,y0,x1,y1)
        ],
        sx: float = 1.0,
        sy: float = 1.0
    ) -> "BBox":
        """Создать BBox из pymupdf.Rect или из кортежа (x0, y0, x1, y1) с учётом масштабов."""
        # разбираем вход
        if hasattr(rect, "x0"):
            x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
        else:
            x0, y0, x1, y1 = rect  # ожидаем кортеж из 4-х чисел

        return cls(
            x1=int(x0 * sx),
            y1=int(y0 * sy),
            x2=int(x1 * sx),
            y2=int(y1 * sy),
        )



@dataclass
class Cell:
    bbox: BBox
    row: int
    col: int
    text: str = None

@dataclass
class Table:
    bbox: BBox
    cells: List[Cell] = field(default_factory=list)

class ParagraphType(enum.Enum):
    HEADER = 0
    FOOTER = 1
    NONE = 2

@dataclass
class Paragraph:
    bbox: BBox
    type: ParagraphType = ParagraphType.NONE
    text: str = None
    

@dataclass
class Page:
    tables: List[Table] = field(default_factory=list)
    paragraphs: List[Paragraph] = field(default_factory=list)
    num_page: int = 0

@dataclass
class Document:
    pdf_bytes: bytes = None
    pages: List[Page] = field(default_factory=list)
    page_count: int = 0

class BaseExtractor(ABC):

    def extract(self, pdf_bytes: bytes) -> Document:
        import pymupdf
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        print(f'Is form scan: {doc.is_form_pdf}')
        page_count = len(doc)
        pages: List[Page] = []
        for i in range(page_count):
            page = doc[i]
            paragraphs, tables = self._process(page)
            
            pages.append(
                Page(
                    tables=tables,
                    paragraphs=paragraphs,
                    num_page=i
                )
            )

        return Document(
                    pdf_bytes=pdf_bytes,
                    pages=pages,
                    page_count=len(pages)
        )
    
    def _process(self, page)-> Tuple[List[Paragraph], List[Table]]:
        ...

