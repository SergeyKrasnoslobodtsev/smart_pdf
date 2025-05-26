from abc import ABC
from dataclasses import dataclass, field
import enum
from typing import List, Optional, Tuple, Union
from PIL import Image
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

class InsertionPosition(enum.Enum):
    TOP = "top"  # Сверху
    BOTTOM = "bottom" # Снизу
    LEFT = "left"
    RIGHT = "right"

@dataclass
class Cell:
    bbox: BBox
    row: int
    col: int
    text: str = None
    blobs: List[BBox] = field(default_factory=list)

    @property
    def has_text(self) -> bool:
        """Проверяет, содержит ли ячейка текст (на основе атрибута text или blobs)."""
        if self.text and self.text.strip():
            return True
        return len(self.blobs) > 0
    
    @property
    def free_space_ratio(self) -> float:
        """
        Оценивает долю свободного пространства внутри bbox ячейки, не занятого blobs.
        Возвращает 1.0, если bbox ячейки имеет нулевую площадь.
        """
        cell_area = self.bbox.width * self.bbox.height
        if cell_area == 0:
            return 1.0  # Если площадь ячейки 0, считаем ее полностью свободной или полностью занятой в зависимости от контекста. 1.0 означает нет места для нового.
                        # Или можно вернуть 0.0 если blobs тоже нет, что означает "нет контента, нет места"

        occupied_area_by_blobs = 0
        for blob in self.blobs:
            occupied_area_by_blobs += blob.width * blob.height
        
        # Ограничиваем occupied_area, чтобы она не превышала cell_area
        occupied_area_by_blobs = min(occupied_area_by_blobs, float(cell_area))

        free_area = cell_area - occupied_area_by_blobs
        return free_area / cell_area

    def get_insertion_area(
        self,
        position: InsertionPosition,
        min_height: int = 10, # Минимальная высота для области вставки
        padding: int = 2      # Отступ от краев основного bbox ячейки
    ) -> Optional[BBox]:
        """
        Вычисляет прямоугольную область внутри ячейки, подходящую для вставки нового текста.
        Область будет занимать ширину ячейки (за вычетом отступов).

        Args:
            position: Член перечисления InsertionPosition (TOP или BOTTOM).
            min_height: Минимально необходимая высота для области вставки.
            padding: Отступ от краев основного bbox ячейки.

        Returns:
            Объект BBox, представляющий доступную область, или None, если подходящая область не найдена.
        """
        if self.bbox.width <= 2 * padding or self.bbox.height <= 2 * padding:
            return None # Ячейка слишком мала для отступов

        # Эффективные границы ячейки после применения отступов
        eff_cell_x1 = self.bbox.x1 + padding
        eff_cell_y1 = self.bbox.y1 + padding
        eff_cell_x2 = self.bbox.x2 - padding
        eff_cell_y2 = self.bbox.y2 - padding

        if eff_cell_x1 >= eff_cell_x2 or eff_cell_y1 >= eff_cell_y2:
            return None # Нет места после применения отступов

        insert_x1 = eff_cell_x1
        insert_x2 = eff_cell_x2
        insert_y1 = -1
        insert_y2 = -1

        # Фильтруем blobs, которые находятся по горизонтали в пределах эффективной ширины ячейки
        relevant_blobs = [
            b for b in self.blobs if b.x1 < eff_cell_x2 and b.x2 > eff_cell_x1
        ]

        if position == InsertionPosition.TOP:
            insert_y1 = eff_cell_y1 
            limit_y = eff_cell_y2 # По умолчанию, предел - это нижняя граница ячейки с отступом
            if relevant_blobs:
                # Ищем самый верхний край существующих blobs, чтобы вставить текст над ними
                blob_tops = [b.y1 for b in relevant_blobs if b.y1 >= eff_cell_y1] 
                if blob_tops:
                    limit_y = min(min(blob_tops) - 1, eff_cell_y2) # -1 для небольшого зазора
            insert_y2 = limit_y

        elif position == InsertionPosition.BOTTOM:
            insert_y2 = eff_cell_y2
            limit_y = eff_cell_y1 # По умолчанию, предел - это верхняя граница ячейки с отступом
            if relevant_blobs:
                # Ищем самый нижний край существующих blobs, чтобы вставить текст под ними
                blob_bottoms = [b.y2 for b in relevant_blobs if b.y2 <= eff_cell_y2]
                if blob_bottoms:
                    limit_y = max(max(blob_bottoms) + 1, eff_cell_y1) # +1 для небольшого зазора
            insert_y1 = limit_y
        else:
            # Это можно расширить для других позиций, таких как LEFT, RIGHT и т.д.
            raise NotImplementedError(f"Позиция для вставки {position} еще не поддерживается.")

        # Проверяем рассчитанную область
        if insert_y1 != -1 and insert_y2 != -1 and \
        insert_x1 < insert_x2 and insert_y1 < insert_y2 and \
        (insert_y2 - insert_y1) >= min_height:
            return BBox(x1=insert_x1, y1=insert_y1, x2=insert_x2, y2=insert_y2)

        return None

@dataclass
class Table:
    bbox: BBox
    cells: List[Cell] = field(default_factory=list)

    @property
    def average_blob_height(self) -> float:
        """
        Рассчитывает среднюю высоту всех blobs во всех ячейках таблицы.
        Возвращает 0.0, если blobs отсутствуют в таблице.
        """
        all_blobs_heights = []
        for cell in self.cells:
            for blob in cell.blobs:
                all_blobs_heights.append(blob.height)
        
        if not all_blobs_heights:
            return 0.0
        return sum(all_blobs_heights) / len(all_blobs_heights)

class ParagraphType(enum.Enum):
    HEADER = 0
    FOOTER = 1
    NONE = 2

@dataclass
class Paragraph:
    bbox: BBox
    type: ParagraphType = ParagraphType.NONE
    text: str = None
    blobs: List[BBox] = field(default_factory=list)
    

@dataclass
class Page:
    image: Image.Image = None
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

