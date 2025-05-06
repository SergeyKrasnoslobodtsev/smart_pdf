
import PIL
import PIL.Image
import cv2 as cv
import numpy as np

from typing import Any, Dict, List, Tuple

from .table_structure import (Table, Cell)
from PIL import Image, ImageDraw, ImageFont

class TableDetector():
    """Реализация детектора таблиц на основе OpenCV"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def detect(self, image: np.ndarray) -> List[Table]:
        """Реализация обнаружения таблиц через OpenCV"""
        preprocessing_image = self._image_processing(image)

        binary_image = self._detect_lines(preprocessing_image)
        Image.fromarray(binary_image).show()   
        rects = self._get_rects(binary_image)
        
        tables: List[Table] = []

        # Сначала ищем все прямоугольники, которые могут быть таблицами
        candidate_tables = []
        for rect in rects:
            inner_cells = [
                inner for inner in rects
                if self._is_inside(inner, rect) and inner != rect
            ]
            if inner_cells:
                candidate_tables.append((rect, inner_cells))

        # 1. Определяем вложенные таблицы
        nested_table_rects = [
            rect for i, (rect, _) in enumerate(candidate_tables)
            for j, (other_rect, _) in enumerate(candidate_tables)
            if i != j and self._is_inside(rect, other_rect)
        ]

        MIN_TABLE_WIDTH = 50   # подберите под ваш случай
        MIN_TABLE_HEIGHT = 50
        MIN_CELLS_COUNT = 2 # подберите под ваш случай

        # 2. Фильтруем таблицы (оставляем только невложенные)
        filtered_tables = [
            (rect, inner_cells)
            for rect, inner_cells in candidate_tables
            if rect not in nested_table_rects
            and rect[2] > MIN_TABLE_WIDTH
            and rect[3] > MIN_TABLE_HEIGHT
            and len(inner_cells) >= MIN_CELLS_COUNT
        ]

        # 3. Исключаем ячейки, которые лежат внутри любой вложенной таблицы
        for rect, inner_cells in filtered_tables:
            x, y, w, h = rect
            table = Table(x, y, w, h)
            filtered_cells = [
                Cell(cx, cy, cw, ch)
                for (cx, cy, cw, ch) in inner_cells
                # Ячейка не вложена ни в одну другую ячейку (кроме самой себя)
                if not any(
                    self._is_inside((cx, cy, cw, ch), (other_cx, other_cy, other_cw, other_ch))
                    for (other_cx, other_cy, other_cw, other_ch) in inner_cells
                    if (other_cx, other_cy, other_cw, other_ch) != (cx, cy, cw, ch)
                )
            ]
            table.add_row(filtered_cells)
            tables.append(table)

        return tables

    def _is_inside(self, inner, outer):
        """Проверка, вложен ли один прямоугольник в другой"""
        ix, iy, iw, ih = inner
        ox, oy, ow, oh = outer
        return ox <= ix and oy <= iy and (ix + iw) <= (ox + ow) and (iy + ih) <= (oy + oh)

    def _image_processing(self, image: np.ndarray) -> np.ndarray:
        
        _, img_bin = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        img_bin = 255-img_bin

        return img_bin
    
    def _get_rects(self, image: np.ndarray) -> Tuple[int, int, int, int]:
         # Поиск контуров
        contours, _ = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # Создание списка координат
        rects = []
        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            if w > 10 and h > 10:
                rects.append((x, y, w, h))

        return rects
    
    def _detect_lines(self, image:np.ndarray) -> np.ndarray:

        # Морфология для длинных линий
        vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, image.shape[0] // 230))
        horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (image.shape[1] // 230, 1))

        vertical_lines = cv.morphologyEx(image, cv.MORPH_OPEN, vertical_kernel, iterations=2)
        horizontal_lines = cv.morphologyEx(image, cv.MORPH_OPEN, horizontal_kernel, iterations=2)

        # Морфология для коротких линий
        small_vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 35))
        small_horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (35, 1))

        small_vertical_lines = cv.morphologyEx(image, cv.MORPH_OPEN, small_vertical_kernel, iterations=2)
        small_horizontal_lines = cv.morphologyEx(image, cv.MORPH_OPEN, small_horizontal_kernel, iterations=2)

        # Объединяем все линии
        all_vertical = cv.bitwise_or(vertical_lines, small_vertical_lines)
        all_horizontal = cv.bitwise_or(horizontal_lines, small_horizontal_lines)
        combined_lines = cv.bitwise_or(all_vertical, all_horizontal)

        return combined_lines

def draw_table(image: np.ndarray, tables: List[Table]) -> PIL.Image.Image:
    """Отрисовка таблиц на изображении с поддержкой русского текста через PIL"""
    
    # Конвертируем из OpenCV (BGR) в PIL (RGB)
    image_rgb = cv.cvtColor(image.copy(), cv.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)
    
    # Загружаем шрифт с поддержкой кириллицы
    try:
        font = ImageFont.truetype("arial.ttf", 14)  # Стандартный шрифт Windows
    except IOError:
        # Запасной вариант, если arial не найден
        font = ImageFont.load_default()
    
    for table in tables:
        x, y, w, h = table.x, table.y, table.w, table.h
        # Рисуем рамку таблицы
        draw.rectangle([(x, y), (x + w, y + h)], outline=(0, 255, 0), width=2)
        
        for i, row in enumerate(table.rows):
            for j, cell in enumerate(row.cells):
                cx, cy, cw, ch = cell.x, cell.y, cell.w, cell.h
                # Рисуем рамку ячейки
                draw.rectangle([(cx, cy), (cx + cw, cy + ch)], outline=(255, 0, 0), width=1)
                
                # Добавляем текст - теперь поддерживает русский язык
                text = f"{i}:{j} - {cell.text}"
                draw.text((cell.x + 5, cell.y + 5), text, font=font, fill=(255, 0, 0))
    
    return pil_image


def main():
    from pdf2image import convert_from_path

    # Пример использования
    FILE = './pdf/АС Евросибэнерго-НКАЗ.pdf'
    
    images = convert_from_path(FILE, dpi=400)

    image = np.array(images[0])
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    detector = TableDetector(config={})
    tables = detector.detect(gray)
    image_with_tables = draw_table(gray, tables)
    image_with_tables.show()

if __name__ == "__main__":
    main()  





