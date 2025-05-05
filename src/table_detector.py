import cv2 as cv
import numpy as np

from typing import Any, Dict, List, Tuple

from table_structure import (Table, Cell)


class TableDetector():
    """Реализация детектора таблиц на основе OpenCV"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def detect(self, image: np.ndarray) -> List[Table]:
        """Реализация обнаружения таблиц через OpenCV"""
        preprocessingd_image = self._image_processing(image)

        binary_image = self._detect_lines(preprocessingd_image)

        rects = self._get_rects(binary_image)
        
        tables: List[Table] = []

        for rect in rects:
            inner_cells = [
                inner for inner in rects 
                if self._is_inside(inner, rect) and inner != rect
            ]
            
            if inner_cells:
                x, y, w, h = rect
                table = Table(x, y, w, h)
                cells = [Cell(cx, cy, cw, ch) for cx, cy, cw, ch in inner_cells]
                table.add_row(cells)
                tables.append(table)

        return tables

    def _is_inside(self, inner, outer):
        """Проверка, вложен ли один прямоугольник в другой"""
        ix, iy, iw, ih = inner
        ox, oy, ow, oh = outer
        return ox <= ix and oy <= iy and (ix + iw) <= (ox + ow) and (iy + ih) <= (oy + oh)

    def _image_processing(self, image: np.ndarray) -> np.ndarray:
        img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        _, img_bin = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
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

        # Обнаружение линий
        vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, np.array(image).shape[1]//250))
        horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (np.array(image).shape[1]//200, 1))

        # Обработка вертикальных линий
        eroded_image_ver = cv.erode(image, vertical_kernel, iterations=5)
        vertical_lines = cv.dilate(eroded_image_ver, vertical_kernel, iterations=5)

        # Обработка горизонтальных линий
        eroded_image_hor = cv.erode(image, horizontal_kernel, iterations=5)
        horizontal_lines = cv.dilate(eroded_image_hor, horizontal_kernel, iterations=5)

        # Объединение линий
        combined_lines = cv.bitwise_or(vertical_lines, horizontal_lines)

        return combined_lines

def draw_table(image: np.ndarray, tables: List[Table]) -> np.ndarray:
    """Отрисовка таблиц на изображении"""
    for table in tables:
        x, y, w, h = table.x, table.y, table.w, table.h
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        for i, row in enumerate(table.rows):
            for j, cell in enumerate(row.cells):
                cx, cy, cw, ch = cell.x, cell.y, cell.w, cell.h
                cv.rectangle(image, (cx, cy), (cx + cw, cy + ch), (255, 0, 0), 1)
                          
                text = f"{i}:{j} - {cell.text}"
                cv.putText(image, text,
                            (cell.x + 5, cell.y + 20),
                            cv.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2)

    return image 


def main():
    from pdf2image import convert_from_path
    # Пример использования
    FILE = './pdf/АС Евросибэнерго-НКАЗ.pdf'
    
    images = convert_from_path(FILE, dpi=400)

    image = np.array(images[0])

    detector = TableDetector(config={})
    tables = detector.detect(image)
    image_with_tables = draw_table(image, tables)

    cv.namedWindow('Tables', cv.WINDOW_NORMAL)      
    cv.imshow('Tables', image_with_tables)
    cv.resizeWindow('Tables', 640, 480)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()  





