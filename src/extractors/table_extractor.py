from dataclasses import dataclass, field
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np

from .ocr_engine import OCR, OcrEngine
from .extractor_base import BBox


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
    page: int = 1


class TableExtractor:

    def __init__(self, ocr:Optional[OcrEngine]=OcrEngine.TESSERACT, max_workers: int = 4):
        self.ocr = OCR(ocr_engine=ocr)
        self.max_workers = max_workers

    def extract_from_image(self, gray: np.ndarray, page: int) -> List[Table]:
        from src.image_processing import find_lines

        # обнаружение линий таблицы
        h_lines, v_lines = find_lines(gray, 50)
        mask = h_lines + v_lines
        contours = self._find_table_contours(mask)

        tables: List[Table] = []
        for x, y, w, h in contours:
            roi_v = v_lines[y:y+h, x:x+w]
            roi_h = h_lines[y:y+h, x:x+w]

            # получаем локальные ячейки
            cells_local = self._cells_from_line_masks(roi_v, roi_h, eps=5)
            if not cells_local:
                continue

            # готовим задачи для OCR
            tasks = []
            for cell in cells_local:
                cx0 = cell.bbox.x1 + x
                cy0 = cell.bbox.y1 + y
                cx1 = cell.bbox.x2 + x
                cy1 = cell.bbox.y2 + y
                roi = gray[cy0:cy1, cx0:cx1]
                tasks.append(((cx0, cy0, cx1, cy1), cell.row, cell.col, roi))

            results = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_cell = {
                    executor.submit(self.ocr.extract, roi): (bbox, row, col)
                    for bbox, row, col, roi in tasks
                }
                for future in as_completed(future_to_cell):
                    bbox, row, col = future_to_cell[future]
                    text = future.result()
                    results.append((bbox, row, col, text))

            # формируем объекты Cell
            cells: List[Cell] = []
            for bbox, row, col, text in results:
                x0, y0, x1, y1 = bbox
                cells.append(Cell(bbox=BBox(x0, y0, x1, y1), row=row, col=col, text=text))

            # добавляем таблицу с распознанными ячейками
            tables.append(Table(bbox=BBox(x, y, x+w, y+h), cells=cells, page=page))

        return tables
    

    def extract_from_pdf(self):
        #TODO Для «digital» PDF можно реализовать через PyMuPDF или Camelot и OCR при необходимости
        raise NotImplementedError("PDF-поддержка пока не реализована")

    @staticmethod
    def _find_table_contours(mask: np.ndarray) -> list:
        """Находит контуры таблиц на изображении."""
        contours, __ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # Удаляем контуры, которые слишком малы
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        cont:List[Table] = []
        for c in contours:
            c_poly = cv2.approxPolyDP(c, 3, True)
            if len(c_poly) < 4:
                continue
            x, y, w, h = cv2.boundingRect(c_poly)
            if w < 100 or h < 100:
                continue
            cont.append((x, y, w, h))
        # Сортируем контуры по y-координате
        cont = sorted(cont, key=lambda c: c[1])
        return cont

    @staticmethod
    def _cells_from_line_masks(v_mask: np.ndarray,
                                h_mask: np.ndarray,
                                eps: int = 5) -> list[Cell]:
        """Возвращает список ячеек с colspan/rowspan."""
        from src.utils import cluster_coord, line_in_bbox
   
        kernel = np.ones((3, 3), np.uint8)
        inter = cv2.bitwise_and(cv2.dilate(v_mask, kernel, 1),
                                cv2.dilate(h_mask, kernel, 1))

        pts = cv2.findContours(inter, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[0]
        
        xs = cluster_coord([cv2.moments(c)['m10']/cv2.moments(c)['m00'] for c in pts], eps)
        ys = cluster_coord([cv2.moments(c)['m01']/cv2.moments(c)['m00'] for c in pts], eps)
        
        xs = list(map(int, xs)) 
        ys = list(map(int, ys))

        if len(xs) < 2 or len(ys) < 2:
            return []

        # матрица посещения, чтоб не создавать ячейку дважды
        n_rows, n_cols = len(ys) - 1, len(xs) - 1
        used = [[False]*n_cols for _ in range(n_rows)]
        
        cells: list[Cell] = []

        for r in range(n_rows):
            for c in range(n_cols):

                if used[r][c]:
                    continue
                # print(f'R{r}:C{c}')    
                # стартовая граница
                x0, y0 = xs[c], ys[r]
                x1, y1 = xs[c+1], ys[r+1]
                max_c, max_r = c, r

                # COLUMNSPAN
                while (max_c + 1 < len(xs) - 1 and
                    not line_in_bbox(v_mask, x0, y0, xs[max_c+1], y1, axis='v')):
                    max_c += 1
                    x1 = xs[max_c+1]

                # ROWSPAN
                while (max_r + 1 < len(ys) - 1 and
                    not line_in_bbox(h_mask, x0, y0, x1, ys[max_r+1], axis='h')):
                    max_r += 1
                    y1 = ys[max_r+1]

                # помечаем прямоугольник как использованный
                for rr in range(r, max_r+1):
                    for cc in range(c, max_c+1):
                        used[rr][cc] = True
                # if r==2 and c == 0:
                #     return cells
                
                cells.append(
                    Cell(
                        bbox=BBox(x0, y0, x1, y1),
                        row=r,
                        col=c,
                        text="",       
                    )
                )
        return cells