from typing import Any, List, Optional, Tuple
import cv2
import numpy as np


from .image_processing import find_lines
from .image_processing import find_max_contours
from .image_processing import detected_text
from .image_processing import remove_lines_by_mask


from .utils import page_to_image
from .utils import find_line_positions 
from .utils import has_line

from .base_extractor import (BBox, 
                             BaseExtractor, 
                             Cell, 
                             Paragraph, 
                             ParagraphType, 
                             Table)

from PIL import Image

from .ocr_engine import OCR, OcrEngine
from concurrent.futures import ThreadPoolExecutor, as_completed

class ScanExtractor(BaseExtractor):
    '''Извлекает структуру документа если он отсканирован'''
    def __init__(self, ocr:Optional[OcrEngine]=OcrEngine.TESSERACT, max_workers: int = 4):
        self.ocr = OCR(ocr_engine=ocr)
        self.max_workers = max_workers

    def _process(self, page) -> Tuple[List[Paragraph], List[Table]]:
        
        image = page_to_image(page)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h_lines, v_lines = find_lines(gray, 200)

        mask = h_lines + v_lines
        # найдем контуры таблиц
        contours = find_max_contours(mask, max=10)
        # находим абзацы
        paragraphs = self._extract_paragraph_blocks(gray, contours, margin=5)

        tables: List[Table] = []
        for x, y, w, h in contours:
            roi_v = v_lines[y:y+h, x:x+w]
            roi_h = h_lines[y:y+h, x:x+w]
            # нужно удалить таблицу(линии) из основного изображения
            # иначе распознавание плохо работает
            mask_roi = cv2.bitwise_or(
                v_lines[y:y+h, x:x+w],
                h_lines[y:y+h, x:x+w]
            )
            
            gray[y:y+h, x:x+w] = remove_lines_by_mask(
                gray[y:y+h, x:x+w],
                mask_roi
            )
            # получаем сетку таблицы (по умолчанию объединим только столбцы)
            # можно объдинить и столбцы и строки, но есть таблицы, где строки объединены 
            # не логично, либо линия скрыта. 
            # -------|-----------|
            #  text  |    0.00   |
            #        |-----------|  
            #  text  |    0.00   |
            # -------|-----------|
            cells = self._grid_table(x, y, roi_v, roi_h, span_mode=1)

            tables.append(
                Table(
                    bbox=BBox(x, y, x+w, y+h),
                    cells=cells, 
                )
            )
        # Обработаем изображение медианный фильтр избавит от перца (зерна) на изображении
        # также немного сгладит буквы
        cleaned = cv2.medianBlur(gray, 3)
        # расширем текст и фон сделаем равномерным белым
        cleaned = cv2.adaptiveThreshold(cleaned, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=17, C=10)
        # посмотри что получилось
        Image.fromarray(cleaned).show()
        
        tasks: List[Tuple[Any, np.ndarray]] = []
        for p in paragraphs:
            roi = cleaned[p.bbox.y1:p.bbox.y2, p.bbox.x1:p.bbox.x2]
            tasks.append((p, roi))
        for tbl in tables:
            for c in tbl.cells:
                pad_box = c.bbox.padding(2)
                roi = cleaned[pad_box.y1:pad_box.y2, pad_box.x1:pad_box.x2]
                # if c.row == 1:
                #     break
                tasks.append((c, roi))

        # распараллеливаем вызовы OCR в потоках
        with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
            future_to_obj = {
                exe.submit(self.ocr.extract, roi): obj
                for obj, roi in tasks
            }
            for fut in as_completed(future_to_obj):
                obj = future_to_obj[fut]
                try:
                    obj.text = fut.result()
                except Exception:
                    obj.text = ''

        return paragraphs, tables


    def _extract_paragraph_blocks(
        self,
        gray: np.ndarray,
        table_contours: List[Tuple[int, int, int, int]], 
        margin: int = 5
    ) -> List[Paragraph]:
        h_img, w_img = gray.shape[:2]
        
        # 1. Создать копию изображения gray для модификации
        gray_for_text_detection = gray.copy()

        # 2. Если есть контуры таблиц, "закрасить" эти области
        if table_contours:
            for x, y, w, h in table_contours:
                cv2.rectangle(gray_for_text_detection, (x, y), (x + w, y + h), (255), -1) # Закрасить белым

        # 3. На модифицированном изображении получить маску текстовых регионов
        text_mask = detected_text(gray_for_text_detection)
        # Image.fromarray(text_mask).show()

        # 4. Найти контуры абзацев
        paragraph_cv_contours, _ = cv2.findContours(
            text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        accepted_boxes = []

        for cnt in paragraph_cv_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Игнорируем слишком маленькие контуры
            if w < 10 or h < 10: # Пороговые значения можно подобрать
                continue
            
            pad = 5 # Отступ для bbox абзаца
            para_bbox = BBox(
                x1=max(0, x - pad),
                y1=max(0, y - pad),
                x2=min(w_img, x + w + pad),
                y2=min(h_img, y + h + pad)
            )
            is_nested = any(parent.contains(para_bbox) for parent in accepted_boxes)
            
            if is_nested:
                continue
            
            accepted_boxes = [
                parent for parent in accepted_boxes
                if not para_bbox.contains(parent)
            ]

            accepted_boxes.append(para_bbox)

        header_region_y_end = 0
        footer_region_y_start = h_img

        if table_contours:
            # table_contours отсортированы по y в _process
            first_table_x, first_table_y, first_table_w, first_table_h = table_contours[0]
            last_table_x, last_table_y, last_table_w, last_table_h = table_contours[-1]
            
            header_region_y_end = max(0, first_table_y - margin)
            footer_region_y_start = min(h_img, last_table_y + last_table_h + margin)
            # Определяем тип абзаца
            current_para_type = ParagraphType.NONE # По умолчанию NONE
        
        paragraphs: List[Paragraph] = []
        for bbox in accepted_boxes:
            if table_contours:
                # Если абзац полностью находится в области HEADER (выше первой таблицы)
                if bbox.y2 <= header_region_y_end:
                    current_para_type = ParagraphType.HEADER
                # Если абзац полностью находится в области FOOTER (ниже последней таблицы)
                elif bbox.y1 >= footer_region_y_start:
                    current_para_type = ParagraphType.FOOTER
                # Иначе остается NONE (текст между таблицами или рядом с ними, но не в header/footer зонах)
            else:
                current_para_type = ParagraphType.NONE

            paragraphs.append(
                Paragraph(
                    bbox=bbox,
                    type=current_para_type,
                    text='' # Текст будет извлечен позже OCR
                )
            )
        
        # Сортируем абзацы по их вертикальному положению
        paragraphs.sort(key=lambda p: p.bbox.y1)
        return paragraphs
    

    def _grid_table(self, x:int, y:int,
                        pure_v: np.ndarray,
                        pure_h: np.ndarray,
                        span_mode: int = 0) -> list[Cell]:
        """Возвращает список ячеек с colspan/rowspan."""

        xs = find_line_positions(pure_v, axis=0)
        ys_detected = find_line_positions(pure_h, axis=1)

        if len(xs) < 2:
            return []
        
        # заплатка на случай отсутствия нижней горизонтальной линиии
        # в конце таблицы
        # |--------|------|-------|
        # |        |      |       |

        ys = list(ys_detected) # Создаем изменяемую копию
        
        table_roi_height = pure_h.shape[0]
        # Минимальная высота, чтобы считать пространство под последней линией потенциальной строкой
        min_row_height_threshold = 10 

        # Если были обнаружены горизонтальные линии, и последняя из них находится не у самого низа ROI,
        # и есть достаточно места для еще одной строки, добавляем нижнюю границу ROI как виртуальную линию.
        if ys: # Если найдена хотя бы одна горизонтальная линия
            if ys[-1] < table_roi_height - min_row_height_threshold:
                # Подразумевается, что есть потенциальная последняя строка, у которой отсутствует нижняя линия.
                # Мы предполагаем, что вертикальные линии (xs) корректно определяют столбцы до низа ROI.
                ys.append(table_roi_height)
        # Если ys_detected был пуст, ys останется пустым.
        # Это будет обработано следующей проверкой.

        if len(ys) < 2: # Необходимо как минимум две y-координаты для определения строки
            return []

        # матрица посещения, чтоб не создавать ячейку дважды
        n_rows, n_cols = len(ys) - 1, len(xs) - 1
        used = [[False]*n_cols for _ in range(n_rows)]
        margin = 5
        line_frac = 0.8
        cells: list[Cell] = []
        for r in range(n_rows):
            for c in range(n_cols):
                if used[r][c]:
                    continue
                # стартовые границы
                x0, y0 = xs[c], ys[r]
                x1, y1 = xs[c+1], ys[r+1]
                col_span = row_span = 1
                # colspan
                if span_mode in (0,1):
                    end_c = c
                    while end_c + 1 < n_cols:
                        new_x = xs[end_c+2]
                        rs, re = y0 + margin, y1 - margin
                        cs, ce = x0 + margin, new_x - margin
                        if re <= rs or ce <= cs:
                            break
                        region = pure_v[rs:re, cs:ce]
                        min_len = int((re - rs) * line_frac)
                        if has_line(region, min_len, axis=0):
                            break
                        end_c += 1
                        x1 = new_x
                    col_span = end_c - c + 1
                # rowspan
                if span_mode in (0,2):
                    end_r = r
                    while end_r + 1 < n_rows:
                        new_y = ys[end_r+2]
                        rs, re = y0 + margin, new_y - margin
                        cs, ce = x0 + margin, x1 - margin
                        if re <= rs or ce <= cs:
                            break
                        region = pure_h[rs:re, cs:ce]
                        min_len = int((ce - cs) * line_frac)
                        if has_line(region, min_len, axis=1):
                            break
                        end_r += 1
                        y1 = new_y
                    row_span = end_r - r + 1
                # помечаем used
                for rr in range(r, r + row_span):
                    for cc in range(c, c + col_span):
                        used[rr][cc] = True

                
                cells.append(
                    Cell(
                        bbox=BBox(x0 + x, y0 + y, x1 + x, y1 + y),
                        row=r,
                        col=c,
                        text='',       
                    )
                )
        return cells  