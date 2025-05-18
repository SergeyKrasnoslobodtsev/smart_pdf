from dataclasses import dataclass
import enum
from typing import List, Optional

import cv2
import numpy as np

from .ocr_engine import OCR, OcrEngine
from .table_extractor import Table
from .extractor_base import BBox

class ParagraphType(enum.Enum):
    HEADER = 0
    FOOTER = 1
    NONE = 2

@dataclass
class Paragraph:
    bbox: BBox
    type: ParagraphType
    text: str = None
    page: int = 1

class ParagraphExtractor:
    def __init__(self, ocr: Optional[OcrEngine] = OcrEngine.TESSERACT):
        self.ocr = OCR(ocr_engine=ocr)
    
    def extract_from_image(self, gray: np.ndarray, tables: List[Table], page: int) -> List[Paragraph]:
        # Получаем блоки параграфов с типами HEADER/FOOTER/NONE
        paras = self._extract_paragraph_blocks(gray, tables)
        # Распознаём текст в каждом блоке
        for p in paras:
            x1, y1, x2, y2 = p.bbox.x1, p.bbox.y1, p.bbox.x2, p.bbox.y2
            roi = gray[y1:y2, x1:x2]
            p.text = self.ocr.extract(roi)
            p.page = page
        return paras
    
    def extract_from_pdf(self) -> List[Paragraph]:
        #TODO Для «digital» PDF можно реализовать через PyMuPDF или Camelot и OCR при необходимости
        raise NotImplementedError("PDF-поддержка пока не реализована")

    @staticmethod
    def _extract_paragraph_blocks(
        gray: np.ndarray,
        tables: List[Table] = None,
        margin: int = 5
    ) -> List[Paragraph]:
        h_img, w_img = gray.shape[:2]
        rois: List[tuple[int, int]] = []
        types: List[ParagraphType] = []

        # Формируем регионы интереса и соответствующие типы
        if tables:
            # Header: от 0 до y1 первой таблицы
            y_end = max(0, tables[0].bbox.y1 - margin)
            if y_end > 0:
                rois.append((0, y_end))
                types.append(ParagraphType.HEADER)
            # Footer: от y2 последней таблицы до низа
            y_start = min(h_img, tables[-1].bbox.y2 + margin)
            if y_start < h_img:
                rois.append((y_start, h_img))
                types.append(ParagraphType.FOOTER)
        else:
            # Нет таблиц — весь кадр как общий текст
            rois.append((0, h_img))
            types.append(ParagraphType.NONE)

        paragraphs: List[Paragraph] = []
        # По каждому ROI выполняем морфологическую сегментацию блоков текста
        for (y0, y1), p_type in zip(rois, types):
            roi = gray[y0:y1, :]
            if roi.size == 0:
                continue

            # Адаптивная бинаризация (текст становится белым на чёрном фоне)
            bin_img = cv2.adaptiveThreshold(
                roi, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                blockSize=15, C=10
            )
            # Склейка слов в строки
            horiz_k = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
            closed_h = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, horiz_k, iterations=2)
            # Склейка строк в параграфы
            vert_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
            closed_p = cv2.morphologyEx(closed_h, cv2.MORPH_CLOSE, vert_k, iterations=2)

            # Находим контуры параграфов
            contours, _ = cv2.findContours(
                closed_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w < 50 or h < 20:
                    continue
                pad = 5
                paragraphs.append(
                    Paragraph(
                        bbox=BBox(
                            x1=max(0, x - pad),
                            y1=max(0, y0 + y - pad),
                            x2=min(w_img, x + w + pad),
                            y2=min(h_img, y0 + y + h + pad)
                        ),
                        type=p_type
                    )
                )
        # Сортируем по положению на странице (сверху вниз)
        paragraphs.sort(key=lambda p: p.bbox.y1)
        return paragraphs
