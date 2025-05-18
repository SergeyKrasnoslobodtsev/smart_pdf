from abc import ABC
import enum
import os
from typing import Optional

import cv2
import numpy as np
import pytesseract

class OcrEngine(enum.Enum):
    TESSERACT = 0
    EASYOCR = 1
    PADDLEOCR = 2



class Engine(ABC):
    def extract_text(self, image: np.ndarray) -> str:
        pass
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        from src.image_processing import binarize
        # Обрежим изображение чтобы случайно рамка не попала 
        # Иначе нас везде будут приследовать символ |
        cropped = image[3:-3, 3:-3]
        border_size = 5
        bordered = cv2.copyMakeBorder(
            cropped,
            top=border_size,
            bottom=border_size,
            left=border_size,
            right=border_size,
            borderType=cv2.BORDER_CONSTANT,
            value=255
        )
        blur = cv2.medianBlur(bordered, 3)
        binary = binarize(blur)
        return binary

class TesseractEngine(Engine):
    def __init__(self):
        self.cfg = r'--oem 1 --psm 4 -l rus+eng'

    def extract_text(self, image: np.ndarray) -> str:
        image = self.preprocess(image)
        return pytesseract.image_to_string(image, config=self.cfg)

class EasyOcrEngine(Engine):
    def __init__(self):
        import easyocr
        self.reader = easyocr.Reader(['ru'], gpu=False)

    def extract_text(self, image: np.ndarray) -> str:
        image = self.preprocess(image)
        result = self.reader.readtext(image)
        return " ".join([res[1] for res in result])

class PaddleOcrEngine(Engine):
    def __init__(self):
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        from paddleocr import PaddleOCR
        self.reader = PaddleOCR(use_angle_cls=True, lang='ru')
    
    def extract_text(self, image: np.ndarray) -> str:
        image = self.preprocess(image)
        result = self.reader.ocr(image, cls=False)
        return " ".join([res[1][0] for res in result])

class OCR:
    def __init__(self, ocr_engine:Optional[OcrEngine]):
        match(ocr_engine):
            case ocr_engine.EASYOCR:
                self.ocr = EasyOcrEngine()
            case ocr_engine.PADDLEOCR:
                self.ocr = PaddleOcrEngine()
            case _: 
                self.ocr = TesseractEngine()

    def extract(self, image: np.ndarray) -> str:
        return self.ocr.extract_text(image)