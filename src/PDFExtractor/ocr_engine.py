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
    def preprocess(self, gray: np.ndarray) -> np.ndarray:

        # h, w = gray.shape

        # _, thresh = cv2.threshold(
        #     gray, 
        #     0, 255, 
        #     cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        # )  




        # clean = cv2.bitwise_not(thresh)
        border = 10
        gray = cv2.copyMakeBorder(
            gray, border, border, border, border,
            borderType=cv2.BORDER_CONSTANT,
            value=255
        )
        # blur = cv2.medianBlur(clean, 3)

        return gray

from PIL import Image
class TesseractEngine(Engine):
    def __init__(self):
        self.cfg = r'--oem 1 --psm 6 -l rus+eng'

    def extract_text(self, image: np.ndarray) -> str:
        image = self.preprocess(image)
        # Image.fromarray(image).show()
        return pytesseract.image_to_string(image, config=self.cfg)

class EasyOcrEngine(Engine):
    def __init__(self):
        os.environ["KMP_DEVICE_THREAD_LIMIT"] = "4"
        os.environ["OMP_THREAD_LIMIT"] = "4"
        import easyocr
        
        self.reader = easyocr.Reader(['ru', 'en'], gpu=False)

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