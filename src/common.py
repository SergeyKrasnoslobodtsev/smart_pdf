
import numpy as np
import cv2

from pathlib import Path


def load_document_pdf(file_path: str, page:int = 0) -> np.ndarray:
    """Загрузка изображения из PDF файла с нормализацией в диапазон [0, 1]."""
    from pdf2image import convert_from_path
    
    images = convert_from_path(file_path, dpi=400)
    return np.array(images[page])

def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Конвертирует изображение в градации серого."""
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 2:
        return image
    else:
        raise ValueError("Unsupported image format. Expected 2D or 3D array.")
    



# ---- настройка имён -----------------------------------------------------------
SCAN_PDF_NAME = "ЕВР-НКАЗ.pdf"          # ожидаем True (скан)
STRUCTURED_PDF_NAME = "Акт сверки.pdf"  # ожидаем False (цифровой)
# ------------------------------------------------------------------------------

# вычисляем абсолютный путь к каталогу pdf, который находится на том же уровне,
# что и директория tests (где лежит этот файл)
TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
PDF_DIR = PROJECT_ROOT / "pdf"

SCAN_PDF = PDF_DIR / SCAN_PDF_NAME
STRUCTURED_PDF = PDF_DIR / STRUCTURED_PDF_NAME

def get_pdf_structure():
    return _read_pdf_bytes(STRUCTURED_PDF)

def get_pdf_scan():
    return _read_pdf_bytes(SCAN_PDF)

def _read_pdf_bytes(pdf_path: Path) -> bytes:
    """Читает файл и возвращает его содержимое в виде bytes."""
    assert pdf_path.exists(), f"Файл {pdf_path} не найден"
    return pdf_path.read_bytes()