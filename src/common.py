
import numpy as np


def load_document_pdf(file_path: str, page:int = 0) -> np.ndarray:
    """Загрузка изображения из PDF файла с нормализацией в диапазон [0, 1]."""
    from pdf2image import convert_from_path
    
    images = convert_from_path(file_path, dpi=400)
    return np.array(images[page])