import cv2
import numpy as np

    
def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Конвертирует изображение в градации серого."""
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 2:
        return image
    else:
        raise ValueError("Unsupported image format. Expected 2D or 3D array.")

def roteate_image(image: np.ndarray) -> np.ndarray:
    """Поворачивает изображение на 90 градусов по часовой стрелке."""
    if image.shape[0] > image.shape[1]:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    else:
        return image

def resize_image(image: np.ndarray, width: int = 500, height: int = 1000) -> np.ndarray:
    """Изменяет размер изображения с заданным коэффициентом масштабирования."""
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)