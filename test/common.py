from typing import List
import numpy as np


def load_document_pdf(file_path: str, page:int = 0) -> np.ndarray:
    """Загрузка изображения из PDF файла с нормализацией в диапазон [0, 1]."""
    from pdf2image import convert_from_path
    images = convert_from_path(file_path, dpi=400)
    return np.array(images[page])


def visualize(original: np.ndarray, noise_images: List[np.ndarray],  indicator:List[str]) -> None:
    """Визуализация оригинала и размытых изображений с метрикой PSNR."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, len(noise_images) + 1, figsize=(15, 5))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")
    for i, (image, ind) in enumerate(zip(noise_images, indicator)):
        axes[i + 1].imshow(image)
        axes[i + 1].set_title(ind)
        axes[i + 1].axis("off")
        print(ind)
    
    plt.tight_layout()
    plt.show()
