from typing import List
import numpy as np


def generate_blurred_images(image: np.ndarray) -> List[np.ndarray]:
    """Генерация тестовых изображений с различными уровнями размытия (Gaussian blur)."""
    import cv2
    blur_levels = [0, 3, 7, 11, 15]  # 0 - без размытия, далее увеличивается радиус ядра
    test_images = []
    for ksize in blur_levels:
        if ksize == 0:
            blurred_image = image.copy()
        else:
            blurred_image = cv2.GaussianBlur(image, (ksize, ksize), 0)
        test_images.append(blurred_image)
    return test_images

def generate_over_exposed_images(image: np.ndarray) -> List[np.ndarray]:
    """Генерация тестовых изображений с различными уровнями переэкспонирования."""
    import cv2
    exposure_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] # Коэффициенты для изменения яркости

    overexposed_images = []
    for level in exposure_levels:
        overexposed = cv2.addWeighted(image, 1 - level, np.ones_like(image) * 255, level, 0)
        overexposed_images.append(overexposed.astype(np.uint8))
    return overexposed_images

def generate_under_exposed_images(image: np.ndarray) -> List[np.ndarray]:
    """Генерация тестовых изображений с различными уровнями недоэкспонирования."""
    import cv2
    exposure_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] # Коэффициенты для изменения яркости

    underexposed_images = []
    for level in exposure_levels:
        underexposed = cv2.addWeighted(image, 1 - level, np.zeros_like(image), level, 0)
        underexposed_images.append(underexposed.astype(np.uint8))
    return underexposed_images

def generate_salt_and_pepper_noise(image: np.ndarray) -> List[np.ndarray]:
    """Генерация тестовых изображений с различными уровнями шума соль-перец."""

    noise_levels = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05] # Уровни шума

    noisy_images = []
    for level in noise_levels:
        noisy_image = image.copy()
        num_salt = np.ceil(level * image.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy_image[coords[0], coords[1], :] = 255

        num_pepper = np.ceil(level * image.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy_image[coords[0], coords[1], :] = 0

        noisy_images.append(noisy_image.astype(np.uint8))
    return noisy_images

def generate_gaussian_noise(image: np.ndarray) -> List[np.ndarray]:
    """Генерация тестовых изображений с различными уровнями гауссовского шума."""
    import cv2
    noise_levels = [0.0, 10, 20, 30, 40, 50] # Уровни шума

    noisy_images = []
    for level in noise_levels:
        noise = np.random.normal(0, level, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
        noisy_images.append(noisy_image)
    return noisy_images