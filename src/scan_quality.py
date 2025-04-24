from dataclasses import dataclass

from typing import Dict
from matplotlib import pyplot as plt
import numpy as np
import cv2

import config_loader as cfg
from image_quality_report import (ImageQualityReport,
                                  IssueLevel, 
                                  IssueType)  

@dataclass
class ExposureAnalysis:
    """Результаты анализа экспозиции изображения."""
    dark_ratio: float  # Доля темных пикселей
    dark_count: int    # Количество темных пикселей
    light_ratio: float  # Доля светлых пикселей
    light_count: int    # Количество светлых пикселей
    mid_ratio: float    # Доля пикселей среднего тона
    mid_count: int      # Количество пикселей среднего тона

@dataclass
class Metrics:
    """Метрики качества изображения."""
    sharpness: float
    entropy: float
    dark_ratio: float
    light_ratio: float
    mid_ratio: float
    contrast: float

class ImageQualityAnalyzer:
    """Анализатор качества изображений с выявлением проблем и рекомендациями."""
    
    def __init__(self, config: cfg.ImageQualityConfig):
        self.config = config

    
    def analyze(self, image: np.ndarray, visualize: bool = False) -> ImageQualityReport:
        """
        Анализирует качество изображения и возвращает отчет с выявленными проблемами.
        
        Args:
            image: Входное BGR изображение
            visualize: Флаг для визуализации результатов анализа
            
        Returns:
            ImageQualityReport: Отчет о качестве изображения с рекомендациями
        """
        # Вычисление всех метрик качества изображения
        metrics = self._compute_metrics(image)
        
        report = ImageQualityReport(
            metrics=metrics.__dict__, 
            image_size=image.shape[:2]
        )

        # Визуализация метрик при необходимости
        if visualize:
            self._visualize_metrics(image, metrics)
        
        self._check_blur(metrics, report)
        self._check_exposure(metrics, report)
        self._check_noise_gaussian(metrics, report)
        self._check_noise_salt_pepper(metrics, report)
        self._check_contrast(metrics, report)
        
        return report
    
    def _compute_metrics(self, image: np.ndarray) -> Metrics:
        """Вычисляет все метрики качества изображения."""
        # Конвертация в оттенки серого
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Вычисление резкости (используем лапласиан)
        sharpness = self._compute_sharpness(gray)
        
        # Анализ экспозиции
        exposure = self._compute_exposure(gray)

        # Расчет энтропии
        entropy = self._compute_entropy(gray)

        # Расчет контраста
        contrast = self._compute_contrast(gray)
        
        return Metrics(
            sharpness=sharpness,
            entropy=entropy,
            dark_ratio=exposure.dark_ratio,
            light_ratio=exposure.light_ratio,
            mid_ratio=exposure.mid_ratio,
            contrast=contrast
        )
    
    def _compute_sharpness(self, gray_img: np.ndarray) -> float:
        """Вычисляет метрику резкости изображения."""
        laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
        return np.var(laplacian)
    
    def _compute_exposure(self, image: np.ndarray, 
                          dark_threshold: int = 50,
                          light_threshold: int = 220) -> ExposureAnalysis:
        """
        Анализ экспозиции изображения с выявлением темных и светлых пикселей.
        """
        # Применение размытия Гаусса для уменьшения шума
        # (это стандартная практика перед анализом гистограмм)
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Вычисление гистограммы
        hist = cv2.calcHist([blurred], [0], None, [256], [0, 256]).flatten()
        total_pixels = blurred.size
        
        # Анализ темных и светлых пикселей
        dark_pixels_count = np.sum(hist[:dark_threshold+1])
        light_pixels_count = np.sum(hist[light_threshold:])
        mid_pixels_count = total_pixels - dark_pixels_count - light_pixels_count
        
        dark_ratio = dark_pixels_count / total_pixels
        light_ratio = light_pixels_count / total_pixels
        mid_ratio = mid_pixels_count / total_pixels

        return ExposureAnalysis(
            dark_ratio=dark_ratio,
            dark_count=int(dark_pixels_count),
            light_ratio=light_ratio,
            light_count=int(light_pixels_count),
            mid_ratio=mid_ratio,
            mid_count=int(mid_pixels_count)
        )
    
    def _compute_entropy(self, gray_img: np.ndarray) -> float:
        """Вычисляет энтропию изображения."""
        hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256]).flatten()
        hist = hist / hist.sum()
        non_zero_hist = hist[hist > 0]
        entropy = -np.sum(non_zero_hist * np.log2(non_zero_hist))
        normalized_entropy = entropy / 8.0  # Нормализация к диапазону 0-1
        return float(normalized_entropy)
    
    def _compute_contrast(self, gray_img: np.ndarray) -> float:
        """Вычисляет контраст изображения."""
        return np.std(gray_img) / 255.0

    def _check_blur(self, metrics: Metrics, report: ImageQualityReport) -> None:
        """Проверяет наличие размытия и определяет уровень проблемы."""
        sharpness = metrics.sharpness
        blur_cfg = self.config.blur

        if sharpness < blur_cfg.not_usable.values.sharpness:
            report.add_issue(
                type=IssueType.BLUR,
                level=IssueLevel.NOT_USABLE,
                metrics={"sharpness": sharpness},
                recommendation=blur_cfg.not_usable.message,
            )
        elif sharpness < blur_cfg.high.values.sharpness:
            report.add_issue(
                type=IssueType.BLUR,
                level=IssueLevel.HIGH,
                metrics={"sharpness": sharpness},
                recommendation=blur_cfg.high.message,
            )
        elif sharpness < blur_cfg.medium.values.sharpness:
            report.add_issue(
                type=IssueType.BLUR,
                level=IssueLevel.MEDIUM,
                metrics={"sharpness": sharpness},
                recommendation=blur_cfg.medium.message,
            )
        elif sharpness < blur_cfg.low.values.sharpness:
            report.add_issue(
                type=IssueType.BLUR,
                level=IssueLevel.LOW,
                metrics={"sharpness": sharpness},
                recommendation=blur_cfg.low.message,
            )
    
    def _check_exposure(self, metrics: Metrics, report: ImageQualityReport) -> None:
        """Проверяет проблемы экспозиции (недодержка/передержка)."""
        overexposure_cfg = self.config.overexposure
        underexposure_cfg = self.config.underexposure
        ligth_ratio = metrics.light_ratio
        dark_ratio = metrics.dark_ratio
        entropy = metrics.entropy

        if ligth_ratio > overexposure_cfg.not_usable.values.light_ratio:
            report.add_issue(
                type=IssueType.OVEREXPOSURE,
                level=IssueLevel.NOT_USABLE,
                metrics={"light_ratio": ligth_ratio, "entropy": entropy},
                recommendation=overexposure_cfg.not_usable.message,
            )
        elif ligth_ratio > overexposure_cfg.high.values.light_ratio:
            report.add_issue(
                type=IssueType.OVEREXPOSURE,
                level=IssueLevel.HIGH,
                metrics={"light_ratio": ligth_ratio, "entropy": entropy},
                recommendation=overexposure_cfg.high.message,
            )
        elif ligth_ratio > overexposure_cfg.medium.values.light_ratio:
            report.add_issue(
                type=IssueType.OVEREXPOSURE,
                level=IssueLevel.MEDIUM,
                metrics={"light_ratio": ligth_ratio, "entropy": entropy},
                recommendation=overexposure_cfg.medium.message,
            )
        elif ligth_ratio > overexposure_cfg.low.values.light_ratio:
            report.add_issue(
                type=IssueType.OVEREXPOSURE,
                level=IssueLevel.LOW,
                metrics={"light_ratio": ligth_ratio, "entropy": entropy},
                recommendation=overexposure_cfg.low.message,
            )
        
        if dark_ratio > underexposure_cfg.not_usable.values.dark_ratio:
            report.add_issue(
                type=IssueType.UNDEREXPOSURE,
                level=IssueLevel.NOT_USABLE,
                metrics={"dark_ratio": dark_ratio, "entropy": entropy},
                recommendation=underexposure_cfg.not_usable.message,
            )
        elif dark_ratio > underexposure_cfg.high.values.dark_ratio:
            report.add_issue(
                type=IssueType.UNDEREXPOSURE,
                level=IssueLevel.HIGH,
                metrics={"dark_ratio": dark_ratio, "entropy": entropy},
                recommendation=underexposure_cfg.high.message,
            )
        elif dark_ratio > underexposure_cfg.medium.values.dark_ratio:
            report.add_issue(
                type=IssueType.UNDEREXPOSURE,
                level=IssueLevel.MEDIUM,
                metrics={"dark_ratio": dark_ratio, "entropy": entropy},
                recommendation=underexposure_cfg.medium.message,
            )
        elif dark_ratio > underexposure_cfg.low.values.dark_ratio:
            report.add_issue(
                type=IssueType.UNDEREXPOSURE,
                level=IssueLevel.LOW,
                metrics={"dark_ratio": dark_ratio, "entropy": entropy},
                recommendation=underexposure_cfg.low.message,
            )
    
    def _check_noise_gaussian(self, metrics: Metrics, report: ImageQualityReport) -> None:
        """Проверяет наличие шума на изображении."""
        noise = metrics.entropy
        noise_cfg = self.config.noise.gaussian

        if noise < noise_cfg.not_usable.values.entropy:
            report.add_issue(
                type=IssueType.NOISE_GAUSSIAN,
                level=IssueLevel.NOT_USABLE,
                metrics={"entropy": noise},
                recommendation=noise_cfg.not_usable.message,
            )
        elif noise < noise_cfg.high.values.entropy:
            report.add_issue(
                type=IssueType.NOISE_GAUSSIAN,
                level=IssueLevel.HIGH,
                metrics={"entropy": noise},
                recommendation=noise_cfg.high.message,
            )
        elif noise < noise_cfg.medium.values.entropy:
            report.add_issue(
                type=IssueType.NOISE_GAUSSIAN,
                level=IssueLevel.MEDIUM,
                metrics={"entropy": noise},
                recommendation=noise_cfg.medium.message,
            )
        elif noise < noise_cfg.low.values.entropy:
            report.add_issue(
                type=IssueType.NOISE_GAUSSIAN,
                level=IssueLevel.LOW,
                metrics={"entropy": noise},
                recommendation=noise_cfg.low.message,
            )

    def _check_noise_salt_pepper(self, metrics: Metrics, report: ImageQualityReport) -> None:
        """Проверяет наличие шума на изображении."""
        noise = metrics.entropy
        noise_cfg = self.config.noise.salt_pepper

        if noise < noise_cfg.not_usable.values.sharpness:
            report.add_issue(
                type=IssueType.NOISE_SALT_PEPPER,
                level=IssueLevel.NOT_USABLE,
                metrics={"entropy": noise},
                recommendation=noise_cfg.not_usable.message,
            )
        elif noise < noise_cfg.high.values.sharpness:
            report.add_issue(
                type=IssueType.NOISE_SALT_PEPPER,
                level=IssueLevel.HIGH,
                metrics={"entropy": noise},
                recommendation=noise_cfg.high.message,
            )
        elif noise < noise_cfg.medium.values.sharpness:
            report.add_issue(
                type=IssueType.NOISE_SALT_PEPPER,
                level=IssueLevel.MEDIUM,
                metrics={"entropy": noise},
                recommendation=noise_cfg.medium.message,
            )
        elif noise < noise_cfg.low.values.sharpness:
            report.add_issue(
                type=IssueType.NOISE_SALT_PEPPER,
                level=IssueLevel.LOW,
                metrics={"entropy": noise},
                recommendation=noise_cfg.low.message,
            )

    def _check_contrast(self, metrics: Metrics, report: ImageQualityReport) -> None:
        """Проверяет контраст изображения."""
        contrast = metrics.contrast
        contrast_cfg = self.config.contrast

        if contrast < contrast_cfg.not_usable.values.contrast_ratio:
            report.add_issue(
                type=IssueType.LOW_CONTRAST,
                level=IssueLevel.NOT_USABLE,
                metrics={"contrast": contrast},
                recommendation=contrast_cfg.not_usable.message,
            )
        elif contrast < contrast_cfg.high.values.contrast_ratio:
            report.add_issue(
                type=IssueType.LOW_CONTRAST,
                level=IssueLevel.HIGH,
                metrics={"contrast": contrast},
                recommendation=contrast_cfg.high.message,
            )
        elif contrast < contrast_cfg.medium.values.contrast_ratio:
            report.add_issue(
                type=IssueType.LOW_CONTRAST,
                level=IssueLevel.MEDIUM,
                metrics={"contrast": contrast},
                recommendation=contrast_cfg.medium.message,
            )
        elif contrast < contrast_cfg.low.values.contrast_ratio:
            report.add_issue(
                type=IssueType.LOW_CONTRAST,
                level=IssueLevel.LOW,
                metrics={"contrast": contrast},
                recommendation=contrast_cfg.low.message,
            )
        
        
    
    def _visualize_metrics(self, image: np.ndarray, metrics: Dict[str, float]) -> None:
        """Визуализирует метрики качества изображения."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        plt.figure(figsize=(15, 12))
        
        # Исходное изображение
        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else gray, cmap='gray')
        plt.title('Исходное изображение')
        plt.axis('off')
        
        # Гистограмма
        plt.subplot(2, 2, 2)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist_norm = hist / hist.sum()
        plt.bar(range(256), hist_norm, color='gray')
        plt.axvline(x=50, color='blue', linestyle='--', label='Dark threshold')
        plt.axvline(x=220, color='red', linestyle='--', label='Light threshold')
        plt.title(f'Гистограмма яркости\nDark: {metrics["dark_ratio"]:.3f}, Light: {metrics["light_ratio"]:.3f}, Mid: {metrics["mid_ratio"]:.3f}')
        plt.legend()
        
        # Карта резкости
        plt.subplot(2, 2, 3)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        plt.imshow(np.abs(laplacian), cmap='hot')
        plt.colorbar(label='Градиент')
        plt.title(f'Карта резкости\nSharpness: {metrics["sharpness"]:.1f}')
        plt.axis('off')
        
        # Карта энтропии
        plt.subplot(2, 2, 4)
        from skimage.filters.rank import entropy as local_entropy
        from skimage.morphology import disk
        
        # Уменьшаем размерность для ускорения вычисления энтропии
        scale_factor = max(1, int(min(gray.shape) / 500))
        small_gray = cv2.resize(gray, (gray.shape[1]//scale_factor, gray.shape[0]//scale_factor))
        
        # Вычисляем локальную энтропию с диском радиуса 5
        entropy_map = local_entropy(small_gray.astype(np.uint8), disk(5))
        plt.imshow(entropy_map, cmap='viridis')
        plt.colorbar(label='Энтропия')
        plt.title(f'Карта энтропии\nEntropy: {metrics["entropy"]:.3f}')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()




# Решение проблемы пересвета и недосвета на основе гистограммы яркости: 
# https://rockyshikoku.medium.com/opencv-is-a-great-way-to-enhance-underexposed-overexposed-too-dark-and-too-bright-images-f79c57441a8a

# convert the warped image to grayscale
# gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

# # sharpen image
# sharpen = cv2.GaussianBlur(gray, (0,0), 3)
# sharpen = cv2.addWeighted(gray, 1.5, sharpen, -0.5, 0)

# # apply adaptive threshold to get black and white effect
# thresh = cv2.adaptiveThreshold(sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)


