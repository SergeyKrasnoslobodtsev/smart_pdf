from dataclasses import dataclass

import numpy as np
import cv2

import config_loader as cfg
from image_quality_report import (ImageQualityReport,
                                  IssueLevel, 
                                  IssueType)  

@dataclass
class Metrics:
    """Метрики качества изображения."""
    blurredness: float
    luminosity: float
    contrast: float


# https://medium.com/engineering-housing/image-scoring-allocating-percentage-score-to-images-for-their-quality-6169abbf850e
# Немного доработанный подход к оценке качества изображения
class ImageQualityAnalyzer:
    """Анализатор качества изображений с выявлением проблем и рекомендаций."""
    
    def __init__(self, config: cfg.ImageQualityConfig):
        self.config = config


    def analyze(self, template: np.ndarray, image: np.ndarray, visualize: bool = False) -> ImageQualityReport:
        """
        Анализирует качество изображения и возвращает отчет с выявленными проблемами.
        
        Args:
            image: Входное BGR изображение
            visualize: Флаг для визуализации результатов анализа
            
        Returns:
            ImageQualityReport: Отчет о качестве изображения с рекомендациями
        """
        # Вычисление всех метрик качества изображения
        metrics = self._compute_metrics(template, image)
        
        report = ImageQualityReport(
            metrics=metrics.__dict__, 
            image_size=image.shape[:2]
        )
        self._check_blur(metrics, report)
        self._check_overexposure(metrics, report)
        self._check_underexposure(metrics, report)
        return report
    
    def _check_blur(self, metrics:Metrics, report: ImageQualityReport) -> None:
        """Проверяет размытие изображения."""
        blur_cfg = self.config.blur

        params = [
            (blur_cfg.not_usable, IssueLevel.NOT_USABLE),
            (blur_cfg.high, IssueLevel.HIGH),
            (blur_cfg.medium, IssueLevel.MEDIUM),
            (blur_cfg.low, IssueLevel.LOW),
        ]
        
        for (config_item, level) in params:
            if metrics.blurredness < config_item.values.score_blurredness:
                report.add_issue(
                    type=IssueType.BLUR,
                    level=level,
                    recommendation=config_item.message,
                )
                break
    
    def _check_overexposure(self, metrics:Metrics, report: ImageQualityReport) -> None:
        """Проверяет контраст изображения."""
        overexposure_cfg = self.config.overexposure

        params = [
            (overexposure_cfg.not_usable, IssueLevel.NOT_USABLE),
            (overexposure_cfg.high, IssueLevel.HIGH),
            (overexposure_cfg.medium, IssueLevel.MEDIUM),
            (overexposure_cfg.low, IssueLevel.LOW),
        ]
        
        for (config_item, level) in params:
            if metrics.contrast < config_item.values.score_contrast and metrics.luminosity > config_item.values.score_luminosity:
                report.add_issue(
                    type=IssueType.LOW_CONTRAST,
                    level=level,
                    recommendation=config_item.message,
                )
                break
    
    def _check_underexposure(self, metrics:Metrics, report: ImageQualityReport) -> None:
        """Проверяет контраст изображения."""
        underexposure_cfg = self.config.underexposure

        params = [
            (underexposure_cfg.not_usable, IssueLevel.NOT_USABLE),
            (underexposure_cfg.high, IssueLevel.HIGH),
            (underexposure_cfg.medium, IssueLevel.MEDIUM),
            (underexposure_cfg.low, IssueLevel.LOW),
        ]
        
        for (config_item, level) in params:
            if metrics.contrast < config_item.values.score_contrast and metrics.luminosity < config_item.values.score_luminosity:
                report.add_issue(
                    type=IssueType.LOW_CONTRAST,
                    level=level,
                    recommendation=config_item.message,
                )
                break

    def _is_grayscale(self, image: np.ndarray) -> bool:
        """Проверяет, является ли изображение градациями серого."""
        return len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1)
    
    def _convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Конвертирует изображение в градации серого."""
        if self._is_grayscale(image):
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _roteate_image(self, image: np.ndarray) -> np.ndarray:
        """Поворачивает изображение на 90 градусов по часовой стрелке."""
        if image.shape[0] > image.shape[1]:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        else:
            return image
    
    def _resize_image(self, image: np.ndarray, width: int = 500, height: int = 1000) -> np.ndarray:
        """Изменяет размер изображения с заданным коэффициентом масштабирования."""
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    def _compute_metrics(self, template:np.ndarray, image: np.ndarray) -> Metrics:
        """Вычисляет все метрики качества изображения."""
        template = self._convert_to_grayscale(template)
        image = self._convert_to_grayscale(image)

        if template.shape != image.shape:
            image = self._resize_image(image, width=template.shape[1], height=template.shape[0])

        score_blurredness = self._compute_blurredness(template, image)
        score_luminosity = self._compute_luminosity(template, image)
        score_contrast = self._compute_contrast(template, image)

        return Metrics(
            blurredness=score_blurredness,
            luminosity=score_luminosity,
            contrast=score_contrast,
            )
    
    def _compute_blurredness(self, template: np.ndarray, image: np.ndarray) -> float:
        """Вычисляет относительную четкость изображения по сравнению с шаблоном.
        
        Args:
            template (np.ndarray): Эталонное изображение (шаблон).
            image (np.ndarray): Входное изображение для анализа.
        
        Returns:
            float: Оценка четкости изображения от 0 до 1, где 1 - четкое, 0 - размытое.
        """
        # Вычисляем лапласиан для обоих изображений для определения резкости
        laplacian_template = cv2.Laplacian(template, cv2.CV_64F)
        laplacian_image = cv2.Laplacian(image, cv2.CV_64F)
        
        # Вычисляем дисперсию лапласиана - показатель четкости
        variance_template = laplacian_template.var()
        variance_image = laplacian_image.var()
        
        # Определяем относительную резкость
        if variance_template > 0:
            # Используем логарифмическое соотношение для уменьшения влияния больших значений
            blurredness_ratio = np.log1p(variance_image) / np.log1p(variance_template)
        else:
            blurredness_ratio = 0.0  # Если шаблон не имеет резкости, возвращаем 0
        
        # Ограничиваем значение в диапазоне [0, 1]
        return min(max(blurredness_ratio, 0.0), 1.0)

    def _compute_luminosity(self, template: np.ndarray, image: np.ndarray) -> float:
        """Вычисляет яркость изображения от 0 до 1, где 0 - темное, 1 - светлое.
        
        Args:
            template (np.ndarray): Эталонное изображение (шаблон).
            image (np.ndarray): Входное изображение для сравнения.
        
        Returns:
            float: Оценка яркости изображения от 0 до 1.
        """
        # Вычисляем среднюю яркость для эталона и изображения
        template_luminosity = np.mean(template) / 255.0
        image_luminosity = np.mean(image) / 255.0
        
        # Используем логарифмическое преобразование для уменьшения влияния больших значений
        if template_luminosity > 0:
            luminosity_ratio = np.log1p(image_luminosity) / np.log1p(template_luminosity)
        else:
            luminosity_ratio = 0.0
        
        # Ограничиваем значение в диапазоне [0, 1]
        return min(max(luminosity_ratio, 0.0), 1.0)
    
    def _compute_contrast(self, template: np.ndarray, image: np.ndarray) -> float:
        """Вычисляет контраст изображения относительно шаблона с использованием логарифмической шкалы.
        
        Args:
            template (np.ndarray): Эталонное изображение (шаблон).
            image (np.ndarray): Входное изображение для сравнения.
        
        Returns:
            float: Нормализованный контраст изображения относительно шаблона.
        """
        # Вычисляем стандартное отклонение (контраст) для шаблона и изображения
        template_contrast = np.std(template)
        image_contrast = np.std(image)
        
        # Используем логарифмическую шкалу для нормализации
        if template_contrast > 0:
            normalized_contrast = np.log1p(image_contrast) / np.log1p(template_contrast)
        else:
            normalized_contrast = 0.0  # Если контраст шаблона равен 0, возвращаем 0
        
        # Ограничиваем значение в диапазоне [0, 1]
        return min(max(normalized_contrast, 0.0), 1.0)
     


# Решение проблемы пересвета и недосвета на основе гистограммы яркости: 
# https://rockyshikoku.medium.com/opencv-is-a-great-way-to-enhance-underexposed-overexposed-too-dark-and-too-bright-images-f79c57441a8a

# convert the warped image to grayscale
# gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

# # sharpen image
# sharpen = cv2.GaussianBlur(gray, (0,0), 3)
# sharpen = cv2.addWeighted(gray, 1.5, sharpen, -0.5, 0)

# # apply adaptive threshold to get black and white effect
# thresh = cv2.adaptiveThreshold(sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)

# TODO: https://medium.com/analytics-vidhya/enhance-a-document-scan-using-python-and-opencv-9934a0c2da3d
