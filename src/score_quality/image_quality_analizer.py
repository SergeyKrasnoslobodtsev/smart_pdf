from dataclasses import dataclass
import os
import sys
from typing import Tuple

import numpy as np
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/')))


import config_loader as cfg
from .report_scores import ( ReportScores,
                            IssueLevel, 
                            IssueType )  

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


    def analyze(self, image: np.ndarray) -> ReportScores:
        """ Анализирует качество изображения и возвращает отчет с выявленными проблемами.
        
        Args:
            image: Входное BGR изображение
            
        Returns:
            ReportScores: Отчет о качестве изображения с рекомендациями
        """
        template, image = self._preprocess_image(image)

        metrics = self._compute_metrics(template, image)
        
        report = ReportScores(
            metrics=metrics.__dict__, 
            image_size=image.shape[:2]
        )
        self._check_blur(metrics, report)
        self._check_overexposure(metrics, report)
        self._check_underexposure(metrics, report)
        return report
    
    def _preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Предобработка изображения перед анализом."""
        from utils import convert_to_grayscale
        from common import load_document_pdf
        from imutils import resize

        # Портретная ориентация страница 0 альбомная 1
        template = None
        if image.shape[0] > image.shape[1]:
            template = load_document_pdf('./pdf/Акт сверки.pdf')
            # template = resize(template, width=1000)
        else:
            template = load_document_pdf('./pdf/Акт сверки.pdf', page=1)
            # template = resize(template, height=1000)
        if template is None:
            raise ValueError("Шаблон не найден. Проверьте путь к файлу шаблона.")
        # Преобразование в градации серого
        image = convert_to_grayscale(image)
        template = convert_to_grayscale(template)

        
        image = resize(image, width=template.shape[1], height=template.shape[0])


        return template, image

    def _check_blur(self, metrics:Metrics, report: ReportScores) -> None:
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
    
    def _check_overexposure(self, metrics:Metrics, report: ReportScores) -> None:
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
    
    def _check_underexposure(self, metrics:Metrics, report: ReportScores) -> None:
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


    def _compute_metrics(self, template:np.ndarray, image: np.ndarray) -> Metrics:
        """Вычисляет все метрики качества изображения."""

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
     
def generate_blurred_images(image: np.ndarray):
    """Генерация тестовых изображений с различными уровнями размытия (Gaussian blur)."""
    import cv2
    blur_levels = [0, 3, 7, 11, 33]  # 0 - без размытия, далее увеличивается радиус ядра
    test_images = []
    for ksize in blur_levels:
        if ksize == 0:
            blurred_image = image.copy()
        else:
            blurred_image = cv2.GaussianBlur(image, (ksize, ksize), 0)
        test_images.append(blurred_image)
    return test_images

if __name__ == "__main__":
    from common import load_document_pdf
    from image_quality_handler import ImageQualityHandler
    from PIL import Image
    config = cfg.load_config("./config/quality_thresholds.yaml", cfg.ImageQualityConfig)
    
    image = load_document_pdf('./pdf/ЕВР-НКАЗ.pdf')
 
    image_blur = generate_blurred_images(image)[3]
    
    Image.fromarray(image_blur).show()

    analyzer = ImageQualityAnalyzer(config=config)
    handler = ImageQualityHandler(analyzer=analyzer)
    
    result_image = handler.process_image(image=image,analizer=analyzer)

    Image.fromarray(result_image).show()
    # report = analyzer.analyze(image_blur)
    
    
    # for metric, value in report.metrics.items():
    #         print(f"  • {metric}: {value:.4f}")
    # print(report.get_recommendations_summary())   
    
    # result_image = None
    # for issue in report.issues:
    #     result_image = ImageQualityHandler().process_image(image, issue)
    

    # report = analyzer.analyze(result_image)
    # Image.fromarray(result_image).show()
    # for metric, value in report.metrics.items():
    #         print(f"  • {metric}: {value:.4f}")
    # print(report.get_recommendations_summary()) 
    

    # устранение blur




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
