import cv2
import numpy as np
from skimage import restoration, exposure
from score_quality.image_quality_analizer import ImageQualityAnalyzer
from utils import convert_to_grayscale

from .report_scores import IssueLevel, IssueType

class ImageQualityHandler:
    
    def __init__(self, analyzer=None, max_iterations=3):
        """
        Инициализация обработчика качества изображений.
        
        Args:
            analyzer: Экземпляр ImageQualityAnalyzer для анализа изображений.
            max_iterations: Максимальное количество итераций улучшения.
        """
        self.analyzer = analyzer
        self.max_iterations = max_iterations

  
        
    def process_image(self, image: np.ndarray, analizer: ImageQualityAnalyzer) -> np.ndarray:
        """ Обработка изображения на основе выявленных проблем с качеством.
        Обрабатывает размытие, переэкспонирование и недоэкспонирование.

        Args:
            image (np.ndarray): Исходное изображение.
            quality_issues (ImageIssue): Проблемы с качеством изображения.

        Returns:
            np.ndarray: Обработанное изображение.
        """
        # Убедимся, что изображение в оттенках серого
        processed_image = convert_to_grayscale(image.copy())
        report = analizer.analyze(processed_image)
        print("--- Анализ изображения ---")
        for metric, value in report.metrics.items():
            print(f"  • {metric}: {value:.4f}")
        print(report.get_recommendations_summary())
        for issue in report.issues:
            match issue.type:
                case IssueType.BLUR:
                    match issue.level:
                        case IssueLevel.NOT_USABLE:
                            return processed_image
                        case IssueLevel.HIGH:
                            processed_image = self.fix_blur_high(processed_image)
                        case IssueLevel.MEDIUM:
                            processed_image = self.fix_blur_medium(processed_image)
                        case IssueLevel.LOW:
                            processed_image = self.fix_blur_low(processed_image)
                case IssueType.OVEREXPOSURE:
                    match issue.level:
                        case IssueLevel.NOT_USABLE:
                            return processed_image
                        case IssueLevel.HIGH:
                            processed_image = self.fix_overexposure_high(processed_image)
                        case IssueLevel.MEDIUM:
                            processed_image = self.fix_overexposure_medium(processed_image)
                        case IssueLevel.LOW:
                            processed_image = self.fix_overexposure_low(processed_image)
                case IssueType.UNDEREXPOSURE:
                    match issue.level:
                        case IssueLevel.NOT_USABLE:
                            return processed_image
                        case IssueLevel.HIGH:
                            processed_image = self.fix_underexposure_high(processed_image)
                        case IssueLevel.MEDIUM:
                            processed_image = self.fix_underexposure_medium(processed_image)
                        case IssueLevel.LOW:
                            processed_image = self.fix_underexposure_low(processed_image)
        report = analizer.analyze(processed_image)
        print("--- Повторный анализ изображения ---")
        for metric, value in report.metrics.items():
            print(f"  • {metric}: {value:.4f}")
        print(report.get_recommendations_summary())     
        return processed_image

    # Методы коррекции размытия
    def fix_blur_high(self, image):
        """
        Применение деконволюции Richardson-Lucy для сильного размытия.
        """
        # Создание PSF (Point Spread Function) для деконволюции
        psf = np.ones((5, 5)) / 25
        # Применение Richardson-Lucy деконволюции
        deconvolved = restoration.richardson_lucy(image, psf, num_iter=30)
        
        # Повышение резкости после деконволюции
        kernel = np.array([[-1,-1,-1], 
                           [-1, 9,-1],
                           [-1,-1,-1]])
        
        sharpened = cv2.filter2D(deconvolved, -1, kernel)
        
        # Конвертация обратно в uint8
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def fix_blur_medium(self, image):
        """Повышает резкость изображения с помощью свёртки."""
        
        return self.unsharp_mask(image, amount=1.5, radius=2)
    
    @staticmethod
    def unsharp_mask(image: np.ndarray, amount=1.5, radius=1, threshold=0) -> np.ndarray:
        import cv2
        import numpy as np
        ksize = max(3, radius * 2 + 1)
        blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
        sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
        if threshold > 0:
            low_contrast_mask = np.abs(image.astype(np.int16) - blurred.astype(np.int16)) < threshold
            np.copyto(sharpened, image, where=low_contrast_mask)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def fix_blur_low(self, image):
        """
        Улучшение изображения с низким уровнем размытия для достижения показателя 1.
        """
        return self.unsharp_mask(image, amount=4.0, radius=2)


    # Методы коррекции переэкспонирования
    def fix_overexposure_high(self, image):
        """
        Применение тонального отображения для высокого переэкспонирования.
        """
        # Нормализация для grayscale
        img_float = image.astype(np.float32) / 255.0
        
        # Применение гамма-коррекции для восстановления деталей
        gamma = 1.5
        corrected = np.power(img_float, gamma)
        
        # Применение CLAHE для улучшения контраста
        corrected_uint8 = np.clip(corrected * 255, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(corrected_uint8)
        
        # Adaptive threshold для документов
        binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 25, 15)
        
        return binary
    
    def fix_overexposure_medium(self, image):
        """
        Коррекция уровней, уменьшение яркости и увеличение контраста в средних тонах.
        """
        # Регулировка диапазона яркости
        adjusted = exposure.rescale_intensity(image, in_range=(0, 255), out_range=(40, 220))
        
        # Увеличение контраста в средних тонах
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        adjusted = clahe.apply(adjusted.astype(np.uint8))
        
        # Повышение четкости текста
        binary = cv2.adaptiveThreshold(adjusted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 21, 8)
        
        return binary
    
    def fix_overexposure_low(self, image):
        """
        Применение незначительного снижения яркости для легкого переэкспонирования.
        """
        # Снижение яркости слегка
        adjusted = cv2.multiply(image.astype(float), 0.9)
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        
        # Повышение контраста
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(adjusted)
        
        # Adaptive threshold для четкости текста
        binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 19, 5)
        
        return image

    # Методы коррекции недоэкспонирования
    def fix_underexposure_high(self, image):
        """
        Применение локальной адаптивной коррекции яркости для сильного недоэкспонирования.
        """
        # Применение CLAHE с повышенными параметрами
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        
        # Увеличение яркости
        brightened = cv2.add(enhanced, 40)
        brightened = np.clip(brightened, 0, 255).astype(np.uint8)
        
        # Adaptive threshold для документов
        binary = cv2.adaptiveThreshold(brightened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 23, 10)
        
        return binary
    
    def fix_underexposure_medium(self, image):
        """
        Применение CLAHE с clipLimit=2.0, tileGridSize=(8,8).
        """
        # Применение CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        
        # Увеличение яркости
        brightened = cv2.add(enhanced, 25)
        brightened = np.clip(brightened, 0, 255).astype(np.uint8)
        
        # Adaptive threshold для документов
        binary = cv2.adaptiveThreshold(brightened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 21, 7)
        
        return binary
    
    def fix_underexposure_low(self, image):
        """
        Увеличение яркости на 10-15% и увеличение контраста в светах.
        """
        # Увеличение яркости на 12%
        brightened = cv2.add(image, 30)
        
        # Улучшение контраста в светах
        gamma = 0.9  # Gamma < 1 увеличивает контраст в светах
        lookup_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        adjusted = cv2.LUT(brightened, lookup_table)
        
        # Adaptive threshold для четкости текста
        binary = cv2.adaptiveThreshold(adjusted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 17, 5)
        
        return binary