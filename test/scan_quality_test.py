
import sys

import numpy as np


sys.path.append('./src')  # Добавляем путь к src в sys.path
import config_loader as cfg
from common import load_document_pdf, visualize
from generate_noise import (generate_blurred_images, 
                            generate_over_exposed_images, 
                            generate_under_exposed_images, 
                            generate_salt_and_pepper_noise, 
                            generate_gaussian_noise)
from scan_quality import ImageQualityAnalyzer


def test_analizator(image: np.ndarray) -> None:
    """Тестирование класса ImageQualityAnalyzer."""
    config = cfg.load_config("./config/quality_thresholds.yaml",  cfg.ImageQualityConfig)
    analyzer = ImageQualityAnalyzer(config=config)
    report = analyzer.analyze(image, visualize=False)

    print("\n=== ОТЧЕТ О КАЧЕСТВЕ ИЗОБРАЖЕНИЯ ===")
    print(f"Размер: {report.image_size[1]}x{report.image_size[0]}")
    print("\nИзмеренные метрики:")
    for metric, value in report.metrics.items():
        print(f"  • {metric}: {value:.4f}")

    print("\nРЕКОМЕНДАЦИИ:")
    print(report.get_recommendations_summary())

def test_run_analizer_tests() -> None:
    """Запуск всех тестов."""
    print("Testing image quality metrics...")
    
    print('Generating blurred images [0, 3, 7, 11, 15]...')
    images = generate_blurred_images(load_document_pdf('./pdf/Акт сверки.pdf'))
    for i, image in enumerate(images):
        print(f"Testing image {i+1}...")
        test_analizator(image)

    # print("Generating overexposed images [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]...")
    # images = generate_over_exposed_images(load_document_pdf('./pdf/Акт сверки.pdf'))
    # for i, image in enumerate(images):
    #     print(f"Testing image {i+1}...")
    #     test_analizator(image)

    # print("Generating underexposed images [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]...")
    # images = generate_under_exposed_images(load_document_pdf('./pdf/Акт сверки.pdf'))
    # for i, image in enumerate(images):
    #     print(f"Testing image {i+1}...")
    #     test_analizator(image)

    # print('Gerating salt and pepper noise [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]...')  
    # images = generate_salt_and_pepper_noise(load_document_pdf('./pdf/Акт сверки.pdf'))
    # for i, image in enumerate(images):
    #     print(f"Testing image {i+1}...")
    #     test_analizator(image)

    # print('Gerating gaussian noise [0.0, 10, 20, 30, 40, 50]...')
    # images = generate_gaussian_noise(load_document_pdf('./pdf/Акт сверки.pdf'))
    # for i, image in enumerate(images):
    #     print(f"Testing image {i+1}...")
    #     test_analizator(image)


if __name__ == "__main__":

    image = load_document_pdf('./pdf/ЕВР-НКАЗ.pdf')
    # test_analizator(image)
    test_run_analizer_tests()

