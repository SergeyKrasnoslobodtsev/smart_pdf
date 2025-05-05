import os
import sys

import numpy as np

from common import (
    load_document_pdf, 
    plot_metrics_with_thresholds, 
    extract_thresholds_from_config,
    visualize
)
from generate_noise import (
    generate_blurred_images,
    generate_over_exposed_images,
    generate_under_exposed_images,
    generate_salt_and_pepper_noise,
    generate_gaussian_noise,
    generate_contrast_images
)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import config_loader as cfg
from scan_quality import ImageQualityAnalyzer


def run_quality_test(template: np.ndarray, images: np.ndarray, analyzer: ImageQualityAnalyzer, config_block, test_name="", image_labels=None):
    """
    Запускает тест качества изображения и выводит результаты.

    Args:
        template (np.ndarray): шаблон изображения для анализа.
        images (np.ndarray): изображения для тестирования.
        analyzer (ImageQualityAnalyzer): анализатор качества изображения.
        config_block (_type_): блок конфигурации для теста.
        test_name (str, optional): Название теста (blur, gaussian ...). Defaults to "".
        image_labels (_type_, optional): Подписиграфиков . Defaults to None.
    """
    all_metrics = []
    for i, img in enumerate(images):
        report = analyzer.analyze(template, img, visualize=False)
        all_metrics.append(report.metrics)
        print(f"\n=== {test_name} | Изображение {i+1} ===")
        print(f"  • Размер: {report.image_size[1]}x{report.image_size[0]}")
        for metric, value in report.metrics.items():
            print(f"  • {metric}: {value:.4f}")
        print("\nРЕКОМЕНДАЦИИ:")
        print(report.get_recommendations_summary())
    # Формируем словарь метрик для визуализации
    metric_names = list(all_metrics[0].keys())
    metrics_dict = {name: [m[name] for m in all_metrics] for name in metric_names}
    thresholds = extract_thresholds_from_config(config_block)
    plot_metrics_with_thresholds(
        metrics=metrics_dict,
        thresholds=thresholds,
        image_labels=image_labels or [str(i+1) for i in range(len(all_metrics))]
    )


def test_blur(template, image, config: cfg.ImageQualityConfig):
    print("=== Тест размытия ===")
    images = generate_blurred_images(image)
    analyzer = ImageQualityAnalyzer(config=config)
    visualize(
        original=image,
        noise_images=images,
        indicator=["Без размытия", "Размытие 3", "Размытие 7", "Размытие 11", "Размытие 30"]
    )
    run_quality_test(
        template=template,
        images=images,
        analyzer=analyzer,
        config_block=config.blur,
        test_name="Blur"
    )
def test_document_blur(template, image, config: cfg.ImageQualityConfig):
    print("=== Тест размытия документа ===")
    analyzer = ImageQualityAnalyzer(config=config)
    run_quality_test(
        template=template,
        images=[image],
        analyzer=analyzer,
        config_block=config.blur,
        test_name="Blur"
    )


def test_noise_salt_pepper(template, image, config: cfg.ImageQualityConfig):
    print("=== Тест шум 'соль и перец' ===")
    images = generate_salt_and_pepper_noise(image)
    analyzer = ImageQualityAnalyzer(config=config)
    run_quality_test(
        template=template,
        images=images,
        analyzer=analyzer,
        config_block=config.noise.salt_pepper,
        test_name="Salt & Pepper Noise"
    )

def test_noise_gaussian(template, image, config):
    print("=== Тест гауссов шум ===")
    images = generate_gaussian_noise(image)
    analyzer = ImageQualityAnalyzer(config=config)
    run_quality_test(
        template=template,
        images=images,
        analyzer=analyzer,
        config_block=config.noise.gaussian,
        test_name="Gaussian Noise"
    )

def test_overexposure(template, image, config: cfg.ImageQualityConfig):
    print("=== Тест передержки ===")
    images = generate_over_exposed_images(image)
    analyzer = ImageQualityAnalyzer(config=config)
    visualize(
        original=image,
        noise_images=images,
        indicator=["Без передержки", "Передержка 1.5", "Передержка 2.0", "Передержка 2.5", "Передержка 3.0"]
    )
    run_quality_test(
        template=template,
        images=images,
        analyzer=analyzer,
        config_block=config.overexposure,
        test_name="Overexposure"
    )

def test_underexposure(template, image, config: cfg.ImageQualityConfig):
    print("=== Тест недодержки ===")
    images = generate_under_exposed_images(image)
    analyzer = ImageQualityAnalyzer(config=config)
    visualize(
        original=image,
        noise_images=images,
        indicator=["Без недодержки", "Недодержка 1.5", "Недодержка 2.0", "Недодержка 2.5", "Недодержка 3.0"]
    )
    run_quality_test(
        template=template,
        images=images,
        analyzer=analyzer,
        config_block=config.underexposure,
        test_name="Underexposure"
    )

def test_contrast(template, image, config: cfg.ImageQualityConfig):
    print("=== Тест контрастности ===")
    images = generate_contrast_images(image)
    analyzer = ImageQualityAnalyzer(config=config)
    visualize(
        original=image,
        noise_images=images,
        indicator=["Без контрастности", "Контраст 1.5", "Контраст 2.0", "Контраст 2.5", "Контраст 3.0"]
    )
    run_quality_test(
        template=template,
        images=images,
        analyzer=analyzer,
        config_block=config.contrast,
        test_name="Contrast"
    )

if __name__ == "__main__":
    
    config = cfg.load_config("./config/quality_thresholds.yaml", cfg.ImageQualityConfig)
    template = load_document_pdf('./pdf/Акт сверки.pdf')
    image = load_document_pdf('./pdf/ЕВР-НКАЗ.pdf')

    test_blur(template, image, config)