from typing import List
import matplotlib.pyplot as plt
import numpy as np

from dataclasses import asdict


def load_document_pdf(file_path: str, page:int = 0) -> np.ndarray:
    """Загрузка изображения из PDF файла с нормализацией в диапазон [0, 1]."""
    from pdf2image import convert_from_path
    images = convert_from_path(file_path, dpi=400)
    return np.array(images[page])


def visualize(original: np.ndarray, noise_images: List[np.ndarray],  indicator:List[str]) -> None:
    """Визуализация оригинала и шумных изображений с метрикой."""
    
    
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

def extract_thresholds_from_config(levels_config):
    """
    Универсально извлекает пороги для всех метрик из блока конфигурации (например, config.blur, config.noise.salt_pepper и т.д.).
    Возвращает словарь: {метрика: [not_usable, high, medium, low]}
    """
    level_names = ["not_usable", "high", "medium", "low"]
    result = {}
    for level_name in level_names:
        level = getattr(levels_config, level_name, None)
        if level is None or not hasattr(level, 'values'):
            continue
        values = asdict(level.values)
        for k, v in values.items():
            result.setdefault(k, []).append(v)
    return result


def plot_metrics_with_thresholds(
    metrics: dict,
    thresholds: dict,
    image_labels: list = None,
    figsize=(6, 4),
    save_path: str = None,
):
    """
    Для каждой метрики строит отдельный subplot в сетке с максимум 3 графиками в строке.
    :param metrics: dict, ключ — метрика, значение — список значений по изображениям
    :param thresholds: dict, ключ — метрика, значение — список порогов (not_usable, high, medium, low)
    :param image_labels: подписи по оси X
    :param figsize: размер одного графика
    :param save_path: путь для сохранения графика (если нужно)
    """
    import matplotlib.pyplot as plt
    import math
    metric_names = list(metrics.keys())
    n_metrics = len(metric_names)
    ncols = 3
    nrows = math.ceil(n_metrics / ncols)
    if image_labels is None:
        image_labels = list(range(1, len(next(iter(metrics.values()))) + 1))
    level_labels = ["not_usable", "high", "medium", "low"]
    level_colors = ["black", "red", "orange", "green"]
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows))
    axes = axes.flatten() if n_metrics > 1 else [axes]
    for idx, metric in enumerate(metric_names):
        ax = axes[idx]
        values = metrics[metric]
        ax.plot(image_labels, values, marker='o', label=f"{metric}")
        if metric in thresholds:
            for i, thr in enumerate(thresholds[metric]):
                color = level_colors[i % len(level_colors)]
                label = f"{level_labels[i]}"
                ax.hlines(thr, image_labels[0], image_labels[-1],
                          linestyles='dashed', colors=color, label=label)
        ax.set_xlabel("Изображения")
        ax.set_ylabel(f"{metric}")
        ax.set_title(f"{metric}")
        ax.legend()
        ax.grid(True)
    # Отключаем лишние пустые оси
    for j in range(idx+1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()