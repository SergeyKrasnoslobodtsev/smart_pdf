
from typing import Tuple
import numpy as np
from PIL import Image



def has_line(region: np.ndarray, min_length: int, axis: int = 0) -> bool:
    """
    Проверяет, есть ли в бинарной матрице region сплошной сегмент единиц вдоль заданной оси длиной >= min_length.

    Args:
        region: 2D numpy массив бинарной маски.
        min_length: минимальная длина сегмента в пикселях.
        axis: 0 для вертикального поиска (колонки), 1 для горизонтального (строки).

    Returns:
        True, если найден сегмент длиной >= min_length, иначе False.
    """
    if axis == 0:
        # поиск вдоль колонок
        for x in range(region.shape[1]):
            run = max_run = 0
            for v in region[:, x]:
                if v:
                    run += 1
                    max_run = max(max_run, run)
                else:
                    run = 0
            if max_run >= min_length:
                return True
    else:
        # поиск вдоль строк
        for y in range(region.shape[0]):
            run = max_run = 0
            for v in region[y, :]:
                if v:
                    run += 1
                    max_run = max(max_run, run)
                else:
                    run = 0
            if max_run >= min_length:
                return True
    return False



def find_line_positions(mask: np.ndarray, axis: int) -> np.ndarray:
    """
    Находит средние координаты вертикальных (axis=0) или горизонтальных (axis=1) 
    линий по бинарной маске.
    """
    # 0) приводим к булеву и проецируем по нужной оси
    if axis == 0:
        proj = np.any(mask > 0, axis=0).astype(np.uint8)   # по столбцам
    else:
        proj = np.any(mask > 0, axis=1).astype(np.uint8)   # по строкам

    # 1) находим границы сегментов (переходы 0→1 — начало, 1→0 — конец)
    dif = np.diff(proj, prepend=0, append=0)
    starts = np.where(dif ==  1)[0]
    ends   = np.where(dif == -1)[0]

    # 2) усредняем начало и конец, чтобы получить «центры» линий
    centers = ((starts + ends - 1) // 2).astype(int)
    return centers


def page_to_image(page) -> np.ndarray:
    pix = page.get_pixmap(dpi=300)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return np.array(img)


def get_pix_rect_page(page, dpi = 300) -> Tuple[float, float]:
    '''Получим коэффициент масштабирования'''
    # Проблема неверных координат https://github.com/pymupdf/PyMuPDF/issues/2868
    pix = page.get_pixmap(dpi=dpi)
    pdf_rect = page.rect
    pdf_w, pdf_h = pdf_rect.width, pdf_rect.height
    # коэффициенты преобразования в пиксели
    scale_x = pix.width  / pdf_w
    scale_y = pix.height / pdf_h
    return scale_x, scale_y