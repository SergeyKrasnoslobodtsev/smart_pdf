import cv2
import numpy as np




def filter_small_and_isolated(mask: np.ndarray, intersec: np.ndarray, min_length:int = 120, min_intersections: int = 1):
    """
    Удаляет из mask все компоненты (линии), 
    у которых длина (число пикселей) < min_length
    или число пересечений <= min_intersections.
    """
    num_mask_labels, labels_mask = cv2.connectedComponents(mask.astype(np.uint8))
    
    for lbl in range(1, num_mask_labels): # Начинаем с 1, чтобы пропустить фон
        comp_boolean_mask = (labels_mask == lbl)
        length = int(np.count_nonzero(comp_boolean_mask))

        # Создаем маску, содержащую только точки пересечения для текущего компонента
        current_comp_intersections_map = np.zeros_like(mask, dtype=np.uint8)
        # Отмечаем пиксели, где текущий компонент (comp_boolean_mask) И есть пересечение (intersec > 0)
        current_comp_intersections_map[comp_boolean_mask & (intersec > 0)] = 255

        # Считаем количество отдельных связных областей (блобов) в этой маске пересечений
        num_intersection_blobs, _ = cv2.connectedComponents(current_comp_intersections_map)
        
        # num_intersection_blobs включает компонент фона (метка 0).
        # Таким образом, фактическое количество отдельных областей пересечений равно num_intersection_blobs - 1.
        # Если current_comp_intersections_map полностью черная (нет пересечений), num_intersection_blobs будет 1.
        # В этом случае crosses = 1 - 1 = 0, что корректно.
        crosses = num_intersection_blobs - 1
        
        if length < min_length or crosses <= min_intersections:
            mask[comp_boolean_mask] = 0 # Удаляем компонент из исходной маски
    return mask



def remove_lines_by_mask(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Удаляет из серого изображения все пиксели, где в маске mask == 255,
    заменяя их белым (255).
    mask: 8-битное, 0 — фон, 255 — линии.
    """
    # Убедимся, что оба массива одного размера и типа
    assert gray.shape == mask.shape, "gray и mask должны быть одинакового размера"

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)

    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # Инвертируем маску: линии → 0, фон → 255
    inv_mask = cv2.bitwise_not(mask)

    # Применяем побитовое И — оставляем только фон по маске
    cleaned = cv2.bitwise_and(gray, inv_mask)

    # Опционально: там, где были линии (mask==255), заливаем белым,
    # чтобы нечёткие края получили идеальный фон
    cleaned[mask == 255] = 255

    return cleaned

def detected_text(roi_image: np.ndarray, vk: int=100, hk=50) -> np.ndarray:
    # Адаптивная бинаризация (текст становится белым на чёрном фоне)
    bin_img = cv2.adaptiveThreshold(
        roi_image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15, C=10
    )
    # Склейка слов в строки
    horiz_k = cv2.getStructuringElement(cv2.MORPH_RECT, (vk, 1))
    closed_h = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, horiz_k, iterations=2)
    # Склейка строк в параграфы
    vert_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, hk))
    closed_p = cv2.morphologyEx(closed_h, cv2.MORPH_CLOSE, vert_k, iterations=2)

    return closed_p

def find_max_contours(mask: np.ndarray, max: int = 10) -> list:
        """Находит контуры таблиц на изображении."""
        contours, __ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # Удаляем контуры, которые слишком малы
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max]
        
        cont = []
        for c in contours:
            c_poly = cv2.approxPolyDP(c, 3, True)
            if len(c_poly) < 4:
                continue
            x, y, w, h = cv2.boundingRect(c_poly)
            if w < 300 or h < 50:
                continue
            cont.append((x, y, w, h))
        # Сортируем контуры по y-координате
        cont = sorted(cont, key=lambda c: c[1])
        return cont

def extract_strong_lines(mask: np.ndarray, orientation: str, length: int) -> np.ndarray:
    """
    Извлекает только настоящие линии заданной ориентации,
    убирая короткие штрихи (буквы и шум).
    orientation: 'vert' или 'horiz'
    length: минимальная длина линии в пикселях
    """
    if orientation == 'vert':
        # узкий и длинный вертикальный kernel, ширина=1, высота=length
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
    else:
        # широкая и плоская горизонтальная «щётка»
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
    # открытие: сначала эрозия (уберёт всё меньше заданной длины), потом дилатация
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return clean

def find_lines(gray: np.ndarray, scale: int = 50) -> np.ndarray:
    """Находит длинные линии на изображении."""
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
    _, img_bin = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_bin = 255-img_bin
    vertical_element, _ = create_structuring_element(img_bin, scale, type=0)
    horizontal_element, _ = create_structuring_element(img_bin, scale - 150, type=1)

    vertical_lines = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, vertical_element, iterations=2)
    horizontal_lines = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, horizontal_element, iterations=2)




    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    v_fixed = cv2.morphologyEx(vertical_lines, cv2.MORPH_CLOSE, kernel, iterations=1)
    h_fixed = cv2.morphologyEx(horizontal_lines, cv2.MORPH_CLOSE, kernel, iterations=1)
    v_fixed = cv2.dilate(v_fixed, cv2.getStructuringElement(cv2.MORPH_RECT,(1,5)), 2)
    h_fixed = cv2.dilate(h_fixed, cv2.getStructuringElement(cv2.MORPH_RECT,(5,1)), 2)
    # Рассчитываем маску пересечений
    intersec = cv2.bitwise_and(v_fixed, h_fixed)

    # Фильтруем горизонтальные линии: удаляем те, у которых < 2 пересечений (т.е. 0 или 1)
    # min_length=1, чтобы фильтрация по длине практически не влияла
    h_filtered = filter_small_and_isolated(h_fixed.copy(), intersec, min_length=1, min_intersections=1)

    # Фильтруем вертикальные линии аналогично
    v_filtered = filter_small_and_isolated(v_fixed.copy(), intersec, min_length=100, min_intersections=1)
    
    
    return h_filtered, v_filtered



def create_structuring_element(image: np.ndarray, scale: int, type: int = 0):
    """Создает структурный элемент для морфологических операций."""
    match type:
        case 0:
            size = image.shape[0] // scale
            element = cv2.getStructuringElement(cv2.MORPH_RECT, (1, size))
        case 1:
            size = image.shape[1] // scale
            element = cv2.getStructuringElement(cv2.MORPH_RECT, (size, 1))
        case _:
            raise ValueError("Ожидается 0 или 1 для типа структурного элемента")
    
    return element, size