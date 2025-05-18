import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont



def image_processing(image: np.ndarray) -> np.ndarray:
    """Обрабатывает изображение для улучшения качества."""
    blur = cv2.medianBlur(image, 3)
    # binary = binarize(blur)
    # gaps = bridge_gaps(binary)

    return blur

def binarize(gray: np.ndarray) -> np.ndarray:
    from skimage.filters import threshold_sauvola
    # Sauvola устойчив к неравномерной засветке                         
    thresh_sauvola = threshold_sauvola(gray, 35, 1.8)
    binary = (gray > thresh_sauvola).astype(np.uint8) * 255
    return binary

def bridge_gaps(binary: np.ndarray, max_gap: int = 4) -> np.ndarray:
    """
    Соединяем мелкие разрывы между штрихами букв.
    binary – 0/255 uint8, текст = 255
    max_gap – допустимый размер «дырки» пикселов
    """
    from skimage.morphology import skeletonize
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)


    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max_gap, 1))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, horiz_kernel)

    skel = skeletonize((closed // 255).astype(bool))
    if not skel.any():
        return closed                     

    dist = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
    radius = np.median(dist[skel])
    if np.isnan(radius) or radius < 1:
        return closed

    stroke_w = int(round(radius * 2))
    k = max(1, min(stroke_w // 2, min(binary.shape)//20))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    thick = cv2.dilate(closed, kernel, 1)

    return thick

def morphological_operations(image: np.ndarray, size, iterations: int = 0) -> np.ndarray:
    """Применяет морфологические операции к изображению."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return image  

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

def find_lines(gray: np.ndarray, scale: int = 50) -> np.ndarray:
    """Находит длинные линии на изображении."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, img_bin = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_bin = 255-img_bin
    vertical_element, _ = create_structuring_element(img_bin, scale + 40, type=0)
    horizontal_element, _ = create_structuring_element(img_bin, scale, type=1)

    vertical_lines = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, vertical_element, iterations=1)
    horizontal_lines = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, horizontal_element, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    v_fixed = cv2.morphologyEx(vertical_lines, cv2.MORPH_CLOSE, kernel, iterations=2)
    h_fixed = cv2.morphologyEx(horizontal_lines, cv2.MORPH_CLOSE, kernel, iterations=2)
    v_fixed = cv2.dilate(v_fixed, cv2.getStructuringElement(cv2.MORPH_RECT,(1,5)), 1)
    h_fixed = cv2.dilate(h_fixed, cv2.getStructuringElement(cv2.MORPH_RECT,(5,1)), 1)

    return h_fixed, v_fixed

def find_table_contours(mask: np.ndarray) -> list:
    """Находит контуры таблиц на изображении."""
    contours, __ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # Удаляем контуры, которые слишком малы
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    cont = []
    for c in contours:
        c_poly = cv2.approxPolyDP(c, 3, True)
        if len(c_poly) < 4:
            continue
        x, y, w, h = cv2.boundingRect(c_poly)
        if w < 100 or h < 100:
            continue
        cont.append((x, y, w, h))
    # Сортируем контуры по y-координате
    cont = sorted(cont, key=lambda c: c[1])
    return cont

def find_joints(table_bbox, vertical_lines: np.ndarray, horizontal_lines: np.ndarray):
    """Находит пересечения вертикальных и горизонтальных линий внутри таблицы."""
    x, y, w, h = table_bbox
    mask = np.multiply(vertical_lines, horizontal_lines)
    roi = mask[y:y+h, x:x+w]

    contours, __ = cv2.findContours(
        roi.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    coords = []
    for c in contours:
        rx, ry, rw, rh = cv2.boundingRect(c)
        cx = x + (2*rx + rw)//2
        cy = y + (2*ry + rh)//2
        coords.append((cx, cy))
    
    return coords

def has_inner_line(x1, y1, x2, y2, vertical_lines, horizontal_lines, min_line_pixels=13, margin=0.5):
    h, w = y2 - y1, x2 - x1

    inner_x1 = int(x1 + w * margin)
    inner_x2 = int(x2 - w * margin)
    inner_y1 = int(y1 + h * margin)
    inner_y2 = int(y2 - h * margin)

    if inner_x2 <= inner_x1 or inner_y2 <= inner_y1:
        return False  # ячейка слишком мала

    # Проверка вертикальных линий
    v_region = vertical_lines[inner_y1:inner_y2, inner_x1:inner_x2]
    v_proj = np.sum(v_region, axis=0)
    v_peaks = np.count_nonzero(v_proj > min_line_pixels)

    # Проверка горизонтальных линий
    h_region = horizontal_lines[inner_y1:inner_y2, inner_x1:inner_x2]
    h_proj = np.sum(h_region, axis=1)
    h_peaks = np.count_nonzero(h_proj > min_line_pixels)

    return v_peaks > 0 or h_peaks > 0


def filtered_cells(bbox: tuple, joints: list, v_lines, h_lines):
    """
    Получает список ячеек таблицы на основе пересечений линий.
    """
    x_coords = sorted({pt[0] for pt in joints})
    y_coords = sorted({pt[1] for pt in joints})

    rows = []
    for i in range(len(y_coords) - 1):
        row_cells = []
        for j in range(len(x_coords) - 1):
            x1, y1 = x_coords[j], y_coords[i]
            x2, y2 = x_coords[j+1], y_coords[i+1]

            if x1 < bbox[0] or y1 < bbox[1] or x2 > bbox[0] + bbox[2] or y2 > bbox[1] + bbox[3]:
                continue
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue

            # Пропуск объединённых ячеек (внутренние границы отсутствуют)
            if not has_inner_line(x1, y1, x2, y2, v_lines, h_lines):
                continue

            w, h = x2 - x1, y2 - y1
            row_cells.append((x1, y1, w, h))
        if row_cells:
            rows.append(row_cells)
    return rows

def range_cells(cells: list):
    range_cells = []
    for i in range(len(cells)):
        for j in range(len(cells[i])):
            x, y, w, h = cells[i][j]
            cells[i][j] = {
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'row': i,
                'col': j
            }
            range_cells.append(cells[i][j])
    return range_cells

def draw_table_and_cells(image, tables, cells):
    """Рисует таблицы и ячейки на копии изображения."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype("arial.ttf", 24)

    # Рисуем таблицы
    for i, table in enumerate(tables):
        x, y, w, h = table
        draw.rectangle([(x, y), (x + w, y + h)], outline='blue', width=3)
        position = (x + 5, y - 35)
        text = f"Table: {i}"
        bbox = draw.textbbox(position, text, font=font)
        padded_bbox = (bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5)
        draw.rectangle(padded_bbox, fill="blue")
        draw.text(position, text, font=font, fill='white')

    # Рисуем ячейки
    for cell in cells:
        x = cell['x']
        y = cell['y']
        w = cell['w']
        h = cell['h']
        row = cell['row']
        col = cell['col']
        draw.rectangle([(x, y), (x + w, y + h)], outline='red', width=2)
        position = (x + 5, y + 5)
        text = f"R{row}:C{col}"
        bbox = draw.textbbox(position, text, font=font)
        padded_bbox = (bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5)
        draw.rectangle(padded_bbox, fill="red")
        draw.text(position, text, font=font, fill='white')
       
    
        

    return pil_image


def draw_joints(image, joints):
    """Рисует точки пересечения на изображении."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype("arial.ttf", 24)

    for i, (x, y) in enumerate(joints):
        draw.ellipse([(x-5, y-5), (x+5, y+5)], fill='green')
        position = (x + 10, y + 10)
        text = str(x) + ":" + str(y)
        bbox = draw.textbbox(position, text, font=font)
        padded_bbox = (bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5)
        draw.rectangle(padded_bbox, fill="green")
        draw.text(position, text, font=font, fill='white')
    return pil_image



# from common import load_document_pdf

# img = load_document_pdf('./pdf/АС Евросибэнерго-НКАЗ.pdf', page=0)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_bin = image_processing(img_gray)
# v_lines, h_lines = find_lines(img_bin, scale=50)
# mask = v_lines + h_lines


# Image.fromarray(mask).show()
# tables = find_table_contours(mask)

# for bbox in tables:
#     joints = find_joints(bbox, v_lines, h_lines)
   

#     # draw_joints(img, joints).show()
#     # Получаем первичный список ячеек
#     fc = filtered_cells(bbox, joints, v_lines, h_lines)
#     cells = range_cells(fc)
    

# result_img = draw_table_and_cells(img, tables, cells)
# result_img.show()
