import cv2

def apply_adaptive_filter(image, block_size, c):
    """
    Применяет адаптивный фильтр к изображению.
    
    Args:
        image (np.ndarray): Исходное изображение.
        block_size (int): Размер блока для адаптивной бинаризации (должен быть нечетным).
        c (int): Константа, вычитаемая из среднего значения.
    
    Returns:
        np.ndarray: Изображение после применения адаптивного фильтра.
    """
    if block_size % 2 == 0:
        block_size += 1  # Блок должен быть нечетным
    

    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c
    )

def on_trackbar_change(_):
    """
    Обработчик изменения трекбаров. Применяет фильтр с текущими параметрами.
    """
    block_size = cv2.getTrackbarPos("Block Size", "Adaptive Filter")
    c = cv2.getTrackbarPos("C", "Adaptive Filter")
    
    # Минимальный размер блока должен быть >= 3 и нечетным
    if block_size < 3:
        block_size = 3
    if block_size % 2 == 0:
        block_size += 1
    
    filtered_image = apply_adaptive_filter(gray_image, block_size, c)
    cv2.imshow("Adaptive Filter", filtered_image)

# Загрузка изображения
image_path = "./pdf/ASR_SDD_2_quarter_2024.png"  # Укажите путь к вашему изображению
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Ошибка: изображение не найдено!")
    exit()

gray_image = image.copy()

# размываем изображение
gray_image = cv2.medianBlur(gray_image, 5)

# Создаем окно для отображения
cv2.namedWindow("Adaptive Filter")
#изменим размер окна
cv2.resizeWindow("Adaptive Filter", 800, 600)
# Добавляем трекбары
cv2.createTrackbar("Block Size", "Adaptive Filter", 11, 300, on_trackbar_change)  # Начальное значение 11
cv2.createTrackbar("C", "Adaptive Filter", 2, 300, on_trackbar_change)  # Начальное значение 2

# Отображаем исходное изображение
cv2.imshow("Adaptive Filter", gray_image)

# Запускаем обработчик для начальных значений
on_trackbar_change(0)

# Ожидаем нажатия клавиши для выхода
cv2.waitKey(0)
cv2.destroyAllWindows()