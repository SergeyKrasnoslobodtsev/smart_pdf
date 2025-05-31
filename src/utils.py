from PIL import Image, ImageDraw
from PIL import ImageFont


from .PDFExtractor.base_extractor import Cell, InsertionPosition
    
def draw_text_to_cell(image: Image.Image, cell: Cell, new_text: str, font_size: int=24) -> Image.Image:
    '''Впишем текст в ячейку (пока в пустую)'''
    x1, y1, x2, y2 = cell.bbox.coords
    
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    if cell.has_text:
        for blob in cell.blobs:
            mid_y = blob.y1 + (blob.y2 - blob.y1) // 2
            draw.line([(blob.x1, mid_y), (blob.x2, mid_y)], fill="black", width=2)

        insertion_area = cell.get_insertion_area(InsertionPosition.BOTTOM, min_height=font_size + 2, padding=2) # +4 для небольшого запаса по высоте
        if not insertion_area:
            insertion_area = cell.get_insertion_area(InsertionPosition.TOP, min_height=font_size + 2, padding=2)

        if insertion_area:
            # Рисуем новый текст в найденной области
            text_bbox = draw.textbbox((0, 0), new_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            area_width = insertion_area.x2 - insertion_area.x1
            area_height = insertion_area.y2 - insertion_area.y1

            # Проверяем, помещается ли текст (хотя бы по высоте, min_height уже учтено)
            if text_width <= area_width and text_height <= area_height:
                x_text = insertion_area.x1 + (area_width - text_width) // 2
                y_text = insertion_area.y1 + (area_height - text_height) // 2
                draw.text((x_text, y_text), new_text, fill="black", font=font)
            else:
                # Текст может не поместиться, рисуем в верхнем левом углу области с отсечением (или можно добавить логику переноса/масштабирования)
                # Для простоты пока просто рисуем в начале области, если он слишком большой
                draw.text((insertion_area.x1, insertion_area.y1), new_text, fill="black", font=font)
                print(f"Предупреждение: Новый текст может не полностью поместиться в ячейку {cell.row},{cell.col} в выделенной области.")
        else:
            # Место для вставки нового текста не найдено
            print(f"Предупреждение: Не удалось найти место для вставки нового текста в ячейке {cell.row},{cell.col}.")
            # Можно добавить логику, что делать в этом случае, например, не рисовать новый текст или вызывать ошибку


    else:
        # Рисуем новый текст по центру ячейки
        text_bbox = draw.textbbox((0, 0), new_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Определяем координаты для центрирования текста в ячейке
        cell_width = x2 - x1
        cell_height = y2 - y1

        x_center = x1 + (cell_width - text_width) // 2
        y_center = y1 + (cell_height - text_height) // 2
        # Рисуем текст
        draw.text((x_center, y_center), new_text, fill="black", font=font)

    return image
