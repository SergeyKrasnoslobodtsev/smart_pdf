from typing import List
from io import BytesIO
from PIL import Image
import pymupdf

_DEFAULT_DPI = 300

def convert_to_pil(pdf_bytes: bytes) -> List[Image.Image]:
    '''Преобразуем pdf байты в список изображений'''
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page in doc:
        pix = page.get_pixmap(dpi=_DEFAULT_DPI)
        img = pix.pil_image()
        pages.append(img)
    return pages

def convert_to_bytes(images: List[Image.Image]) -> bytes:
    '''
    Преобразуем список изображений обратно в PDF байты.
    Каждое изображение помещается на новую страницу стандартного размера A4
    (210мм x 297мм) с указанной ориентацией.
    Изображение масштабируется для заполнения страницы с сохранением пропорций и центрируется.
    
    Args:
        images: Список изображений PIL.Image.
        
    Returns:
        bytes: PDF документ в виде байтов.
    '''
    if not images:
        raise ValueError("Список изображений пуст")

    # Размеры A4 в пунктах (1 пункт = 1/72 дюйма)
    a4_size_pt = pymupdf.paper_size("a4")  # Возвращает (ширина, высота) для A4 портретной ориентации



    pdf_doc = pymupdf.open()  # Создаем новый пустой PDF документ

    for img_pil in images:
        # Создаем новую страницу с заданными размерами A4 и ориентацией
        if img_pil.width > img_pil.height:
            page_width_pt, page_height_pt = a4_size_pt[1], a4_size_pt[0]
        else:
            page_width_pt, page_height_pt = a4_size_pt[0], a4_size_pt[1]
        
        page = pdf_doc.new_page(width=page_width_pt, height=page_height_pt)

        img_w_px = img_pil.width
        img_h_px = img_pil.height

        if img_w_px == 0 or img_h_px == 0:
            # Пропускаем пустое изображение, страница A4 останется пустой
            continue

        # Рассчитываем прямоугольник для вставки изображения на страницу PDF,
        # сохраняя пропорции изображения и центрируя его.
        
        # Масштабные коэффициенты по ширине и высоте
        scale_factor_w = page_width_pt / img_w_px
        scale_factor_h = page_height_pt / img_h_px
        
        # Используем наименьший коэффициент, чтобы изображение полностью поместилось
        scale_factor = min(scale_factor_w, scale_factor_h)
        
        # Размеры отрендеренного изображения на странице PDF
        render_w = img_w_px * scale_factor
        render_h = img_h_px * scale_factor
        
        # Смещения для центрирования изображения
        x_offset = (page_width_pt - render_w) / 2
        y_offset = (page_height_pt - render_h) / 2
        
        # Прямоугольник для вставки изображения на PDF странице
        image_rect_on_page = pymupdf.Rect(x_offset, y_offset, x_offset + render_w, y_offset + render_h)

        # Конвертируем PIL Image в байты PNG
        img_byte_io = BytesIO()
        img_pil.save(img_byte_io, format="JPEG", quality=85)
        img_byte_io.seek(0)
        
        # Вставляем изображение в рассчитанный прямоугольник
        page.insert_image(image_rect_on_page, stream=img_byte_io)

    # Сохраняем PDF в буфер BytesIO
    output_pdf_buffer = BytesIO()
    pdf_doc.save(output_pdf_buffer)
    pdf_bytes_result = output_pdf_buffer.getvalue()
    
    output_pdf_buffer.close() # Закрываем буфер
    pdf_doc.close()           # Закрываем документ PDF
    
    return pdf_bytes_result