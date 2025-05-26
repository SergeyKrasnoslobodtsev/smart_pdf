

from .PDFExtractor.base_extractor import Document
from .utils import draw_text_to_cell

from .io_stream import convert_to_pil
from .io_stream import convert_to_bytes



def update_cell_text_in_document(
    document: Document, 
    page_num: int, 
    table_num: int, 
    row: int, 
    col: int, 
    new_text: str
) -> Document:
    """
    Обновляет текст в указанной ячейке и возвращает новый документ.
    
    Args:
        document: Исходный документ
        page_num: Номер страницы (начиная с 0)
        table_num: Номер таблицы на странице (начиная с 0)
        row: Номер строки в таблице (начиная с 0)
        col: Номер колонки в таблице (начиная с 0)
        new_text: Новый текст для ячейки
        
    Returns:
        Document: Обновленный документ
    """
    # Проверка корректности входных данных
    if page_num < 0 or page_num >= document.page_count:
        raise ValueError(f"Номер страницы {page_num} вне диапазона [0, {document.page_count-1}]")
    
    page = document.pages[page_num]
    
    if table_num < 0 or table_num >= len(page.tables):
        raise ValueError(f"Номер таблицы {table_num} вне диапазона [0, {len(page.tables)-1}]")
    
    table = page.tables[table_num]
    
    # Находим нужную ячейку
    target_cell = None
    for cell in table.cells:
        if cell.row == row and cell.col == col:
            target_cell = cell
            break
    
    if not target_cell:
        raise ValueError(f"Ячейка с координатами row={row}, col={col} не найдена в таблице {table_num}")
    
    # Конвертируем PDF в список изображений
    images = convert_to_pil(document.pdf_bytes)
    

    updated_image = draw_text_to_cell(
        image=images[page_num], 
        cell=target_cell, 
        new_text=new_text,
        font_size=int(table.average_blob_height)
    )
    updated_image.show()
    images[page_num] = updated_image
    
    # Обновляем текст в структуре документа
    target_cell.text = new_text
    
    # Конвертируем список изображений обратно в PDF байты
    new_pdf_bytes = convert_to_bytes(images)
    
    # Создаем новый документ с обновленными байтами
    new_document = Document(
        pdf_bytes=new_pdf_bytes,
        pages=document.pages,
        page_count=document.page_count
    )
    
    return new_document