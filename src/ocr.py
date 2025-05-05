import numpy as np
from paddleocr import PaddleOCR

class OCR:
    """OCR класс для извлечения текста из изображений с использованием PaddleOCR."""
    def __init__(self):
        self.ocr = PaddleOCR(
            use_angle_cls=True, 
            lang='ru',  # Use Russian specifically
            rec_model_dir='ru_PP-OCRv3_rec_infer',  # Specify model if you have it
            det_db_thresh=0.3,  # Lower threshold for detection
            rec_char_dict_path=None,  # Default dictionary
            rec_batch_num=6,
            det_limit_side_len=960,
            det_limit_type='max',
            show_log=False
        )
        
    def extract_paddle_text(self, image:np.ndarray) -> str:
        """Извлечение текста из изображения."""
        result = self.ocr.ocr(image, cls=False)
        result = result[0]
        if not result:
            return ""
        txts = [line[1][0] for line in result]

        return txts
    def extract_tesseract_text(self, image:np.ndarray) -> str:
        """Извлечение текста из изображения с использованием Tesseract."""
        import pytesseract

        cfg = r'--oem 1 --psm 4 -l rus+eng'

        text = pytesseract.image_to_string(image, config=cfg)
        return text

if __name__ == "__main__":
    from common import load_document_pdf
    from table_detector import TableDetector, draw_table
    from PIL import Image
    ocr = OCR()
    image = load_document_pdf('./pdf/ЕВР-НКАЗ.pdf')
    detector = TableDetector(config={})
    tables = detector.detect(image)

    for table in tables:
        for i, row in enumerate(table.rows):
            for j, cell in enumerate(row.cells):
                x, y, w, h = cell.x, cell.y, cell.w, cell.h
                processed_image = image[y:y+h, x:x+w]
                text = ocr.extract_tesseract_text(processed_image)
                print(f"Extracted text from cell ({i}:{j}) |---| {text}")
                cell.text = text              
    draw_table(image, tables)   
    
    image = Image.fromarray(image)
    image.show()
    