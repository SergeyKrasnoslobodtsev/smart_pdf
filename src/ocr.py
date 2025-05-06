import numpy as np
from paddleocr.paddleocr import PaddleOCR



class OCR:
    """OCR класс для извлечения текста из изображений с использованием PaddleOCR."""
    def __init__(self):
        self.ocr = PaddleOCR(lang='ru', show_log=False)
        
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
    from table_detector.table_detector import TableDetector, draw_table
    from score_quality.image_quality_handler import ImageQualityHandler
    from score_quality.image_quality_analizer import ImageQualityAnalyzer
    import config_loader as cfg
    ocr = OCR()
    image = load_document_pdf('./pdf/ЕВР-НКАЗ.pdf')
    config = cfg.load_config("./config/quality_thresholds.yaml", cfg.ImageQualityConfig)
    analyzer = ImageQualityAnalyzer(config=config)
    handler = ImageQualityHandler(analyzer=analyzer)
    result_image = handler.process_image(image=image,analizer=analyzer)
    detector = TableDetector(config={})
    tables = detector.detect(result_image)

    for table in tables:
        for i, row in enumerate(table.rows):
            for j, cell in enumerate(row.cells):
                x, y, w, h = cell.x, cell.y, cell.w, cell.h
                processed_image = image[y:y+h, x:x+w]
                text = ocr.extract_tesseract_text(processed_image)
                print(f"Extracted text from cell ({i}:{j}) |---| {text}")
                cell.text = text              
    draw_image = draw_table(image, tables)   
    draw_image.show()
    

    #  ocr = PaddleOCR(
    #     lang='ru',
    #     show_log=False,

    #     rec_model_dir='models/ru_mobile_v2.0_rec_infer',
    #     use_angle_cls=True,
    # )
    # result = ocr.ocr(image, cls=True)
    # for idx in range(len(result)):
    #     res = result[idx]
    #     for line in res:
    #         print(line)
    

    # result = result[0]

    # boxes = [line[0] for line in result]
    # txts = [line[1][0] for line in result]
    # scores = [line[1][1] for line in result]
    # im_show = draw_ocr(image, boxes, txts, scores, font_path='C:/Windows/Fonts/Arial.ttf')
    # im_show = Image.fromarray(im_show)
    # im_show.save('result.jpg')