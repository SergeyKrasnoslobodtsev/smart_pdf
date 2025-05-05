from paddleocr import PaddleOCR
from common import load_document_pdf


ocr = PaddleOCR(use_angle_cls=True, lang='cyrillic')

image = load_document_pdf('./pdf/ЕВР-НКАЗ.pdf')
result = ocr.ocr(image, cls=True)

for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)


# draw result
result = result[0]
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]