from paddleocr import PaddleOCR
import time

ocr = PaddleOCR(use_angle_cls=True, lang='en')

start_time = time.time()
result = ocr.ocr('images/Hearts.png', cls=True)

print('time', time.time() - start_time)
print(' '.join([line[1][0] for line in result[0]]))