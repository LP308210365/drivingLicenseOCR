import os

import cv2
# from cnocr import CnOcr
from paddleocr import PaddleOCR, draw_ocr




# import paddlehub as hub
import cv2






crop_img = cv2.imread("/home/DrivingLicenseOCR/app/2.png")

# ocr = hub.Module(name="chinese_ocr_db_crnn_server")
reader = PaddleOCR(lang="ch",det=False,version='ch_ppocr_server_v2.0_xx')
result = reader.ocr(crop_img)
print(result[0][1][0])