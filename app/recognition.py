import os
import base64

import cv2
import numpy as np
import uvicorn

from fastapi import FastAPI
from starlette.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import paddlehub as hub
from paddleocr import PaddleOCR
from cropper import Cropper
from detector import Detector





from app.utils import Item

app = FastAPI()

dir_path = os.path.dirname(os.path.realpath(__file__))
static_path = os.path.join(dir_path, 'static')

app.mount('/static', StaticFiles(directory=static_path), name='static')
templates = Jinja2Templates(directory=os.path.join(dir_path, 'templates'))


cropper = Cropper()

detector = Detector()

ocr = hub.Module(name="chinese_ocr_db_crnn_server")
# ocr = PaddleOCR(det = False,lang="ch")

@app.post('/extract')
def extract(item: Item):

    # image = cv2.imdecode(np.fromstring(item.base64_img, np.uint8), cv2.IMREAD_COLOR)
    img_object = base64.b64decode(item.base64_img)
    image = cv2.imdecode(np.fromstring(img_object, np.uint8), cv2.IMREAD_COLOR)
    is_card, is_drivingLicense, warped = cropper.process(image=image)

    if is_card is False and is_drivingLicense is None:
        return {'message': 'approved', 'description': 'please upload your DrivingLicense'}

    if is_drivingLicense is not None and warped is None:
        return {'message': 'approved', 'description': 'please upload DrivingLicense again'}
    info_images = detector.process(warped)
    # for i in range(len(info_images)):
    #     cv2.imshow("img",info_images[i])
    #     cv2.waitKey()
    info = dict()

    for id in list(info_images.keys()):
        label = detector.i2label_cc[id]
        if isinstance(info_images[id], np.ndarray):
            # pred = ocr.ocr(info_images[id])
            pred = ocr.recognize_text(images=[info_images[id]])
            try:
                text = pred[0]['data'][0]['text']
                info[label] = text
            except IndexError:
                print("IndexError")
                info[label] = ''


    return {'message': 'approved', 'description': 'image is drivingLicense', 'info': info}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")
