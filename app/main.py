# -*- coding: utf-8 -*- 
import os

import time
import base64
import requests

import cv2
import numpy as np
import uvicorn

from fastapi import FastAPI, UploadFile, File
from fastapi import Request
from fastapi.responses import HTMLResponse
from starlette.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/'+'..'))
from paddleocr import PaddleOCR
from cropper import Cropper
from detector import Detector
import paddlehub as hub
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

    img_object = base64.b64decode(item.base64_img)
    image = cv2.imdecode(np.fromstring(img_object, np.uint8), cv2.IMREAD_COLOR)

    is_card, is_drivingLicense, warped = cropper.process(image=image)

    if is_card is False and is_drivingLicense is None:
        return {'message': 'approved', 'description': 'please upload your DrivingLicense'}

    if is_drivingLicense is not None and warped is None:
        return {'message': 'approved', 'description': 'please upload DrivingLicense again'}
    info_images = detector.process(warped)

    info = dict()

    for id in list(info_images.keys()):
        label = detector.i2label_cc[id]
        if isinstance(info_images[id], np.ndarray):
            # pred = ocr.ocr(info_images[id])
            pred = ocr.recognize_text(images=[info_images[id]])
            try:
                text = pred[0]['data'][0]['text']
                # text = pred[0][1][0]
                info[label] = text
            except IndexError:
                print("IndexError")
                info[label] = ''


    return {'message': 'approved', 'description': 'image is drivingLicense', 'info': info}
@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(os.path.join('upload_image.html'), {'request': request})


@app.post('/upload_image/')
def upload_image(request: Request, file: UploadFile = File(...)):
    if file.content_type in ['image/jpeg', 'image/png']:

        img_object = file.file.read()


        my_string = base64.b64encode(img_object)
        my_string = my_string.decode('utf-8')
        url = 'http://127.0.0.1:8000' + '/extract'
        request_body = {'base64_img': my_string}

        response = requests.post(url=url, json=request_body)
        response = response.json()
        if response['description'] != 'image is drivingLicense':
            return templates.TemplateResponse(os.path.join('upload_image_again.html'), {'request': request})

        base64_img = "data:image/png;base64," + my_string
        return templates.TemplateResponse(os.path.join('predict.html'),
                                          {'request': request, 'base64_img': base64_img, 'info': response['info']})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")
