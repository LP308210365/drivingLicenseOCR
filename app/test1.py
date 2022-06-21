import os
import time
import base64

import cv2
import numpy as np

# import paddlehub as hub
from paddleocr import PaddleOCR


from cropper import Cropper
from detector import Detector



cropper = Cropper()

detector = Detector()


reader = PaddleOCR(lang="ch",det=False,version='ch_ppocr_server_v2.0_xx')


def extract(img_path):

    image = cv2.imread(img_path, cv2.IMREAD_COLOR)

    is_card, is_drivingLicense, warped = cropper.process(image=image)

    # cv2.imwrite("warpedImg.png",warped)
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
            info[label] = reader.ocr(info_images[id])
            # info[label] = reader.OCR(info_images[id])
        elif isinstance(info_images[id], list):
            info[label] = []
            for i in range(len(info_images[id])):
                info[label].append(reader.ocr(info_images[id][i]))
                # info[label].append(ocr.recognize_text(images=[info_images[id][i]]))


    return {'message': 'approved', 'description': 'image is drivingLicense', 'info': info}

if __name__=="__main__":
    img_path = "/home/DrivingLicenseOCR/app/drivinglicnese.jpg"
    message = extract(img_path)
    info = message['info']
    plateNo = ''
    vehicleType = ''
    VIN = ''
    EngineNo = ''
    RegisterDate = ''
    if 'plateNo' in info:
        # plateNo = info['plateNo'][0]['data'][0]['text']
        plateNo = info['plateNo'][0][1][0]
    if 'vehicleType' in info:
        # vehicleType = info['vehicleType'][0]['data'][0]['text']
        vehicleType = info['vehicleType'][0][1][0]
    if 'VIN' in info:
        # VIN = info['VIN'][0]['data'][0]['text']
        VIN = info['VIN'][0][1][0]
    if 'EngineNo' in info:
        # EngineNo = info['EngineNo'][0]['data'][0]['text']
        EngineNo = info['EngineNo'][0][1][0]
    if 'RegisterDate' in info:
        # RegisterDate = info['RegisterDate'][0]['data'][0]['text']
        RegisterDate = info['RegisterDate'][0][1][0]
    print(plateNo)
    print(vehicleType)
    print(VIN)
    print(EngineNo)
    print(RegisterDate)




