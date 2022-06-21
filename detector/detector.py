import cv2
import numpy as np
from typing import List, Dict, Any, Union

import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
import os
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.augmentations import letterbox
root = str(Path(os.getcwd()))
# root = str(root.parent)
device = ''
device = select_device(device)
weights = root + '/detector/best.pt'
data = root + '/detector/drivingtLicenseDetecCoco.yaml'
imgsz = (640, 640)
model = DetectMultiBackend(weights, device=device, dnn=False, data=data)
stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine

# Half
half = False
half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
if pt or jit:
    model.model.half() if half else model.model.float()

class Detector:

    def __init__(self, score_threshold=0.5):

        self.score_threshold = score_threshold
        self.i2label_cc = {0: 'plateNo', 1: 'vehicleType', 2: 'VIN', 3: 'EngineNo', 4: 'RegisterDate'}


    @staticmethod
    def preprocess_img(img):
        img = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        return img

    @staticmethod
    def crop(original_image, b_box):
        x_min, y_min, x_max, y_max, _, _ = list(map(int, list(b_box)))
        cropped_image = original_image[y_min:y_max, x_min: x_max]
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

        return cropped_image

    @staticmethod
    def decode_address(address_bbox: np.ndarray):
        """
        address_info[i] = nd.array([x_min, y_min, x_max, y_max, score, class_id])
        :Return
        """

        y_mins = address_bbox[:, 1]
        args = np.argsort(y_mins)
        address_bbox = address_bbox[args]

        num_address = address_bbox.shape[0]

        address = {}

        if num_address == 4:
            address['place_of_birth'] = address_bbox[:2]
            address['place_of_residence'] = address_bbox[2:4]

            return address
        elif num_address == 2:
            address['place_of_birth'] = address_bbox[0].reshape(1, -1)
            address['place_of_residence'] = address_bbox[1].reshape(1, -1)

            return address

        bbox_1 = list(address_bbox[0])
        bbox_2 = list(address_bbox[1])
        bbox_3 = list(address_bbox[2])

        distance_12 = bbox_2[1] - bbox_1[3]
        distance_23 = bbox_3[1] - bbox_2[3]

        address['place_of_birth'] = []
        address['place_of_residence'] = []
        address['place_of_birth'].append(bbox_1)
        if distance_12 < distance_23:
            address['place_of_birth'].append(bbox_2)
        else:
            address['place_of_residence'].append(bbox_2)

        address['place_of_residence'].append(bbox_3)

        address['place_of_birth'] = np.array(address['place_of_birth'])
        address['place_of_residence'] = np.array(address['place_of_residence'])

        return address

    def infer(self, image: np.ndarray):

        img = letterbox(image)[0]
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        pred = model(img)

        # NMS
        conf_thres = 0.25
        iou_thres = 0.45
        max_det = 1000
        classes = None
        agnostic_nms = False
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        pred = pred[0].cpu().numpy()

        #delete repeated box
        for i in np.unique(pred[:,5]):
            repeated_index = (np.where(pred[:,5]==i))[0]
            if(len(repeated_index) > 1):
                repeated_scores = pred[:,5][repeated_index]
                min_index = repeated_index[np.where(repeated_scores < np.max(repeated_scores))]
                pred = np.delete(pred, min_index, axis=0)
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], image.shape).round()
        return pred

    def select_best_bbox(self, b_boxes):
        classes = b_boxes[:, 5].astype(int)
        scores = b_boxes[:, 4]

        info = dict()

        for i in list(self.i2label_cc.keys()):
            mask = classes == i
            if np.sum(mask) != 0:
                idx = np.argmax(scores * mask)
                info[i] = b_boxes[idx]

        return info

    def process(self, aligned_image):

        b_boxes = self.infer(aligned_image)

        info: dict = self.select_best_bbox(b_boxes)

        info_img: Dict[Union[int, Any], Union[List[Any], Any]] = dict()

        for id in info.keys():
            info_img[id] = self.crop(aligned_image, info[id])
            cv2.imwrite(str(id)+".png", info_img[id])
        return info_img
