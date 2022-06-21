import cv2
import numpy as np
from typing import List

from .process_image import align_image

import argparse
import os
import sys
from pathlib import Path

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
weights = root + '/cropper/best.pt'
device = select_device(device)
data = root + '/cropper/drivingtLicenseCropCoco.yaml'
imgsz = (640, 640)
half = False
model = DetectMultiBackend(weights, device=device, dnn=False, data=data)
stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
imgsz = check_img_size(imgsz, s=stride)  # check image size

# Half
half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
if pt or jit:
    model.model.half() if half else model.model.float()

class Cropper:
    def __init__(self,score_threshold=0.5):
        self.drivingLicense_threshold = score_threshold

        self.best_bboxes = None
        # coordinate of 4 corners
        self.points = None
        self.drivingLicense_score = None
        self.is_drivingLicense = None

    @staticmethod
    def preprocess_img(img):
        img = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        return img

    @torch.no_grad()
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

    def _is_chosen(self, best_b_boxes):

        best_b_boxes: np.ndarray = np.array(best_b_boxes)
        num_objs = [0 for _ in range(4)]
        for i in range(len(best_b_boxes)):
            class_idx = int(best_b_boxes[i, 5])
            num_objs[class_idx] += 1

        # check image whether contains 5 classes
        if 0 in num_objs:
            return False
        else:
            # select best box of each class
            final_best_bboxes = np.zeros((4, best_b_boxes.shape[1]))
            classes = best_b_boxes[:, 5].astype(int)
            scores = best_b_boxes[:, 4]

            for i in range(4):
                mask = classes == i
                idx = np.argmax(scores * mask)
                final_best_bboxes[i] = best_b_boxes[idx]

            setattr(self, "best_bboxes", final_best_bboxes)
            points = self._convert_bbox_to_points()

            # check the position of corners whether is correct
            if self._check_points(points):
                setattr(self, "points", points)
            else:
                setattr(self, "points", None)
                return False

            return True

    def _convert_bbox_to_points(self) -> List[List[int]]:
        """
        :return: Coordinate of 4 corners
        """
        classes = self.best_bboxes[:, 5]
        idx = np.argsort(classes)
        left_top_box, left_bottom_box, right_bottom_box, drivingLicense = self.best_bboxes[idx]
        x_min = np.amin([left_top_box[0],left_bottom_box[0], right_bottom_box[0], drivingLicense[0]],axis=0)
        y_min = np.amin([left_top_box[1],left_bottom_box[1], right_bottom_box[1], drivingLicense[1]],axis=0)
        x_max = np.amax([left_top_box[2],left_bottom_box[2], right_bottom_box[2], drivingLicense[2]],axis=0)
        y_max = np.amax([left_top_box[3],left_bottom_box[3], right_bottom_box[3], drivingLicense[3]],axis=0)
        left_top = [x_min,y_min]
        right_top = [x_max,y_min]
        left_bottom = [x_min,y_max]
        right_bottom = [x_max,y_max]

        points = list([left_top, right_top, left_bottom, right_bottom])

        return points

    def _is_drivingLicense(self):

        idx = list(np.where((self.best_bboxes[:, 5]).astype(int) == 3))

        if not idx:
            return False
        else:
            drivingLicense_score = self.best_bboxes[idx[0], 4]
            # id_card_score = id_card_box[0, 4]
            setattr(self, 'id_score', drivingLicense_score)
            if drivingLicense_score < self.drivingLicense_threshold:
                return False

        return True

    def _check_points(self, points):
        """
        Check points whether are correctly position
        """
        top_left, top_right, bottom_left, bottom_right = points

        # top_left
        if not (top_left[0] < top_right[0] and top_left[1] < bottom_left[1]):
            return False

        # top_right
        if not (top_right[0] > top_left[0] and top_right[1] < bottom_right[1]):
            return False
        # bottom_left
        if not (bottom_left[0] < bottom_right[0] and bottom_left[1] > top_left[1]):
            return False

        # bottom_right
        if not (bottom_right[0] > bottom_left[0] and bottom_right[1] > top_right[1]):
            return False

        return True

    def process(self, image) -> object:
        """
        Process image. Return True if image is id card. Otherwise return False
        """
        # Raw Image
        image_0 = image

        # Rotate 90
        image_90 = cv2.rotate(image_0, cv2.ROTATE_90_CLOCKWISE)

        # Rotate 180
        image_180 = cv2.rotate(image_0, cv2.ROTATE_180)

        # Rotate 270
        image_270 = cv2.rotate(image_0, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # inference via yolo model
        best_b_boxes_0 = self.infer(image_0)

        best_b_boxes_90 = self.infer(image_90)

        best_b_boxes_180 = self.infer(image_180)

        best_b_boxes_270 = self.infer(image_270)
        # TODO draw box for each tangle
        # if(len(best_b_boxes_0)>0):
        #     for box in best_b_boxes_0:
        #         x_min, y_min, x_max, y_max, _, __ = box
        #         x_min = int(x_min)
        #         y_min = int(y_min)
        #         x_max = int(x_max)
        #         y_max = int(y_max)
        #         cv2.rectangle(image_0, (x_min,y_min),(x_max,y_max),(0,255,0),2)
        # if (len(best_b_boxes_90) > 0):
        #     for box in best_b_boxes_0:
        #         x_min, y_min, x_max, y_max, _, __ = box
        #         x_min = int(x_min)
        #         y_min = int(y_min)
        #         x_max = int(x_max)
        #         y_max = int(y_max)
        #         cv2.rectangle(image_90, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        # if (len(best_b_boxes_180) > 0):
        #     for box in best_b_boxes_0:
        #         x_min, y_min, x_max, y_max, _, __ = box
        #         x_min = int(x_min)
        #         y_min = int(y_min)
        #         x_max = int(x_max)
        #         y_max = int(y_max)
        #         cv2.rectangle(image_180, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        # if (len(best_b_boxes_270) > 0):
        #     for box in best_b_boxes_0:
        #         x_min, y_min, x_max, y_max, _, __ = box
        #         x_min = int(x_min)
        #         y_min = int(y_min)
        #         x_max = int(x_max)
        #         y_max = int(y_max)
        #         cv2.rectangle(image_270, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        # cv2.namedWindow("0",0)
        # cv2.imshow("0", image_0)
        #
        # cv2.namedWindow("90", 0)
        # cv2.imshow("90",image_90)
        #
        # cv2.namedWindow("180", 0)
        # cv2.imshow("180",image_180)
        #
        # cv2.namedWindow("270", 0)
        # cv2.imshow("270",image_270)
        # cv2.waitKey()

        img_0_is_chosen: bool = self._is_chosen(best_b_boxes_0)
        img_90_is_chosen: bool = self._is_chosen(best_b_boxes_90)
        img_180_is_chosen: bool = self._is_chosen(best_b_boxes_180)
        img_270_is_chosen: bool = self._is_chosen(best_b_boxes_270)

        warped = None
        if img_0_is_chosen:
            warped = align_image(image_0, self.points)
        elif img_90_is_chosen:
            warped = align_image(image_90, self.points)
        elif img_180_is_chosen:
            warped = align_image(image_180, self.points)
        elif img_270_is_chosen:
            warped = align_image(image_270, self.points)

        if warped is not None:
            is_drivingLicense = self._is_drivingLicense()
            if is_drivingLicense:
                setattr(self, 'is_drivingLicense', True)
            else:
                setattr(self, 'is_drivingLicense', False)
                return False, getattr(self, "is_drivingLicense"), None

            return True, getattr(self, "is_drivingLicense"), warped

        return False, None, None
