# --------------------------------------THIẾT-BỊ-NHẬN-DIỆN-BIỂN-BÁO-TỐC--------------------------------------------------
# ----------------------------------------------------YOLOV5M------------------------------------------------------------
# ------------------------------------------------IMPORT-LIBRARY---------------------------------------------------------
import argparse
import os
import sys
import time
from tokenize import Imagnumber
import cv2  # Opencv4.1.0
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import datetime

from pathlib import Path
from playsound import playsound
from time import sleep
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


# ----------------------------------------------------DISTANCE-----------------------------------------------------------
def distance_to_camera(knownWidth, focalLength, perWidth):
    return (knownWidth * focalLength) / perWidth


KNOWN_DISTANCE = 35.0
KNOWN_WIDTH = 16
focalLength = 27

# ______________________________________________________________________________________________________________________
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# -----------------------------------------------------FOLDER------------------------------------------------------------
path_main = "C:/Users/HuuTrien/yolov5"
def create_folder(my_local, date):
    # address folder
    path_main = my_local
    # check all folder in folder
    file_list = os.listdir(path_main)
    # checking folder not in folder
    if date not in file_list:
        # named of folder = day
        dir = date
        # create folder
        path = os.path.join(path_main, dir)
        os.mkdir(path)
    # if already had folder using this folder
    clone = "C:/Users/HuuTrien/yolov5/{}".format(date)
    return clone


# FULL_SCREEN


# ------------------------------------------------HIỂN-THỊ-BIỂN-BÁO------------------------------------------------------
FolderPath = "car"
lst = os.listdir(FolderPath)
lst_2 = []
for i in lst:
    mau = cv2.imread(f"{FolderPath}/{i}")
    lst_2.append(mau)
# --------------------------------------------------------SETUP----------------------------------------------------------
def run(weights=ROOT / 'yolov5m.pt',  # model.pt path(s)
        source=ROOT / '0',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/KLTN.yaml',  # dataset.yaml path
        imgsz=(128, 128),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1,  # Tối đa ảnh nhạn dạng được trong 1 khung hình
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'SaveVideo',  # path save project
        name='Car',  # Name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    pTime = 0

#---------------------------------------------------VIDEO-CAP-READ-----------------------------------------------------

    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, CAM = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, img0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, img0s)

# -----------------------------------------------------IMG0-LABEl--------------------------------------------------------
# Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, img0, frame = path[i], img0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, img0, frame = path, img0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = img0.copy() if save_crop else img0  # for save_crop
            annotator = Annotator(img0, line_width=line_thickness, example=str(names))
# ------------------------------------------------------DATE-TIME--------------------------------------------------------
            date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            date = date_time[:10]  # get day
            year = date_time[:4]
            month = date_time[5:7]
            day = date_time[8:10]
            hour = date_time[11:13]
            min = date_time[14:16]
            sec = date_time[17:19]
            clone = create_folder(path_main, date)
# -----------------------------------------------------------------------------------------------------------------------
            if len(det):
                # Rescale boxes from img_size to img0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    cv2.rectangle(img0, (-30, 480), (650, 435), (137, 137, 137), -1)  # (x_trái, y_dưới ; x_phải,y_trên)
                    cv2.rectangle(img0, (-30, 478), (650, 435), (0, 0, 0), 2)

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]}')

                        local_time_filename = str(datetime.datetime.now().strftime("%H-%M-%S"))
# ----------------------------------------------------IF-ELSE-PUTEXT-----------------------------------------------------
                        date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        date = date_time[:10]
                        # Day-Month-Year
                        year = date_time[:4]
                        month = date_time[5:7]
                        day = date_time[8:10]
                        # Hour-min-secf
                        hour = date_time[11:13]
                        min = date_time[14:16]
                        sec = date_time[17:19]

                # ---------------------------------------------20---------------------------------------------#
                        if label == '20':  # bien 20
                            #label_frame = '20'
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            h, w, c = lst_2[0].shape
                            img0[0:h, 0:w] = lst_2[2]
                            #cv2.rectangle(img0, (700, 30), (480, 80), (137, 137, 137), -1)
                            #cv2.rectangle(img0, (700, 30), (480, 80), (0, 0, 0), 2)
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            distance = distance_to_camera(KNOWN_WIDTH, focalLength, xywh[2] * 96)
                            #cv2.putText(img0, "%.2fcm" % (distance),
                                        #(img0.shape[1] - 148, img0.shape[0] - 406), cv2.FONT_HERSHEY_SIMPLEX,
                                        #1, (0, 0, 255), 2)

                            cv2.rectangle(img0, (100, 0), (650, 40), (137, 137, 137),-1)  # (x_trái, y_trên ; x_phải,y_dưới)
                            cv2.rectangle(img0, (100, 1), (650, 40), (0, 0, 0), 2)  # (x_trái, y_trên ; x_phải,y_dưới)
                            cv2.putText(img0,' ' + day + ':' + month + ':' + year + '      ' + hour + ':' + min + ':' + sec,
                                        (130, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)

                            cv2.putText(img0, "     Speed Limited - 20" + "km/h",
                                        (30, 468), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

                            if distance > 30 and distance < 32:
                                filename = 'BIEN20 ' + local_time_filename + '.jpg'
                                cv2.imwrite("{}/{}".format(clone, filename), img0)
                               # playsound('C:/Users/HuuTrien/yolov5/sound/sound20.mp3', False)

            # ---------------------------------------------30---------------------------------------------#
                        if label == '30':
                            #label_frame = '30'
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            h, w, c = lst_2[0].shape
                            img0[0:h, 0:w] = lst_2[3]
                            #cv2.rectangle(img0, (700, 30), (480, 80), (137, 137, 137), -1)
                            #cv2.rectangle(img0, (700, 30), (480, 80), (0, 0, 0), 2)
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            distance = distance_to_camera(KNOWN_WIDTH, focalLength, xywh[2] * 96)
                            #cv2.putText(img0, "%.2fcm" % (distance),
                                        #(img0.shape[1] - 148, img0.shape[0] - 406), cv2.FONT_HERSHEY_SIMPLEX,
                                        #1, (0, 0, 255), 2)

                            cv2.rectangle(img0, (100, 0), (650, 40), (137, 137, 137),-1)  # (x_trái, y_trên ; x_phải,y_dưới)
                            cv2.rectangle(img0, (100, 1), (650, 40), (0, 0, 0), 2)  # (x_trái, y_trên ; x_phải,y_dưới)
                            cv2.putText(img0,' ' + day + ':' + month + ':' + year + '      ' + hour + ':' + min + ':' + sec,
                                        (130, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)

                            cv2.putText(img0, "     Speed Limited - 30" + "km/h",
                                        (30, 468), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

                            if distance > 30 and distance < 32:
                                filename = 'BIEN30 ' + local_time_filename + '.jpg'
                                cv2.imwrite("{}/{}".format(clone, filename), img0)
                              #  playsound('C:/Users/HuuTrien/yolov5/sound/sound30.mp3', False)
            # ---------------------------------------------40---------------------------------------------#
                        if label == '40':
                            #label_frame = '40'
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            h, w, c = lst_2[0].shape
                            img0[0:h, 0:w] = lst_2[4]
                            #cv2.rectangle(img0, (700, 30), (480, 80), (137, 137, 137), -1)
                            #cv2.rectangle(img0, (700, 30), (480, 80), (0, 0, 0), 2)
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            distance = distance_to_camera(KNOWN_WIDTH, focalLength, xywh[2] * 96)
                            #cv2.putText(img0, "%.2fcm" % (distance),
                                       # (img0.shape[1] - 148, img0.shape[0] - 406), cv2.FONT_HERSHEY_SIMPLEX,
                                        #1, (0, 0, 255), 2)

                            cv2.rectangle(img0, (100, 0), (650, 40), (137, 137, 137),-1)  # (x_trái, y_trên ; x_phải,y_dưới)
                            cv2.rectangle(img0, (100, 1), (650, 40), (0, 0, 0), 2)  # (x_trái, y_trên ; x_phải,y_dưới)
                            cv2.putText(img0,' ' + day + ':' + month + ':' + year + '      ' + hour + ':' + min + ':' + sec,
                                        (130, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)

                            cv2.putText(img0, "     Speed Limited - 40" + "km/h",
                                        (30, 468), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

                            if distance > 30 and distance < 32:
                                filename = 'BIEN40 ' + local_time_filename + '.jpg'
                                cv2.imwrite("{}/{}".format(clone, filename), img0)
                              #  playsound('C:/Users/HuuTrien/yolov5/sound/sound40.mp3', False)
            # ---------------------------------------------50---------------------------------------------#
                        if label == '50':
                            #label_frame = '50'
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            h, w, c = lst_2[5].shape
                            img0[0:h, 0:w] = lst_2[5]
                            #cv2.rectangle(img0, (700, 30), (480, 80), (137, 137, 137), -1)
                            #cv2.rectangle(img0, (700, 30), (480, 80), (0, 0, 0), 2)
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            distance = distance_to_camera(KNOWN_WIDTH, focalLength, xywh[2] * 96)
                           # cv2.putText(img0, "%.2fcm" % (distance),
                                       # (img0.shape[1] - 148, img0.shape[0] - 406), cv2.FONT_HERSHEY_SIMPLEX,
                                        #1, (0, 0, 255), 2)

                            cv2.rectangle(img0, (100, 0), (650, 40), (137, 137, 137),-1)  # (x_trái, y_trên ; x_phải,y_dưới)
                            cv2.rectangle(img0, (100, 1), (650, 40), (0, 0, 0), 2)  # (x_trái, y_trên ; x_phải,y_dưới)
                            cv2.putText(img0,' ' + day + ':' + month + ':' + year + '      ' + hour + ':' + min + ':' + sec,
                                        (130, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)

                            cv2.putText(img0, "     Speed Limited - 50" + "km/h",
                                        (30, 468), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

                            if distance > 30 and distance < 32:
                                filename = 'BIEN50 ' + local_time_filename + '.jpg'
                                cv2.imwrite("{}/{}".format(clone, filename), img0)
                               # playsound('C:/Users/HuuTrien/yolov5/sound/sound50.mp3', False)
            # ---------------------------------------------60---------------------------------------------#
                        if label == '60':
                            #label_frame = '60'
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            h, w, c = lst_2[0].shape
                            img0[0:h, 0:w] = lst_2[6]
                            #cv2.rectangle(img0, (700, 30), (480, 80), (137, 137, 137), -1)
                            #cv2.rectangle(img0, (700, 30), (480, 80), (0, 0, 0), 2)
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            distance = distance_to_camera(KNOWN_WIDTH, focalLength, xywh[2] * 96)
                            #cv2.putText(img0, "%.2fcm" % (distance),
                                        #(img0.shape[1] - 148, img0.shape[0] - 406), cv2.FONT_HERSHEY_SIMPLEX,
                                       # 1, (0, 0, 255), 2)

                            cv2.rectangle(img0, (100, 0), (650, 40), (137, 137, 137),-1)  # (x_trái, y_trên ; x_phải,y_dưới)
                            cv2.rectangle(img0, (100, 1), (650, 40), (0, 0, 0), 2)  # (x_trái, y_trên ; x_phải,y_dưới)
                            cv2.putText(img0,' ' + day + ':' + month + ':' + year + '      ' + hour + ':' + min + ':' + sec,
                                        (130, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)

                            cv2.putText(img0, "     Speed Limited - 60" + "km/h",
                                        (30, 468), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

                            if distance > 30 and distance < 32:
                                filename = 'BIEN60 ' + local_time_filename + '.jpg'
                                cv2.imwrite("{}/{}".format(clone, filename), img0)
                              #  playsound('C:/Users/HuuTrien/yolov5/sound/sound60.mp3', False)
            # ---------------------------------------------70---------------------------------------------#
                        if label == '70':
                            #label_frame = '70'
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            h, w, c = lst_2[0].shape
                            img0[0:h, 0:w] = lst_2[7]
                            #cv2.rectangle(img0, (700, 30), (480, 80), (137, 137, 137), -1)
                            #cv2.rectangle(img0, (700, 30), (480, 80), (0, 0, 0), 2)
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            distance = distance_to_camera(KNOWN_WIDTH, focalLength, xywh[2] * 96)
                           # cv2.putText(img0, "%.2fcm" % (distance),
                                       # (img0.shape[1] - 148, img0.shape[0] - 406), cv2.FONT_HERSHEY_SIMPLEX,
                                       # 1, (0, 0, 255), 2)

                            cv2.rectangle(img0, (100, 0), (650, 40), (137, 137, 137),-1)  # (x_trái, y_trên ; x_phải,y_dưới)
                            cv2.rectangle(img0, (100, 1), (650, 40), (0, 0, 0), 2)  # (x_trái, y_trên ; x_phải,y_dưới)
                            cv2.putText(img0,' ' + day + ':' + month + ':' + year + '      ' + hour + ':' + min + ':' + sec,
                                        (130, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)

                            cv2.putText(img0, "     Speed Limited - 70" + "km/h",
                                        (30, 468), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

                            if distance > 30 and distance < 32:
                                filename = 'BIEN70 ' + local_time_filename + '.jpg'
                                cv2.imwrite("{}/{}".format(clone, filename), img0)
                               # playsound('C:/Users/HuuTrien/yolov5/sound/sound70.mp3', False)
            # ---------------------------------------------80---------------------------------------------#
                        if label == '80':
                            #label_frame = '80'
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            h, w, c = lst_2[0].shape
                            img0[0:h, 0:w] = lst_2[8]
                            #cv2.rectangle(img0, (700, 30), (480, 80), (137, 137, 137), -1)
                            #cv2.rectangle(img0, (700, 30), (480, 80), (0, 0, 0), 2)
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            distance = distance_to_camera(KNOWN_WIDTH, focalLength, xywh[2] * 96)
                            #cv2.putText(img0, "%.2fcm" % (distance),
                                       # (img0.shape[1] - 148, img0.shape[0] - 406), cv2.FONT_HERSHEY_SIMPLEX,
                                     #   1, (0, 0, 255), 2)
                            cv2.putText(img0, "     Speed Limited - 80" + "km/h",
                                        (30, 468), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

                            cv2.rectangle(img0, (100, 0), (650, 40), (137, 137, 137),-1)  # (x_trái, y_trên ; x_phải,y_dưới)
                            cv2.rectangle(img0, (100, 1), (650, 40), (0, 0, 0), 2)  # (x_trái, y_trên ; x_phải,y_dưới)
                            cv2.putText(img0,' ' + day + ':' + month + ':' + year + '      ' + hour + ':' + min + ':' + sec,
                                        (130, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)

                            if distance > 30 and distance < 32:
                                filename = 'BIEN80 ' + local_time_filename + '.jpg'
                                cv2.imwrite("{}/{}".format(clone, filename), img0)
                                #playsound('C:/Users/HuuTrien/yolov5/sound/sound80.mp3', False)
            # ---------------------------------------------90---------------------------------------------#
                        if label == '90':
                            #label_frame = '90'
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            h, w, c = lst_2[0].shape
                            img0[0:h, 0:w] = lst_2[9]
                            #cv2.rectangle(img0, (700, 30), (480, 80), (137, 137, 137), -1)
                            #cv2.rectangle(img0, (700, 30), (480, 80), (0, 0, 0), 2)
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            distance = distance_to_camera(KNOWN_WIDTH, focalLength, xywh[2] * 96)
                           # cv2.putText(img0, "%.2fcm" % (distance),
                                    #    (img0.shape[1] - 148, img0.shape[0] - 406), cv2.FONT_HERSHEY_SIMPLEX,
                                    #    1, (0, 0, 255), 2)

                            cv2.rectangle(img0, (100, 0), (650, 40), (137, 137, 137),-1)  # (x_trái, y_trên ; x_phải,y_dưới)
                            cv2.rectangle(img0, (100, 1), (650, 40), (0, 0, 0), 2)  # (x_trái, y_trên ; x_phải,y_dưới)
                            cv2.putText(img0,' ' + day + ':' + month + ':' + year + '      ' + hour + ':' + min + ':' + sec,
                                        (130, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)

                            cv2.putText(img0, "     Speed Limited - 90" + "km/h",
                                        (30, 468), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

                            if distance > 30 and distance < 32:
                                filename = 'BIEN90 ' + local_time_filename + '.jpg'
                                cv2.imwrite("{}/{}".format(clone, filename), img0)
                              #  playsound('C:/Users/HuuTrien/yolov5/sound/sound90.mp3', False)
            # ---------------------------------------------100---------------------------------------------#
                        if label == '100':
                            #label_frame = '100'
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            h, w, c = lst_2[0].shape
                            img0[0:h, 0:w] = lst_2[0]
                            #cv2.rectangle(img0, (700, 30), (480, 80), (137, 137, 137), -1)
                            #cv2.rectangle(img0, (700, 30), (480, 80), (0, 0, 0), 2)
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            distance = distance_to_camera(KNOWN_WIDTH, focalLength, xywh[2] * 96)
                           # cv2.putText(img0, "%.2fcm" % (distance),
                                      #  (img0.shape[1] - 148, img0.shape[0] - 406), cv2.FONT_HERSHEY_SIMPLEX,
                                      #  1, (0, 0, 255), 2)

                            cv2.rectangle(img0, (100, 0), (650, 40), (137, 137, 137),-1)  # (x_trái, y_trên ; x_phải,y_dưới)
                            cv2.rectangle(img0, (100, 1), (650, 40), (0, 0, 0), 2)  # (x_trái, y_trên ; x_phải,y_dưới)
                            cv2.putText(img0,' ' + day + ':' + month + ':' + year + '      ' + hour + ':' + min + ':' + sec,
                                        (130, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)

                            cv2.putText(img0, "     Speed Limited - 100" + "km/h",
                                        (30, 468), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

                            if distance > 30 and distance < 32:
                                filename = 'BIEN100 ' + local_time_filename + '.jpg'
                                cv2.imwrite("{}/{}".format(clone, filename), img0)
                              #  playsound('C:/Users/HuuTrien/yolov5/sound/sound100.mp3', False)
            # ---------------------------------------------120---------------------------------------------#
                        if label == '120':
                            #label_frame = '120'
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            h, w, c = lst_2[0].shape
                            img0[0:h, 0:w] = lst_2[1]
                            #cv2.rectangle(img0, (700, 30), (480, 80), (137, 137, 137), -1)
                            #cv2.rectangle(img0, (700, 30), (480, 80), (0, 0, 0), 2)
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            distance = distance_to_camera(KNOWN_WIDTH, focalLength, xywh[2] * 96)
                         #   cv2.putText(img0, "%.2fcm" % (distance),
                                    #    (img0.shape[1] - 148, img0.shape[0] - 406), cv2.FONT_HERSHEY_SIMPLEX,
                                    #    1, (0, 0, 255), 2)

                            cv2.rectangle(img0, (100, 0), (650, 40), (137, 137, 137),-1)  # (x_trái, y_trên ; x_phải,y_dưới)
                            cv2.rectangle(img0, (100, 1), (650, 40), (0, 0, 0), 2)  # (x_trái, y_trên ; x_phải,y_dưới)
                            cv2.putText(img0,' ' + day + ':' + month + ':' + year + '      ' + hour + ':' + min + ':' + sec,
                                        (130, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)

                            cv2.putText(img0, "    Speed Limited - 120" + "km/h",
                                        (30, 468), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

                            if distance > 30 and distance < 32:
                                filename = 'BIEN120 ' + local_time_filename + '.jpg'
                                cv2.imwrite("{}/{}".format(clone, filename), img0)
                              #  playsound('C:/Users/HuuTrien/yolov5/sound/sound120.mp3', False)

# -----------------------------------------------------------------------------------------------------------------------
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

# ------------------------------------------------DAY-TODAY-TIME-FPS-PUTEXT----------------------------------------------
            # Stream results
            img0 = annotator.result()
            if view_img:
                # Date_time
                date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                date = date_time[:10]
                # Day-Month-Year
                year = date_time[:4]
                month = date_time[5:7]
                day = date_time[8:10]
                # Hour-min-secf
                hour = date_time[11:13]
                min = date_time[14:16]
                sec = date_time[17:19]

                # Vẽ nền và khung cho Date_Time_FPS
                cv2.rectangle(img0, (100, 0), (650, 40), (137, 137, 137), -1)  # (x_trái, y_trên ; x_phải,y_dưới)
                cv2.rectangle(img0, (100, 1), (650, 40), (0, 0, 0), 2)  # (x_trái, y_trên ; x_phải,y_dưới)

                # Puttext
                cv2.putText(img0, ' '+day + ':' + month + ':' + year + '      ' + hour + ':' + min + ':' + sec,
                            (130, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)
                # FPS
                cTime = time.time()
                fps = (1 / (cTime - pTime))
                pTime = cTime
                cv2.putText(img0, f"FPS:{int(fps)}", (540, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)

                cv2.namedWindow("KLTN", cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty("KLTN", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

                cv2.imshow("KLTN", img0)
                cv2.waitKey(1)
#--------------------------------------------------SAVE-FILE-MP4--------------------------------------------------------
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, img0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(CAM[i], cv2.VideoWriter):
                            CAM[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 6, img0.shape[1], img0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        CAM[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    CAM[i].write(img0)


# -----------------------------------------------------SETUP-------------------------------------------------------------
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'exp2.pt')
    parser.add_argument('--source', type=str, default=ROOT / '0')
    parser.add_argument('--data', type=str, default=ROOT / 'data/KLTN.yaml')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[128], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1)  # Tối đa ảnh nhạn dạng được trong 1 khung hình
    parser.add_argument('--device', default='')  # CPU
    parser.add_argument('--view-img', action='store_true')  # Hiển thị kết quả
    parser.add_argument('--save-txt', action='store_true')  # Save kết quả
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


# -----------------------------------------------------------------------------------------------------------------------

def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

# -----------------------------------------------------------------------------------------------------------------------
