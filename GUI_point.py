from multiprocessing import Process, Queue
import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import True_, random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def proc(cam_num, frame, q, opt):
    print(cam_num)
    t5 = time_synchronized()

    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    if cam_num == 79:
        # rect_list = np.array([[80, 411], [283, 113], [435, 115], [620, 410]], np.int32)  # 차량 검출 범위 박스 80
        rect_list = np.array([[86, 430], [279, 119], [435, 120], [622, 417]], np.int32)  # 차량 검출 범위 박스 79
        # rect_list = np.array([[267, 132], [96, 406], [648, 416], [439, 128]], np.int32)  # 차량 검출 범위 박스 B2 A75
        # Create a black image
        img_b = np.zeros((480, 720, 3), dtype=np.uint8)
        img_b = cv2.fillPoly(img_b, [rect_list], (255, 255, 255))
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
        res, thr = cv2.threshold(img_b, 127, 255, cv2.THRESH_BINARY)
        contours_b, his = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # 차량 트래킹 검출 박스(before)
        before_track_list = np.array([[124, 312], [577, 303], [646, 410], [58, 412]])
        img_bt = np.zeros((480, 720, 3), dtype=np.uint8)
        img_bt = cv2.fillPoly(img_bt, [before_track_list], (255, 255, 255))
        img_bt = cv2.cvtColor(img_bt, cv2.COLOR_BGR2GRAY)
        res, thr = cv2.threshold(img_bt, 127, 255, cv2.THRESH_BINARY)
        contours_bt, his = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Homograpy 처리 후 매핑작업
        im_src = cv2.imread('ch_b1.png')  # 1층
        # im_src = cv2.imread('ch_b2.png')  # 2층
        # pts_src = np.array([[991, 284], [957, 283], [956, 239], [993, 239]])
        # pts_src = np.array([[6872, 1656], [7374, 1664], [7355, 1488], [6880, 1446]])  # 79
        # pts_src = np.array([[7736, 1688], [7396, 1686], [7402, 1439], [7720, 1442]])  # 80
        # pts_src = np.array([[21072, 934], [20743, 909], [20735, 1142], [21073, 1130]])  # B2 A75
        # pts_src = np.array([[7443,1699],[7163,1427],[7945,1425],[7943,1699]]) # nakkk
        pts_src = np.array([[6823, 1699],[6823, 1421],[7382, 1421],[7382, 1695]]) #hunnn 79
        cv2.polylines(im_src, [pts_src], True, (0, 0, 0), 2)
        im_dst = cv2.imread('cam80.png')
        # pts_dst = np.array([[268, 130], [84, 428], [640, 407], [433, 121]])  # 79
        # pts_dst = np.array([[202, 182], [259, 117], [448, 118], [506, 183]])  # 80
        # pts_dst = np.array([[245, 130], [96, 296], [610, 289], [461, 128]])  # B2 A75
        # pts_dst = np.array([[263,122],[424,91],[650,405],[54,406]]) # nakkk
        pts_dst = np.array([[264, 130],[441, 126], [667, 414],[63, 430]]) # hunnn 79
        cv2.polylines(im_dst, [pts_dst], True, (255, 255, 255), 2)
        h1, status = cv2.findHomography(pts_dst, pts_src)
    elif cam_num == 80:
        # rect_list = np.array([[295, 85],[52, 456],[648, 456],[418, 87]], np.int32)
        rect_list = np.array([[80, 411], [283, 113], [435, 115], [620, 410]], np.int32)  # 차량 검출 범위 박스 80
        # rect_list = np.array([[86, 430], [279, 119], [435, 120], [622, 417]], np.int32)  # 차량 검출 범위 박스 79
        # rect_list = np.array([[267, 132], [96, 406], [648, 416], [439, 128]], np.int32)  # 차량 검출 범위 박스 B2 A75
        # Create a black image
        img_b = np.zeros((480, 720, 3), dtype=np.uint8)
        img_b = cv2.fillPoly(img_b, [rect_list], (255, 255, 255))
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
        res, thr = cv2.threshold(img_b, 127, 255, cv2.THRESH_BINARY)
        contours_b, his = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # 차량 트래킹 검출 박스(after)
        track_list = np.array([[284, 100],[428, 103], [441, 118], [271, 116]])
        img_at = np.zeros((480, 720, 3), dtype=np.uint8)
        img_at = cv2.fillPoly(img_at, [rect_list], (255, 255, 255))
        img_at = cv2.cvtColor(img_at, cv2.COLOR_BGR2GRAY)
        res, thr = cv2.threshold(img_at, 127, 255, cv2.THRESH_BINARY)
        contours_at, his = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Homograpy 처리 후 매핑작업
        im_src = cv2.imread('ch_b1.png')  # 1층
        # im_src = cv2.imread('ch_b2.png')  # 2층
        # pts_src = np.array([[991, 284], [957, 283], [956, 239], [993, 239]])
        # pts_src = np.array([[6872, 1656], [7374, 1664], [7355, 1488], [6880, 1446]])  # 79
        # pts_src = np.array([[7736, 1688], [7396, 1686], [7402, 1439], [7720, 1442]])  # 80
        # pts_src = np.array([[21072, 934], [20743, 909], [20735, 1142], [21073, 1130]])  # B2 A75
        # pts_src = np.array([[7443,1699],[7163,1427],[7945,1425],[7943,1699]]) # nakkk
        pts_src = np.array([[7382, 1695], [7382, 1421], [7948, 1421], [7944, 1695]]) # hunnn 80
        cv2.polylines(im_src, [pts_src], True, (0, 0, 0), 2)
        im_dst = cv2.imread('cam80.png')
        # pts_dst = np.array([[268, 130], [84, 428], [640, 407], [433, 121]])  # 79
        # pts_dst = np.array([[202, 182], [259, 117], [448, 118], [506, 183]])  # 80
        # pts_dst = np.array([[245, 130], [96, 296], [610, 289], [461, 128]])  # B2 A75
        # pts_dst = np.array([[263,122],[424,91],[650,405],[54,406]]) # nakkk
        pts_dst = np.array([[270, 116],[441, 117], [646, 411], [57, 411]]) # hunnn 80
        cv2.polylines(im_dst, [pts_dst], True, (255, 255, 255), 2)
        h1, status = cv2.findHomography(pts_dst, pts_src)
    
    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # # Set Dataloader
    # vid_path, vid_writer = None, None
    # if webcam:
    #     view_img = check_imshow()
    #     cudnn.benchmark = True  # set True to speed up constant image size inference
    #     dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    # else:
    #     dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    
    ####################################################################
    # Padded resize
    img0 = frame.copy()
    img = letterbox(img0, imgsz, stride=32)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    t2 = time_synchronized()

    # Apply Classifier
    if classify:
        pred = apply_classifier(pred, modelc, img, frame)

    data_list = []
    person_list = []
    im0 = frame.copy()

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        #  이미지 리사이즈
        im0 = frame.copy()
        im0 = cv2.resize(im0, (720, 480))
        s = ''
        s += '%gx%g ' % img.shape[2:]  # print string(img shape)
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

        # dat = []
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            
            # Write results
            for *xyxy, conf, cls in reversed(det):
                if save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                    

                if save_img or opt.save_crop or view_img:  # Add bbox to image
                    x1, y1 = (int(xyxy[0]) + int(xyxy[2])) / 2, int(xyxy[3])  # 중심점 좌표 x, y
                                     
                    dist = cv2.pointPolygonTest(contours_b[0], (x1, y1), False)

                    # if names[int(cls)] == 'car' or names[int(cls)] == 'person':  # box 가 차량이나 사람일 경우
                    if names[int(cls)] == 'car'and dist == 1:
                        label = f'{names[int(cls)]}'
                        xyxy_ = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]) - int(xyxy[0]),
                                    int(xyxy[3]) - int(xyxy[1])]
                        data_list.append(xyxy_)
                        # x1, y1 = (int(xyxy[0]) + int(xyxy[2])) / 2, (int(xyxy[1]) + int(xyxy[3])) / 2  # 중심점 좌표 x, y
                        #  크기? 중심점 위치? 를 사용하여 디텍팅 되는 차량을 한정지음

                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                        # 차량 탐지 정확도 출력
                        cv2.putText(im0, str(conf), (xyxy_[0], xyxy_[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
        # data_list.append(dat)  # dat: 검출된 box xy 좌표 배열 목록

        # Print time (inference + NMS)
        print(f'{s}Done. ({t2 - t1:.3f}s)')

    t3 = time_synchronized()
    if len(data_list) != 0:
        dst2 = im_src.copy()

    for i in range(len(data_list)):  # boxes: yolo 검출된 box 의 x, y, w, h
        x, y, w, h = data_list[i][0], data_list[i][1], data_list[i][2], data_list[i][3]
        color = (0, 0, 255)


        fx = x + (w / 2)
        fy = y + h  # 차 밑 부분 찍는게 맞음 450
        p_point = np.array([fx, fy, 1], dtype=int)
        p_point.T
        np.transpose(p_point)
        Cal = h1 @ p_point
        realC = Cal / Cal[2]
        # print(realC)
        a = round(int(realC[0]), 0)  # 중심점 x 좌표
        b = round(int(realC[1]), 0)  # 중심점 y 좌표

        q.put([a,b])
    t6 = time_synchronized()
    print(f'{t6-t5:0.3f}s')

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/videos/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    # parser.add_argument('--totalmap-conf', default='[[7736, 1688], [7396, 1686], [7402, 1439], [7720, 1442]]',
    #                     help='totalmap')
    # parser.add_argument('--map-conf', default='[[202, 182], [259, 117], [448, 118], [506, 183]]',
    #                     help='totalmap')

    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    im_src = cv2.imread('ch_b1.png')
    for i in range(90, 201):
        dst2 = im_src.copy()
        car_width = 40
        car_length = 113
        q = Queue()
        img = cv2.imread('./data/videos/img79/%d.png' % (3*i)) # 30 FPS
        p1 = Process(target=proc, args=(79,img,q, opt))
        img = cv2.imread('./data/videos/img80/%d.png' % (2*i)) # 20 FPS
        p2 = Process(target=proc, args=(80,img,q, opt))

        p1.start()
        p2.start()
        
        p1.join()
        p2.join()

        
        for _ in range(q.qsize()):
            a, b = map(int, q.get())
            print(a, b)
            cv2.rectangle(dst2, (a - car_length, b - car_width//2), (a , b + car_width//2), (0, 0, -255), car_width)
        # t4 = time_synchronized()
        dst2 = cv2.resize(dst2, dsize=(0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
        cv2.imshow("Source Image", dst2)
        cv2.waitKey(1)
        
        
