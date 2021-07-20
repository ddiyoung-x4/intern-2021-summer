import cv2
import numpy as np
import os


url = 'rtsp://admin:admin1234@218.153.209.100:506/cam/realmonitor?channel=2&subtype=1'
cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()  # 윈도우 창 출력용
    cv2.imshow("video", frame)
    cv2.waitKey(1)

# absdiff_show = absdiff_frame
# absdiff_frame = cv2.cvtColor(absdiff_frame, cv2.COLOR_BGR2GRAY)
# # absdiff_frame[absdiff_frame < 40] = 0
#
# absdiff_frame[absdiff_frame < 50] = 0
# # absdiff_frame[absdiff_road == 0] = 0
#
# absdiff_frame = cv2.adaptiveThreshold(absdiff_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
#                                       25, 5)
# absdiff_frame = cv2.medianBlur(absdiff_frame, ksize=3)
# cnts, hier = cv2.findContours(absdiff_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#
# rects = []
#
# for cnt in cnts:
#     xx, yy, ww, hh = cv2.boundingRect(cnt)
#     zh = (yy + yy + hh) / 2
#     if zh >= 200:
#         if cv2.contourArea(cnt) < 200:
#             del cnt
#             continue
#     if zh < 200:
#         if cv2.contourArea(cnt) < 200:
#             del cnt
#             continue
#
#     empty_mask = np.zeros(frame_.shape[:2], dtype=np.uint8)
#     absdiff_mask = cv2.drawContours(empty_mask, [cnt], -1, 255, -1)
#
#     # 마스크에서 흰색 부분에 해당하는 픽셀들을 loc에 저장
#     loc = np.where(absdiff_mask == 255)
#     hsv_mask = cv2.bitwise_and(hsv_frame, hsv_frame, mask=absdiff_mask)
#     roihist_mask = cv2.calcHist([hsv_frame], [0, 1], absdiff_mask, [180, 256], [0, 180, 0, 256])
#     pixels = hsv_mask[loc]
#     pixels_180 = np.where(hsv_mask[:, 2] > 180)
#     # pixels의 2번째 channel, 즉 value값만을 추출한다..
#     pixels = pixels[:, 2]
#     mask_mean_value = np.mean(pixels)
#
#     if mask_mean_value < 170:
#         # frame = cv2.drawContours(frame, [cnt], -1, (255,255,255), -1)
#         # cv2.dilate(absdiff_frame, kernel, iterations=1)
#         # cv2.erode(absdiff_frame, kernel, iterations=1)
#         # cv2.dilate(absdifvf_frame, kernel, iterations=1)
#         # absdiff_frame = c2.drawContours(absdiff_frame, [cnt], -1, (255, 255, 255), 2)
#
#         # if abs(compare) < 55:
#         frame_ = cv2.drawContours(frame_, [cnt], -1, (255, 255, 255), 1)
#         x, y, w, h = cv2.boundingRect(cnt)
#         # if w > 2 * h:
#         #  frame = cv2.drawContours(frame, [cnt], -1, (255, 255, 255), 1)
#         startx, starty, endx, endy = x, y, x + w, y + h
#         cx, cy = int(x + w / 2), int(y + h / 2)
#         # cv2.dilate(absdiff_frame, kernel, iterations=1)
#         # cv2.erode(absdiff_frame, kernel, iterations=1)
#         # cv2.dilate(absdiff_frame, kernel, iterations=1)
#
#         absdiff_frame = cv2.drawContours(absdiff_frame, [cnt], -1, (255, 255, 255), 1)
#         compare_mask = cv2.drawContours(empty_mask, [cnt], -1, 255, -1)
#         compare_mask[pixels_180] = 0
#         cv2.circle(absdiff_frame, (cx, cy), 7, (255, 0, 255), -1)
#         box = np.array([startx, starty, endx, endy, cv2.contourArea(cnt)])
#         rects.append(box.astype("int"))
#
# # absdiff_frame = cv2.drawContours(absdiff_frame, cnts, -1, (255, 255, 255), 1)
# cv2.imshow('res', absdiff_frame)
# print('end')
# k = cv2.waitKey(1) or 0xff
# # 동영상 정지 기능 "스페이스바" 누루면 정지하고 다시 누르면 재개
# if k == 32:
#     cv2.waitKey()
# esc 누르면 영상 종료
# if k == 27:
#     break