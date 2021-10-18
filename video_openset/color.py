import cv2
import numpy as np

# 동영상 불러오기
capture = cv2.VideoCapture('videos/challenge.mp4')

if capture.isOpened() == False:
   print("카메라를 열 수 없습니다.")
   exit(1)

while True:
   ret, img_src = capture.read()
   img_dst = img_src.copy()

   # 이미지를 BGR에서 HSV로 색변환
   img_hsv = cv2.cvtColor(img_src, cv2.COLOR_BGR2HSV)  # HSV
   # HSV로 노란색 정보를 좀 더 구체적으로 표시

   lower_yellow = (15, 100, 100)  # 자료형은 튜플형태로(H,S,V)
   upper_yellow = (40, 255, 255)  # 자료형은 튜플형태로(H,S,V)
   yellow_mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)  # 초록 정보 추출
   img_yellow = cv2.bitwise_and(img_src, img_src, mask=yellow_mask)

   # BGR에서 white를 좀 더 구체적으로 표시
   lower_white = (150, 150, 150)  # 자료형은 튜플형태로(B,G,R)
   upper_white = (255, 255, 255)  # 자료형은 튜플형태로(B,G,R)
   white_mask = cv2.inRange(img_src, lower_white, upper_white)  # 초록 정보 추출

   img_white = cv2.bitwise_and(img_src, img_src, mask=white_mask)
   img_dst = cv2.addWeighted(img_white, 1., img_yellow, 1., 0)

   cv2.imshow('Video-dst', img_dst)
   key = cv2.waitKey(25)  # 33ms마다
   if key == 27:  # Esc 키
       break

capture.release()
cv2.destroyAllWindows()
