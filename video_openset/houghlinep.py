import cv2
import numpy as np

# 동영상 불러오기
capture = cv2.VideoCapture('videos/challenge.mp4')

if capture.isOpened() == False:
   print("카메라를 열 수 없습니다.")
   exit(1)

while True:
   ret, img_src = capture.read()

   img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
   # 이미지 블러
   img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
   # 이진화 수행
   _, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
   img_canny = cv2.Canny(img_binary, 50, 150)


   img_roi = img_canny
   rho = 2
   theta = 1 * np.pi / 180
   threshold = 15
   min_line_length = 10
   max_line_gap = 20
   lines = cv2.HoughLinesP(img_roi, rho, theta, threshold,
                           minLineLength=min_line_length,
                           maxLineGap=max_line_gap)
   for i, line in enumerate(lines):
       cv2.line(img_src, (line[0][0], line[0][1]),
                (line[0][2], line[0][3]), (0, 255, 0), 2)

   cv2.imshow('Video-dst', img_src)
   key = cv2.waitKey(25)  # 33ms마다
   if key == 27:  # Esc 키
       break

capture.release()
cv2.destroyAllWindows()
