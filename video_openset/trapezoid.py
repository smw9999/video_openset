import cv2
import numpy as np

# 동영상 불러오기
capture = cv2.VideoCapture('videos/challenge.mp4')

if capture.isOpened() == False:
   print("카메라를 열 수 없습니다.")
   exit(1)

while True:
   ret, img_src = capture.read()

   trap_bottom_width = 0.8
   trap_top_width = 0.1
   trap_height = 0.4

   img_mask = np.zeros_like(img_src)
   height, width = img_mask.shape[:2]
   mask_color = (255, 255, 255)

   pts = np.array([[
      ((width * (1 - trap_bottom_width)) // 2, height),
      ((width * (1 - trap_top_width)) // 2, (1 - trap_height) * height),
      (width - (width * (1 - trap_top_width)) // 2, (1 - trap_height) * height),
      (width - (width * (1 - trap_bottom_width)) // 2, height)]],
      dtype=np.int32)

   cv2.fillPoly(img_mask, pts, mask_color)
   img_src = cv2.bitwise_and(img_src, img_mask)

   cv2.imshow('Video-dst', img_src)
   key = cv2.waitKey(25)  # 33ms마다
   if key == 27:  # Esc 키
       break

capture.release()
cv2.destroyAllWindows()