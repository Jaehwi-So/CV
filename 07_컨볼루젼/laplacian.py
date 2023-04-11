import cv2 as cv
import sys
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('soccer.jpg')

if img is None:
    sys.exit('file not found')

img = cv.resize(img, dsize=(0,0), fx=0.4, fy=0.4)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray16 = np.int16(gray)

#텍스트 추가

cv.putText(gray, 'soccer', (10,20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
cv.imshow('Original', gray)

# ================================================================================

#kernel == 필터
kernel = np.array([[-1.0, -1.0, -1.0],
                    [-1.0, 8.0, -1.0],
                    [-1.0, -1.0, -1.0]])

# 마찬가지로 음수 발생 문제로 128을 더함
result = np.uint8(np.clip(cv.filter2D(gray16, -1, kernel) + 128, 0, 255))
cv.imshow('Matrix', result)



cv.waitKey()
cv.destroyAllWindows()
