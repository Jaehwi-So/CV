import cv2 as cv
import sys
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('lena.bmp')

if img is None:
    sys.exit('file not found')

# img = cv.resize(img, dsize=(0,0), fx=0.4, fy=0.4)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray16 = np.int16(gray)


# ==================================
# Prewitt 마스크 생성

kernelx = np.array([[1, 0, -1], [-1, 0, 1], [-1, 0, 1]])
kernely = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

# Prewitt 필터 적용
prewittx = cv.filter2D(gray16, -1, kernelx)
prewitty = cv.filter2D(gray16, -1, kernely)

prewitt_absx = cv.convertScaleAbs(prewittx)
prewitt_absy = cv.convertScaleAbs(prewitty)

prewitt = cv.addWeighted(prewitt_absx, 0.5, prewitt_absy, 0.5, 0)
# 결과 출력
# 엣지 강도 계산
# edge = np.sqrt(np.power(prewittx, 2) + np.power(prewitty, 2)).astype(np.uint8)
#
# # 엣지 강도 임계값 적용
threshold_value = 20
ret, threshold = cv.threshold(prewitt, threshold_value, 255, cv.THRESH_BINARY)


print(ret)

# 결과 출력
cv.imshow('Original', gray)
cv.imshow('Prewitt X', np.uint8(np.clip(prewittx, 0, 255)))
cv.imshow('Prewitt Y', np.uint8(np.clip(prewitty, 0, 255)))
cv.imshow('Prewitt ', np.uint8(np.clip(prewitt, 0, 255)))
cv.imshow('Threshold', np.uint8(np.clip(threshold, 0, 255)))

cv.waitKey()
cv.destroyAllWindows()
