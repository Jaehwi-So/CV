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
# Sobel 마스크 생성

# kernelx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
# kernely = np.array([[-1, 0, -1], [-2, 0, 2], [-1, 0, 1]])

# Sobel 마스크 정의
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

# 엣지 임계값 적용
threshold_value = 80
max_value = 255
edges_x = cv.threshold(cv.convertScaleAbs(sobelx), threshold_value, max_value, cv.THRESH_BINARY)[1]
edges_y = cv.threshold(cv.convertScaleAbs(sobely), threshold_value, max_value, cv.THRESH_BINARY)[1]

# 가중치 조합하여 이미지 생성
alpha = 0.5
beta = 0.5
gamma = 0
result = cv.addWeighted(edges_x, alpha, edges_y, beta, gamma)

# 결과 이미지 출력
cv.imshow('Result', result)
cv.waitKey(0)
cv.destroyAllWindows()
