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


# ===============================================================



# ===============================================================

# Edge Detect - Laplacin

#kernel == 필터
kernel = np.array([[-1.0, -1.0, -1.0],
                    [-1.0, 8.0, -1.0],
                    [-1.0, -1.0, -1.0]])

# 마찬가지로 음수 발생 문제로 128을 더함
result = np.uint8(np.clip(cv.filter2D(gray16, -1, kernel) + 128, 0, 255))
cv.imshow('Matrix', result)


# ===============================================================


# 영상 영역밖 (0,0)이면 주변 3x3이 없음 ->
# 방법1. (Warp-around-convolution) 영상의 모서리가 연결된 것으로 처리
# 방법2. (Zero-Padding) 비어있는 셀은 0으로 처리


# ===============================================================
# 지역통과필터 -> 저주파 성분 통과, 고주파 성분 감쇄
# 이웃화소밝기값과 자신을 평균해주는 역할의 마스크
# 영상 흐림화
kernel = np.array([[1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0]]) / 9


lenna = cv.imread('lena.bmp')
# lenna_origin = cv.imread('test.jpg')
# lenna = cv.cvtColor(lenna_origin, cv.COLOR_BGR2GRAY)
lenna_16 = np.int16(lenna)

# 요래해도 되긴함
# matrix_result = cv.filter2D(lenna, -1, kernel)

matrix_result = np.uint8(np.clip(cv.filter2D(lenna_16, -1, kernel), 0, 255))
cv.imshow('Lenna-Origin', lenna)
cv.imshow('Lenna-LowPass', matrix_result)



# ===============================================================
# 고역통과필터 -> 저주파 성분 감쇄, 고주파 성분 통과
# 엣지와 같은 고주파 성분의 부분을 강조하는 샤프닝 효과

kernel = np.array([[-1.0, -1.0, -1.0],
                    [-1.0, 9.0, -1.0],
                    [-1.0, -1.0, -1.0]])



# matrix_result = np.clip(cv.filter2D(lenna, -1, kernel), 0, 255)
matrix_result = np.uint8(np.clip(cv.filter2D(lenna_16, -1, kernel), 0, 255))
cv.imshow('Lenna-Highpass', matrix_result)

# ===============================================================

# 경계선 검출

kernel = np.array([[-1.0, -1.0, 2.0],
                    [-1.0, 2.0, -1.0],
                    [2.0, -1.0, -1.0]]) * 30



# matrix_result = np.clip(cv.filter2D(lenna, -1, kernel), 0, 255)
matrix_result = np.uint8(np.clip(cv.filter2D(lenna_16, -1, kernel), 0, 255))
cv.imshow('Lenna-Edge', matrix_result)




# ==================================
# Prewitt 마스크 생성

kernelx = np.array([[1, 0, -1], [-1, 0, 1], [-1, 0, 1]])
kernely = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

# Prewitt 필터 적용
prewittx = cv.filter2D(lenna_16, -1, kernelx)
prewitty = cv.filter2D(lenna_16, -1, kernely)

prewitt_absx = cv.convertScaleAbs(prewittx)
prewitt_absy = cv.convertScaleAbs(prewitty)

prewitt = cv.addWeighted(prewitt_absx, 0.5, prewitt_absy, 0.5, 0)
# 결과 출력
# 엣지 강도 계산
# edge = np.sqrt(np.power(prewittx, 2) + np.power(prewitty, 2)).astype(np.uint8)
#
# # 엣지 강도 임계값 적용
threshold_value = 80
ret, threshold = cv.threshold(prewitt, threshold_value, 255, cv.THRESH_BINARY)

# 결과 출력
cv.imshow('Original', img)
cv.imshow('Prewitt X', np.uint8(np.clip(prewittx, 0, 255)))
cv.imshow('Prewitt Y', np.uint8(np.clip(prewitty, 0, 255)))
cv.imshow('Prewitt ', np.uint8(np.clip(prewitt, 0, 255)))
cv.imshow('Threshold', np.uint8(np.clip(threshold, 0, 255)))

cv.waitKey()
cv.destroyAllWindows()


