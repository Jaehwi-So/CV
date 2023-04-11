import cv2 as cv
import sys
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('lena.bmp')

if img is None:
    sys.exit('file not found')

img = cv.resize(img, dsize=(0,0), fx=0.4, fy=0.4)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray16 = np.int16(gray)

#텍스트 추가

cv.putText(gray, 'soccer', (10,20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
cv.imshow('Original', gray)


# 영상 영역밖일 경우 : (0,0)이면 주변 3x3이 없음 ->
# 방법1. (Warp-around-convolution) 영상의 모서리가 연결된 것으로 처리
# 방법2. (Zero-Padding) 비어있는 셀은 0으로 처리


# ===============================================================
# 지역통과필터 -> 저주파 성분 통과, 고주파 성분 감쇄
# 이웃화소밝기값과 자신을 평균해주는 역할의 마스크
# 영상 흐림화
kernel = np.array([[1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0]]) / 9

# 요래해도 되긴함
# matrix_result = cv.filter2D(gray, -1, kernel)

result = np.uint8(np.clip(cv.filter2D(gray16, -1, kernel), 0, 255))
cv.imshow('Lenna-Origin', gray16)
cv.imshow('Lenna-LowPass', result)



# ===============================================================
# 고역통과필터 -> 저주파 성분 감쇄, 고주파 성분 통과
# 엣지와 같은 고주파 성분의 부분을 강조하는 샤프닝 효과

kernel = np.array([[-1.0, -1.0, -1.0],
                    [-1.0, 9.0, -1.0],
                    [-1.0, -1.0, -1.0]])



# matrix_result = np.clip(cv.filter2D(lenna, -1, kernel), 0, 255)
result = np.uint8(np.clip(cv.filter2D(gray16, -1, kernel), 0, 255))
cv.imshow('Lenna-Highpass', result)


cv.waitKey()
cv.destroyAllWindows()
