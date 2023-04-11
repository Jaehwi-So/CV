import cv2 as cv
import sys
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('soccer.jpg')

if img is None:
    sys.exit('file not found')

img = cv.resize(img, dsize=(0,0), fx=0.4, fy=0.4)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#텍스트 추가

cv.putText(gray, 'soccer', (10,20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
cv.imshow('Original', gray)

# ================================================================================
# 엠보싱 필터 정의
femboss = np.array([[-1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0]])
# OpenCV는 기본적으로 부호 있는 8비트 정수 형식을 사용하기 때문에, 필터링 결과가 0보다 작으면 해당 픽셀의 값은 0이 됩니다.

# 필터 적용을 위해 데이터타입을 int16으로 변환 -> [-255~255]이외의 범위가 나올수 있기 때문
gray16 = np.int16(gray)

# 엠보싱 : 입체감 있는 효과를 부여하는 컴볼루젼 필터 적용
# 중앙의 픽셀을 강조하면서 흑백 이미지에 입체감을 더해주는 효과가 있습니다.

# cv2.filter2D() 함수를 이용해 필터를 적용합니다. 이때, 두번째 인자값으로 -1을 전달하면 출력 이미지의 데이터 타입이 입력 이미지와 같게 됩니다.
# 출력 이미지의 데이터를 uint8 타입으로 변환합니다.
# np.clip() 함수를 이용해 값의 범위를 0~255로 제한하여 이외의 값들은 0이나 255로 처리되도록 합니다

# 128을 더하는 이유는 컨볼루젼 결과값이 음수가 포함될수 있기 때문에 clip이 0으로 만들면 전반적으로 어두워져 조정하는 것
emboss = np.uint8(np.clip(cv.filter2D(gray16, -1, femboss) + 128, 0, 255))
emboss_not128 = np.uint8(np.clip(cv.filter2D(gray16, -1, femboss), 0, 255))
emboss_bad = np.uint8(cv.filter2D(gray16, -1, femboss) + 128)

# cv2.filter2D() 함수를 이용해 필터를 적용한 뒤, int16 타입을 uint8 타입으로 변환하지 않은 출력 이미지를 만듭니다.
emboss_worse = cv.filter2D(gray, -1, femboss)

cv.imshow('Emboss', emboss)
cv.imshow('Emboss_bad', emboss_bad)
cv.imshow('Emboss_not128', emboss_not128)
cv.imshow('Emboss_worse', emboss_worse)



cv.waitKey()
cv.destroyAllWindows()
