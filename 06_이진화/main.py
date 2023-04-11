import cv2 as cv
import sys
import matplotlib.pyplot as plt

img = cv.imread('soccer.jpg')

if img is None:
    sys.exit('file not found')

# 이진화의 방법에는 곤잘레스와 오츄 방법이 있다.

t, bin_img = cv.threshold(img[:,:,2], 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# 오츄 알고리즘이 찾은 최적 임계값
# print(t)

cv.imshow('Origin R', img[:,:,2])
cv.imshow('Binarization', bin_img)


cv.waitKey()
cv.destroyAllWindows()

