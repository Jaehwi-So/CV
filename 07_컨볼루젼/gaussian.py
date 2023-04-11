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
# 가우시안 스무딩 : 스무딩 필터, 블러처리, 컨볼루젼(평활화)의 종류

smooth = np.hstack((cv.GaussianBlur(gray, (5,5), 0.0), cv
    .GaussianBlur(gray, (9,9), 0.0), cv.GaussianBlur(gray, (15,15),0.0)))

cv.imshow('Smooth', smooth)



cv.waitKey()
cv.destroyAllWindows()
