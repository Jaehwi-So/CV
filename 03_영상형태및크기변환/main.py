import cv2 as cv
import sys


img = cv.imread('soccer.jpg')

if img is None:
    sys.exit('file not found')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  #컬러 영상 -> 명암 영상
gray_small = cv.resize(gray, dsize=(0,0), fx=0.5, fy=0.5)   #반으로 축소

cv.imwrite('soccer_gray.jpg', gray) #Output 영상 저장
cv.imwrite('soccer_gray_small.jpg', gray_small) #Output 영상 저장

cv.imshow('1', gray)
cv.imshow('2', gray_small)

cv.waitKey()
cv.destroyAllWindows()

