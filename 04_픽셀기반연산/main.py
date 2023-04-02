import cv2 as cv
import sys
import numpy as np
img = cv.imread('soccer.jpg')

if img is None:
    sys.exit('file not found')

def cliping(f, value):
    f1 = f + value
    if(f1 > 255):
        f1 = 255
    elif(f1 < 0) :
        f1 = 0
    return f1

# 선형 연산
def linearAdd(f, value=0):
    clipingVec = np.vectorize(cliping)
    return np.uint8(clipingVec(f, value))  #numpy.uint8형으로 볂롼

# 감마 보정
def gamma(f, gamma=1.0):
    f1 = f/255.0    #numpy.float64형
    return np.uint8(255 * (f1**gamma))  #numpy.uint8형으로 볂롼

# 컬러 -> 명암
def colorToGray(f):
    f1 = np.round(0.299 * f[:, :, 2] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 0])
    return np.uint8(f1)  #numpy.uint8형으로 볂롼


#OpenCV의 img 영상은 numpy.ndarray 클래스의 객체
print(type(img))


#hstack으로 이어붙이기
gc = np.hstack((gamma(img, 0.5), gamma(img, 0.75), gamma(img, 1.0), gamma(img, 2.0), gamma(img, 3.0)))
gc2 = np.hstack((linearAdd(img, 20), linearAdd(img, 40)))
gc3 = colorToGray(img)
gc4 = cv.add(img, (20, 20, 20, 0))  #OpenCV 제공 선형연산


cv.imshow('Image Title', gc)
cv.imshow('Linear', gc2)
cv.imshow('Gray', gc3)
cv.imshow('Linear2', gc4)

cv.waitKey()
cv.destroyAllWindows()

