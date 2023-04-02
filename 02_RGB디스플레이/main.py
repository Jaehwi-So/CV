import cv2 as cv
import sys
img = cv.imread('soccer.jpg')

if img is None:
    sys.exit('file not found')

cv.imshow('original_RGB', img)

# // : 몫 계산
print(img.shape)    #(948, 1434, 3) 세로,가로,BGR
print(img.shape[0]//2)  #474
print(img.shape[1]//2)  # 717
cv.imshow('Upper left half', img[0:img.shape[0]//2, 0:img.shape[1]//2, :])
# 0~474, 0~717, BGR 전체

cv.imshow('Center half', img[img.shape[0]//4 : (3 * img.shape[0]//4), img.shape[1]//4 : (3 * img.shape[1]//4), :])
# 237~711, 358~1075, BGR 전체


cv.imshow('R Channel', img[:,:,2])  #R
cv.imshow('G Channel', img[:,:,1])  #G
cv.imshow('B Channel', img[:,:,0])  #B

cv.waitKey()
cv.destroyAllWindows()
