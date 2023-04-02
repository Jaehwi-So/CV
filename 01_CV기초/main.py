import cv2 as cv
import sys
img = cv.imread('soccer.jpg')

if img is None:
    sys.exit('file not found')

cv.imshow('Image Title', img)

cv.waitKey()
cv.destroyAllWindows()




