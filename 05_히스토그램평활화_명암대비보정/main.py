import cv2 as cv
import sys
import matplotlib.pyplot as plt

img = cv.imread('soccer.jpg')

if img is None:
    sys.exit('file not found')


gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  #명암 영상으로 변환
plt.imshow(gray, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.show()

h = cv.calcHist([gray], [0], None, [256], [0, 256]) #히스토그램 출력
plt.plot(h, color='r', linewidth=1), plt.show()

equal = cv.equalizeHist(gray)   #히스토그램 평활화 후 출력
plt.imshow(equal, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.show()

h = cv.calcHist([equal], [0], None, [256], [0, 256]) #히스토그램 출력
plt.plot(h, color='r', linewidth=1), plt.show()

cv.imshow('original', gray)
cv.imshow('eqv', equal)

cv.waitKey()
cv.destroyAllWindows()


# hist = cv2.calcHist(images, channels, mask, histSize, ranges(, hist(, accumulate)))
# h = cv.calcHist([gray], [0], None, [256], [0, 256])
#
# images : 히스토그램을 계산할 영상의 배열입니다.
# channels : 히스토그램을 계산할 channel의 배열. RGB면 channels이 3개
# mask : images[i]와 같은 크기의 8bit 이미지로, mask(x, y)가 0이 아닌 경우에만 image[i](x,y)을 히스토그램 계산에 사용합니다., None이면 마스크를 사용하지 않고, 모든 화소에서 히스토그램을 계산합니다.
# histSize : 히스토그램 hist 크기에 대한 정수 배열
# ranges : 히스토그램 각 빈의 경계값에 대한 배열입니다. opencv는 기본적으로 등간격 히스토그램
# accumulate : True 이면 calcHist() 함수를 수행할 때, 히스토그램을 초기화 하지 않고, 이전 값을 계속 누적합니다.
# hist : 히스토그램 리턴값