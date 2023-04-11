import cv2

# 이미지 읽어오기
img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

# Prewitt 마스크 정의
kernel_size = 3
prewittx = cv2.getDerivKernels(1, 0, kernel_size)[0] # x방향 미분
prewitty = cv2.getDerivKernels(0, 1, kernel_size)[0] # y방향 미분

# Prewitt 마스크 적용
prewitt_absx = cv2.convertScaleAbs(cv2.filter2D(img, -1, prewittx))
prewitt_absy = cv2.convertScaleAbs(cv2.filter2D(img, -1, prewitty))

# 엣지 임계값 적용
threshold_value = 0
max_value = 255
edges_x = cv2.threshold(prewitt_absx, threshold_value, max_value, cv2.THRESH_BINARY)[1]
edges_y = cv2.threshold(prewitt_absy, threshold_value, max_value, cv2.THRESH_BINARY)[1]

# 가중치 조합하여 이미지 생성
alpha = 0.5
beta = 0.5
gamma = 0
result = cv2.addWeighted(edges_x, alpha, edges_y, beta, gamma)

# 결과 이미지 출력
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()