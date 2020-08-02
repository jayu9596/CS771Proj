import cv2

image1 = cv2.imread('ACN.png')
image2 = cv2.imread('AJOX.png')

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray1 = cv2.copyMakeBorder(gray1, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
thresh1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
gray2 = cv2.copyMakeBorder(gray2, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
thresh2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

cv2.imwrite('acnthresh.png', thresh1)
cv2.imwrite('ajoxthresh.png', thresh2)