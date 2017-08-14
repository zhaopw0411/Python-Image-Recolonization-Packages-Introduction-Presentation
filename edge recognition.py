import cv2.cv as cv
import cv2
import numpy as np

im=cv.LoadImage('WechatIMG92.jpeg', cv.CV_LOAD_IMAGE_COLOR)

# Laplace on a gray scale picture
gray = cv.CreateImage(cv.GetSize(im), 8, 1)
cv.CvtColor(im, gray, cv.CV_BGR2GRAY)

aperture=3

dst = cv.CreateImage(cv.GetSize(gray), cv.IPL_DEPTH_32F, 1)
cv.Laplace(gray, dst,aperture)

cv.Convert(dst,gray)

thresholded = cv.CloneImage(im)
cv.Threshold(im, thresholded, 50, 255, cv.CV_THRESH_BINARY_INV)

cv.ShowImage('Laplaced grayscale',gray)
cv2.imwrite('~/Users/horace/Documents/TensorFlow/linedetection.jpg',gray)
cv.WaitKey(0)
#------------------------------------

# Laplace on color
planes = [cv.CreateImage(cv.GetSize(im), 8, 1) for i in range(3)]
laplace = cv.CreateImage(cv.GetSize(im), cv.IPL_DEPTH_16S, 1)
colorlaplace = cv.CreateImage(cv.GetSize(im), 8, 3)

cv.Split(im, planes[0], planes[1], planes[2], None) #Split channels to apply laplace on each
for plane in planes:
    cv.Laplace(plane, laplace, 3)
    cv.ConvertScaleAbs(laplace, plane, 1, 0)

cv.Merge(planes[0], planes[1], planes[2], None, colorlaplace)

cv.ShowImage('Laplace Color', colorlaplace)
#-------------------------------------

cv.WaitKey(0)