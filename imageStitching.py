import cv2 as cv
import numpy

img1 = cv.imread("IMG_9191.jpeg")
img2 = cv.imread("IMG_9192.jpeg")

orb = cv.ORB_create()

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)

img = cv.drawMatches(img1, kp1, img2, kp2, matches ,None, flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv.imwrite("ORB_image.jpeg", img)