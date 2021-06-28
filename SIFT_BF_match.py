import cv2 as cv

img1 = cv.imread("IMG1.jpeg")
img2 = cv.imread("IMG2.jpeg")

sift = cv.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv.BFMatcher()
matches = bf.match(des1,des2,None)

img = cv.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv.imwrite("SIFT_BF_match.jpeg", img)
