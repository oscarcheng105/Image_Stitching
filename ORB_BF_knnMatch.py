import cv2 as cv

img1 = cv.imread("IMG1.jpeg", cv.IMREAD_GRAYSCALE)
img2 = cv.imread("IMG2.jpeg", cv.IMREAD_GRAYSCALE)

orb = cv.ORB_create()

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

matches = bf.knnMatch(des1,des2,k=2)

good = []
for m,n in matches:
    if m.distance <0.75*n.distance:
        good.append([m])

img = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv.imwrite("ORB_BF_knnMatch.jpeg", img)