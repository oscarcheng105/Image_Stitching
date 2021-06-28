import cv2 as cv

#Load Image
img1 = cv.imread("IMG1.jpeg")
img2 = cv.imread("IMG2.jpeg")

#Initiate SIFT detector
sift = cv.SIFT_create()

#Find Keypoints and Descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

#Default BFMatcher
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)

#Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

img = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv.imwrite("SIFT_BF_knnMatch.jpeg", img)