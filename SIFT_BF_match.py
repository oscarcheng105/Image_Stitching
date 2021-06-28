import cv2 as cv
import numpy as np
import warping

img1 = cv.imread("IMG1.jpeg")
img2 = cv.imread("IMG2.jpeg")

sift = cv.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv.BFMatcher()
matches = bf.match(des1,des2,None)

img = cv.drawMatches(img1, kp1, img2, kp2, matches, None, matchColor = (0,255,0), singlePointColor = (0,0,255), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

MIN_MATCH_COUNT = 10

if len(matches) > MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

    #WarpImages1
    #M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    #result = warping.warpImages1(img2, img1, M)

    #WarpImages2
    M, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0)
    result = warping.warpImages2(img2, img1, M)


cv.imshow("Match", img)
cv.imshow("Image Stitch", result)
k = cv.waitKey(0)
