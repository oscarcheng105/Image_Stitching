import cv2 as cv
import numpy as np
import warping 

img1 = cv.imread("IMG1.jpeg")
img2 = cv.imread("IMG2.jpeg")

# Initiate SIFT detector
orb = cv.ORB_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params= dict(algorithm = 6, # FLANN_INDEX_LSH
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
good = []
for i, m_n in enumerate(matches):
  if len(m_n) != 2:
    continue
  if m_n[0].distance < 0.75*m_n[1].distance:
    matchesMask[i] = [1,0]
    good.append([m_n[0]])

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (0,0,255),
                   #matchesMask = matchesMask,
                   flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

img = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,**draw_params)

MIN_MATCH_COUNT = 10

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)

    #WarpImages1
    #M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    #result = warping.warpImages1(img2, img1, M)

    #WarpImages2
    M, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0)
    result = warping.warpImages2(img2, img1, M)

cv.imshow("Match", img)
cv.imshow("Image Stitch", result)
k = cv.waitKey(0)