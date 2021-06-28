import cv2 as cv
import numpy as np
import warping
img1 = cv.imread("IMG1.jpeg")
img2 = cv.imread("IMG2.jpeg")

# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)


# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

# ratio test as per Lowe's paper
good = []
for i,(m,n) in enumerate(matches):
    if m.distance < 0.75*n.distance:
        good.append(m)

MIN_MATCH_COUNT = 10

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    #WarpImages1
    #M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    #result = warping.warpImages1(img2, img1, M)

    #WarpImages2
    M, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0)
    result = warping.warpImages2(img2, img1, M)

    #Show matching points
    matchesMask = mask.ravel().tolist()

    h,w,d = img1.shape
    pts = np.float32([[0,0], [0,h-1], [w-1,h-1], [w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts, M)

    img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (0,0,255),
                   matchesMask = matchesMask,
                   flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

img = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

cv.imshow("Match", img)
cv.imshow("Image Stitch", result)
k = cv.waitKey(0)