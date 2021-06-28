import cv2 as cv

img1 = cv.imread("IMG1.jpeg", cv.IMREAD_GRAYSCALE)
img2 = cv.imread("IMG2.jpeg", cv.IMREAD_GRAYSCALE)

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
for i, m_n in enumerate(matches):
  if len(m_n) != 2:
    continue
  elif m_n[0].distance < 0.75*m_n[1].distance:
    matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (0,0,255),
                   matchesMask = matchesMask,
                   flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

img = cv.drawMatchesKnn(img1,kp1,img2,kp2, matches,None,**draw_params)

cv.imwrite("ORB_FLANN_knnMatch.jpeg", img)