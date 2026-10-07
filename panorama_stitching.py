import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread("set2/42.png", 1)
img2 = cv2.imread("set2/33.png", 1)
img3 = cv2.imread("set2/30.png", 1)
img4 = cv2.imread("set2/24.png", 1)
img5 = cv2.imread("set2/11.png", 1)
img6 = cv2.imread("set2/4.png", 1)
img7 = cv2.imread("set2/15.png", 1)

#In order to match the corresponding points between every pair of images, the images are put into a list
images = [img1, img2, img3, img4, img5, img6, img7]

# Parameters
max_features = 500  # the maximum number of keypoints is 500

# ORB detector initialization
orb = cv2.ORB_create(max_features)

# DescriptorMatcher - BruteForce Hamming
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

# Processing the images in pairs
for i in range(len(images) - 1):
    imgA = images[i]
    imgB = images[i + 1]
    good_match_percentage = 0.18 #the same good_match_percentage for every pair of images
    # good_match_percentage= random.uniform(0.14, 0.20)  # keep only the best 14%-20% matches, variable for every pair of image

    #Grayscale conversion for fine detection of features
    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    #Detecting keypoints and calculating descriptors for A and B
    kp1, des1 = orb.detectAndCompute(grayA, None)
    kp2, des2 = orb.detectAndCompute(grayB, None)

    if des1 is not None and des2 is not None:  #to avoid errors (if des is none, then there is no match to be made. It must be different from none)
        # Brute-Force Matcher with Hamming (QueryDescriptor, TrainDescriptor)
        matches = matcher.match(des1, des2)
        #For every descriptor from des1 looks for the descriptor from des2 with the smallest Hamming distance
        # Sorting the matches by the attribute .distance (similarity score, Orb small--- good score)
        matches=sorted(matches, key=lambda x: x.distance)

        # Selecting the best matches
        numGoodMatches = int(len(matches) * good_match_percentage)
        good_matches = matches[:numGoodMatches] #takes the first matches

        #For every pair, only the good matches are displayed
        result_img = cv2.drawMatches(grayA, kp1, grayB, kp2, good_matches, None,
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        plt.figure()
        plt.title(f"Match: Img {i + 1} - Img {i + 2} | Best Matches: {len(good_matches)}")
        plt.imshow(result_img[..., ::-1])  # Conversion from BGR to RGB for plotting
        plt.axis('off')
        plt.show()



#-----------section 2
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt


max_features=1000    #ORB keypoints to detect
good_match_percent=0.18 #Only top 18% matches will be used
img1 = cv2.imread("set2/24.png", 1)
img2 = cv2.imread("set2/11.png", 1)
img3 = cv2.imread("set2/42.png", 1)

#images are converted to grayscale for feature detection
im1gray=cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
im2gray=cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
im3gray=cv2.cvtColor(im3,cv2.COLOR_BGR2GRAY)

#ORB detector initialization and paramters
orb=cv2.ORB_create(max_features)
keypoints1, descriptors1 = orb.detectAndCompute(im1gray,None)
keypoints2, descriptors2 = orb.detectAndCompute(im2gray,None)
keypoints3, descriptors3 = orb.detectAndCompute(im3gray,None)

#matching features between images, then sorting
matcher=cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
matches=matcher.match(descriptors1,descriptors2,None)
matches = sorted(matches, key=lambda x:x.distance) #sort by distance
numGoodMatches=int(len(matches)*good_match_percent)
matches=matches[:numGoodMatches] #keep only top matches

imMatches=cv2.drawMatches(im1gray,keypoints1,im2gray,keypoints2,matches,None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                          )

plt.figure()
plt.imshow(imMatches[:,:,::-1]), plt.title("Good Matches")
plt.show()

#Homography is computed using findHomography and RANSAC
points1=np.zeros((len(matches),2),np.float32)
points2=np.zeros((len(matches),2),np.float32)

for i, match in enumerate(matches):
    points1[i,:]=keypoints1[match.queryIdx].pt
    points2[i,:]=keypoints2[match.trainIdx].pt

# finding the homography
h,mask=cv2.findHomography(points2,points1,cv2.RANSAC)
print("Homography matrix \n{}".format(h))

#applying the perspective transformation h
im1height, im1width, im1channel = im1.shape
im2height, im2width, im2channel = im2.shape

#warp second image to align with the first image
im2Aligned=cv2.warpPerspective(im2,h,(im2width+im2height,im2height))

# stitch im1 with aligned im2
stitchedImage=np.copy(im2Aligned)
stitchedImage[0:im1height,0:im1width]=im1
plt.figure()
plt.imshow(stitchedImage[:,:,::-1]), plt.title("First Stiched Image (Im1+Im2)")
plt.show()

cropped_image = stitchedImage[0:1080,0:1500] #crop for display, parameters are chosen experimentally
#cropped_image = stitchedImage

#convert the result to grayscale
stitched_gray=cv2.cvtColor(cropped_image,cv2.COLOR_BGR2GRAY)

#here keypoints are detected on stiched image and im3
keypoints_s, descriptors_s = orb.detectAndCompute(stitched_gray,None)
keypoints3, descriptors3 = orb.detectAndCompute(im3gray,None)

#matching features between image 3 and stiched image
matcher_2=cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
matches_2=matcher_2.match(descriptors_s,descriptors3,None)
matches_2 = sorted(matches_2, key = lambda x:x.distance)

num2_Matches=int(len(matches_2)*good_match_percent)
matches_2=matches_2[:num2_Matches] ##takes the first matches

im_2_matches=cv2.drawMatches(stitched_gray, keypoints_s, im3gray, keypoints3, matches_2, None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

#Homography between stiched image and im3
points1_2=np.zeros((len(matches_2),2),np.float32)
points2_2=np.zeros((len(matches_2),2),np.float32)

for i, match in enumerate(matches_2):
    points1_2[i,:]=keypoints_s[match.queryIdx].pt
    points2_2[i,:]=keypoints3[match.trainIdx].pt

h_2,mask_2=cv2.findHomography(points2_2,points1_2,cv2.RANSAC,4.0)
print("Homography matrix \n{}".format(h_2))

#im3 is aligned and then combined
imCheight, imCwidth, imCchannel = cropped_image.shape
im3height, im3width, im3channel = im3.shape

im3Aligned=cv2.warpPerspective(im3,h_2, (im3width+im3height,im3height))
combined=np.copy(im3Aligned)
combined[0:imCheight,0:imCwidth]=cropped_image

plt.figure()
plt.imshow(combined[:,:,::-1]), plt.title("Final Image with borders")
plt.show()


#Automatically cropping the borders
#Convert the stitched image to grayscale and create a binary mask (every bit different from 0 is changed to white)
gray = cv2.cvtColor(combined, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

#Find the outer contour of the stitched panorama
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.RETR_EXTERNAL: find outermost contour
# cv2.CHAIN_APPROX_SIMPLE: compresses horizontal, vertical, and diagonal segments

if contours:
    largest_cnt = max(contours, key=cv2.contourArea) # Select the largest contour (the main stitched area)

#Create a mask of the same size as the image, initialized with zeros (black)
    mask = np.zeros(thresh.shape, dtype=np.uint8)
    x, y, w, h = cv2.boundingRect(largest_cnt) #Compute the bounding rectangle of the largest contour
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1) #Draw a white rectangle on the mask covering the bounding rect

#Erode the mask iteratively until it fits perfectly within the colored (non-black) area
    eroded_mask = mask.copy() #Copy mask to start erosion
    check_mask = cv2.subtract(eroded_mask, thresh) #crop final image

#While there are black pixels inside the rectangle, erode the rectangle
    while cv2.countNonZero(check_mask) > 0:
        eroded_mask = cv2.erode(eroded_mask, None) #Erode the rectangle by 1 pixel
        check_mask = cv2.subtract(eroded_mask, thresh) #Check again if any black pixels remain

#crop final clean image
    contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_cnt)
#Crop the final stitched image using the bounding rectangle
    final_image = combined[y:y+h, x:x+w]


    plt.figure()
    plt.imshow(final_image[:, :, ::-1])
    plt.title("Final Stitched Image-Black Borders Cropped")
    plt.show()