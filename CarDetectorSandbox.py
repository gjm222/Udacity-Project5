import cv2
import glob
import numpy as np
import pickle
import time

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from CarDetectorFunctions import *



# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 06:15:54 2017

@author: SamKat2
"""

fold = 'images\\cutouts\\'

#Do not use: Test to see bounding boxes on images done manually
'''image = mpimg.imread('images\\cutouts\\bbox-example-image.jpg')

# Add bounding boxes in this format, these are just example coordinates.
bboxes = [((100, 100), (200, 200)), ((300, 300), (400, 400))]
bboxes = [((275, 572), (380, 510)), ((488, 563), (549, 518)), ((554, 543), (582, 522)), 
          ((601, 555), (646, 522)), ((657, 545), (685, 517)), ((849, 678), (1135, 512))]
result = draw_boxes(image, bboxes)
plt.imshow(result)
'''
#Do not use: Strict template matching
'''
#image = mpimg.imread('bbox-example-image.jpg')

image = mpimg.imread('images\\cutouts\\bbox-example-image.jpg')
templist = [fold + 'cutout1.jpg', fold + 'cutout2.jpg', fold + 'cutout3.jpg',
            fold + 'cutout4.jpg', fold + 'cutout5.jpg', fold + 'cutout6.jpg']
bboxes = find_matches(image, templist)
result = draw_boxes(image, bboxes)
plt.imshow(result)
'''
#Histogram - Histogram of R G B color intensity values
'''

image = mpimg.imread(fold+'cutout1.jpg')
rh, gh, bh, bincen, feature_vec = color_hist2(image, nbins=32)

# Plot a figure with all three bar charts
if rh is not None:
    fig = plt.figure(figsize=(12,3))
    plt.subplot(131)
    plt.bar(bincen, rh[0])
    plt.xlim(0, 256)
    plt.title('R Histogram')
    plt.subplot(132)
    plt.bar(bincen, gh[0])
    plt.xlim(0, 256)
    plt.title('G Histogram')
    plt.subplot(133)
    plt.bar(bincen, bh[0])
    plt.xlim(0, 256)
    plt.title('B Histogram')
    fig.tight_layout()
else:
    print('Your function is returning None for at least one variable...')
'''
#Visual analysis of color spaces
'''
# Read a color image

#img = cv2.imread(fold+"000275.png")
#img = cv2.imread(fold+"000528.png")
img = cv2.imread(fold+"3.png")

# Select a small fraction of pixels to plot by subsampling it
scale = max(img.shape[0], img.shape[1], 64) / 64  # at most 64 rows and columns
img_small = cv2.resize(img, (np.int(img.shape[1] / scale), np.int(img.shape[0] / scale)), interpolation=cv2.INTER_NEAREST)

# Convert subsampled image to desired color space(s)
img_small_RGB = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR, matplotlib likes RGB
img_small_HSV = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
img_small_LUV = cv2.cvtColor(img_small, cv2.COLOR_BGR2LUV)
img_small_rgb = img_small_RGB / 255.  # scaled to [0, 1], only for plotting

# Plot and show
plot3d(img_small_RGB, img_small_rgb)
plt.show()

plot3d(img_small_HSV, img_small_rgb, axis_labels=list("HSV"))
plt.show()

plot3d(img_small_LUV, img_small_rgb, axis_labels=list("LUV"))
plt.show()
'''
#spatial binning - Reduce resolution and combine channels to get 1-d array
#of identifying values
'''
image = mpimg.imread(fold+'cutout1.jpg')
feature_vec = bin_spatial2(image, color_space='LUV', size=(16, 16))

# Plot features
plt.plot(feature_vec)
plt.title('Spatially Binned Features')
'''
#Get dataset info
'''
cars = []
notcars = []

#Vehicle
fold = 'images\\vehicles_smallset\\cars1\\'
images = glob.glob(fold + '*.jpeg')
for image in images:
    cars.append(image)
    
#Non vehicle
fold = 'images\\non-vehicles_smallset\\notcars1\\'        
images = glob.glob(fold + '*.jpeg')        
for image in images:
     notcars.append(image)
         
#compile results     
data_info = data_look(cars, notcars)

print('Your function returned a count of', 
      data_info["n_cars"], ' cars and', 
      data_info["n_notcars"], ' non-cars')
print('of size: ',data_info["image_shape"], ' and data type:', 
      data_info["data_type"])
# Just for fun choose random car / not-car indices and plot example images   
car_ind = np.random.randint(0, len(cars))
notcar_ind = np.random.randint(0, len(notcars))
    
# Read in car / not-car images
car_image = mpimg.imread(cars[car_ind])
notcar_image = mpimg.imread(notcars[notcar_ind])


# Plot the examples
fig = plt.figure()
plt.subplot(121)
plt.imshow(car_image)
plt.title('Example Car Image')
plt.subplot(122)
plt.imshow(notcar_image)
plt.title('Example Not-car Image')     
'''    

#HOG
'''
cars = []
notcars = []
#Vehicle
fold = 'images\\vehicles_smallset\\cars1\\'
images = glob.glob(fold + '*.jpeg')
for image in images:
    cars.append(image)
#Non vehicle
fold = 'images\\non-vehicles_smallset\\notcars1\\'        
images = glob.glob(fold + '*.jpeg')        
for image in images:
     notcars.append(image)
     
# Generate a random index to look at a car image
ind = np.random.randint(0, len(cars))
# Read in the image
image = mpimg.imread(cars[ind])
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# Define HOG parameters
orient = 9
pix_per_cell = 8
cell_per_block = 2
# Call our function with vis=True to see an image output
features, hog_image = get_hog_features(gray, orient, 
                        pix_per_cell, cell_per_block, 
                        vis=True, feature_vec=False)


# Plot the examples
fig = plt.figure()
plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title('Example Car Image')
plt.subplot(122)
plt.imshow(hog_image, cmap='gray')
plt.title('HOG Visualization')
'''

# Multiple features
'''cars = []
notcars = []
#Vehicle
fold = 'images\\vehicles_smallset\\cars1\\'
images = glob.glob(fold + '*.jpeg')
for image in images:
    cars.append(image)
#Non vehicle
fold = 'images\\non-vehicles_smallset\\notcars1\\'        
images = glob.glob(fold + '*.jpeg')        
for image in images:
     notcars.append(image)
     
car_features = extract_features(cars, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32)
notcar_features = extract_features(notcars, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32)

if len(car_features) > 0:
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    car_ind = np.random.randint(0, len(cars))
    # Plot an example of raw and scaled features
    fig = plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.imshow(mpimg.imread(cars[car_ind]))
    plt.title('Original Image')
    plt.subplot(132)
    plt.plot(X[car_ind])
    plt.title('Raw Features')
    plt.subplot(133)
    plt.plot(scaled_X[car_ind])
    plt.title('Normalized Features')
    fig.tight_layout()
else: 
    print('Your function only returns empty feature vectors...')
'''

#Classify - color only
'''
cars = []
notcars = []
#Vehicle
fold = 'images\\vehicles_smallset\\cars1\\'
images = glob.glob(fold + '*.jpeg')
for image in images:
    cars.append(image)
#Non vehicle
fold = 'images\\non-vehicles_smallset\\notcars1\\'        
images = glob.glob(fold + '*.jpeg')        
for image in images:
     notcars.append(image)
     
# TODO play with these values to see how your classifier
# performs under different binning scenarios
spatial = 8
histbin = 128

car_features = extract_features(cars, color_space='RGB', spatial_size=(spatial, spatial),
                        hist_bins=histbin, hog_feat=False)
notcar_features = extract_features(notcars, color_space='RGB', spatial_size=(spatial, spatial),
                        hist_bins=histbin, hog_feat=False)

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using spatial binning of:',spatial,
    'and', histbin,'histogram bins')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
'''

#Classify - HOG + Color
'''
cars = []
notcars = []
#Vehicle

fold = 'images\\vehicles_smallset\\cars1\\'
images = glob.glob(fold + '*.jpeg')
for image in images:
    cars.append(image)
#Non vehicle
fold = 'images\\non-vehicles_smallset\\notcars1\\'        
images = glob.glob(fold + '*.jpeg')        
for image in images:
     notcars.append(image)
   
#PNG    
images = glob.glob('images\\Train\\**\\*.png', recursive=True)
   
for image in images:
    if 'non-vehicles' in image:
        notcars.append(image)
    else:
        cars.append(image)        
 

# Reduce the sample size because HOG features are slow to compute
# The quiz evaluator times out after 13s of CPU time
sample_size = 500
cars = cars[0:sample_size]
notcars = notcars[0:sample_size]

### TODO: Tweak these parameters and see how the results change.
colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"

t=time.time()
car_features = extract_features(cars, color_space=colorspace, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel)
notcar_features = extract_features(notcars, color_space=colorspace, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract HOG features...')
# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
'''


#Slide windows
'''
image = mpimg.imread('images\\cutouts\\bbox-example-image.jpg')


windows = slide_window(image, x_start_stop=[None,None], y_start_stop=[None, None], 
                    xy_window=(128, 128), xy_overlap=(0.5, 0.5))

window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6)                    
plt.imshow(window_img)
'''

#Find Cars extracting hog from every window
cars = []
notcars = []
#Vehicle
'''fold = 'images\\vehicles_smallset\\cars1\\'
images = glob.glob(fold + '*.jpeg')
for image in images:
    cars.append(image)
#Non vehicle
fold = 'images\\non-vehicles_smallset\\notcars1\\'        
images = glob.glob(fold + '*.jpeg')        
for image in images:
     notcars.append(image)
'''     
'''
images = glob.glob('images\\Train\\**\\*.png', recursive=True)
   
for image in images:
    if 'non-vehicles' in image:
        notcars.append(image)
    else:
        cars.append(image)    
     

# Reduce the sample size because
# The quiz evaluator times out after 13s of CPU time
sample_size = 500
cars = cars[0:sample_size]
notcars = notcars[0:sample_size]

### TODO: Tweak these parameters and see how the results change.
color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [450, 600] # Min and max in y to search in slide_window()

car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X

print('X=', X)
print(X.shape)
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()

fold2 = 'images\\cutouts\\'
image = mpimg.imread(fold2+'bbox-example-image.jpg')
draw_image = np.copy(image)
image = image.astype(np.float32)/255

# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
#image = image.astype(np.float32)/255

windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                    xy_window=(96, 96), xy_overlap=(0.5, 0.5))

hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)                       

window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)                    

plt.imshow(window_img)
'''





#Find cars using one HOG
'''
cars = []
notcars = []
#Vehicle


images = glob.glob('images\\Train\\**\\*.png', recursive=True)
   
for image in images:
    if 'non-vehicles' in image:
        notcars.append(image)
    else:
        cars.append(image)    
     
print('notcars=', len(notcars))        
print('cars=', len(cars))        
###############
# Reduce the sample size because
# The quiz evaluator times out after 13s of CPU time
sample_size = 3000
cars = cars[0:sample_size]
notcars = notcars[0:sample_size]

# TRAIN 
color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [500, 656] # Min and max in y to search in slide_window()

car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

print(len(car_features[0]))
print(len(notcar_features[0]))

X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
print('X shape', X.shape)

scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()


#Save off calibration so we wont have to do it again
dpickle = {}
dpickle["svc"] = svc
#dpickle["dist"] = dist
pickle.dump(dpickle, open('train_pickle.p', 'wb'))  
print('Training data saved')     





fold2 = 'images\\cutouts\\'
image = mpimg.imread(fold2+'bbox-example-image.jpg')
#draw_image = np.copy(image)

# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
#image = image.astype(np.float32)/255     

#Find cars using one HOG
ystart = y_start_stop[0] #ystart = 400
ystop = y_start_stop[1] #ystop = 656

scale = 1.5
print('Find cars')    
out_img, heatmap = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

#Threshold boxes
threshHeatImg = apply_threshold(heatmap, 2)

#Heat labels
labels = label(threshHeatImg)
print(labels[1], 'cars found')

#Cobine labeled boxes
draw_img = np.copy(image)
draw_img = draw_labeled_bboxes(draw_img, labels)

#plt.imshow(draw_img)

r = 4
c = 2
plt.figure(figsize=(11,12))
plt.subplot(r, c, 1)    
plt.imshow(out_img)
plt.subplot(r, c, 2)    
plt.imshow(heatmap)
plt.subplot(r, c, 3)    
plt.imshow(out_img)
plt.subplot(r, c, 4)    
plt.imshow(threshHeatImg)
plt.subplot(r, c, 5)    
plt.imshow(out_img)
plt.subplot(r, c, 6)    
plt.imshow(labels[0], cmap='gray')
plt.subplot(r, c, 7)    
plt.imshow(draw_img)
'''






'''
cspace = 'YCrCb' # RGB, HSV, LUV, HLS, YUV or YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL" # 0, 1, 2, or "ALL"

# Load file names
#cars = glob.glob('./vehicles/*/*.png')
cars = glob.glob('images\\Train\\**\\*.png', recursive=True)
# Get random image for vehicle
random_index = np.random.randint(0, len(cars))
img = cars[random_index]

features, hog_image = features_for_vis(mpimg.imread(img),
                                       cspace=cspace,
                                       pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel)
'''


dist_pickle = pickle.load( open("train_pickle.p", "rb" ) )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]

print(str(pix_per_cell))
print(str(spatial_size))
print(str(hist_bins))
print(str(orient))
print(svc)
print(X_scaler)


fold2 = 'images\\cutouts\\'
image = mpimg.imread(fold2+'bbox-example-image.jpg')
#draw_image = np.copy(image)

# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
#image = image.astype(np.float32)/255     

#Find cars using one HOy_start_stop = [450, 600] # Min and max in y to search in slide_window()
y_start_stop = [400, 660] # Min and max in y to search in slide_window()
ystart = y_start_stop[0] #ystart = 400
ystop = y_start_stop[1] #ystop = 656

scale = 1.2
print('Find cars')    
out_img, heatmap = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

#Threshold boxes
threshHeatImg = apply_threshold(heatmap, 1)

#Heat labels
labels = label(threshHeatImg)
print(labels[1], 'cars found')

#Cobine labeled boxes
draw_img = np.copy(image)
draw_img = draw_labeled_bboxes(draw_img, labels)

#plt.imshow(draw_img)

r = 4
c = 2
plt.figure(figsize=(11,12))
plt.subplot(r, c, 1)    
plt.imshow(out_img)
plt.subplot(r, c, 2)    
plt.imshow(heatmap)
plt.subplot(r, c, 3)    
plt.imshow(out_img)
plt.subplot(r, c, 4)    
plt.imshow(threshHeatImg)
plt.subplot(r, c, 5)    
plt.imshow(out_img)
plt.subplot(r, c, 6)    
plt.imshow(labels[0], cmap='gray')
plt.subplot(r, c, 7)    
plt.imshow(draw_img)


###########################################################################



