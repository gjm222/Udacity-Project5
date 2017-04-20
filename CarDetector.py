import cv2
import glob
import numpy as np
import pickle
import time

from moviepy.editor import VideoFileClip

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from CarDetectorFunctions import *

framesq = []  
maxframes = 15

dist_pickle = pickle.load( open("train_pickle.p", "rb" ) )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]

def process_image(image):
    global plotcount
    global totimgs
    global framesq
    global svc
    global X_scaler
    global orient
    global pic_per_cell
    global cell_per_block
    global spatial_size
    global hist_bins
    
    #image = mpimg.imread(fold2+'bbox-example-image.jpg')
    #draw_image = np.copy(image)
    
    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    #image = image.astype(np.float32)/255     
    
    #Find cars using one HOy_start_stop = [450, 600] # Min and max in y to search in slide_window()
    #Threshold boxes    
    y_start_stop = [400, 660] # Min and max in y to search in slide_window()
    ystart = y_start_stop[0] #ystart = 400
    ystop = y_start_stop[1] #ystop = 656
    
    scale = 2.0    
    out_img, heatmap1 = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    
    
    #Threshold boxes    
    y_start_stop = [400, 600] # Min and max in y to search in slide_window()
    ystart = y_start_stop[0] #ystart = 400
    ystop = y_start_stop[1] #ystop = 656
    
    scale = 1.5
    out_img1, heatmap2 = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    
    #Threshold boxes    
    y_start_stop = [400, 500] # Min and max in y to search in slide_window()
    ystart = y_start_stop[0] #ystart = 400
    ystop = y_start_stop[1] #ystop = 656
    
    scale = 1.0
    out_img2, heatmap3 = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    
    
    #Add all scaled heatmaps together
    heatmap = heatmap1 + heatmap2 + heatmap3
    
    #visual
    plt.figure(figsize=(6,12))
    if totimgs >= 20 and totimgs <= 25:
        #output marked up lanes
        #outname = 'output_images\\thresh.jpg'
        #cv2.imwrite(outname, (threshHeatMap * 255) 
        print(totimgs)        
        plt.imshow(heatmap, cmap='hot')
    
    
    #Threshold out the ones with not enough heat
    threshHeatMap = apply_threshold(heatmap, 2)
    
    #Add thresholed heatmap to frames queue
    framesq.append(threshHeatMap)
    if len(framesq)  > maxframes:
        framesq.pop(0)
        
    #Average out the heat maps
    newmap = np.zeros_like(heatmap)
    for heatmap in framesq:
        newmap += heatmap
     
    threshHeatMap = newmap // len(framesq)    
        
    
    #Heat labels
    labels = label(threshHeatMap)
    print(labels[1], 'cars found')
    
    #Cobine labeled boxes
    draw_img = np.copy(image)
    draw_img = draw_labeled_bboxes(draw_img, labels)
    
    #visual
    if totimgs == 25:    
        plt.figure(figsize=(6,12))
        plt.imshow(draw_img)
        plt.figure(figsize=(6,12))
        plt.imshow(labels[0], cmap='gray')
        
    
    totimgs += 1
    
    return draw_img
    #plt.imshow(draw_img)
    
    '''
    c = 2
    r = totimgs
    plotcount += 1
    plt.figure(figsize=(11,12))
    plt.subplot(r, c, plotcount)    
    plt.imshow(out_img)
    plotcount += 1
    plt.subplot(r, c, plotcount)    
    plt.imshow(draw_img)
    '''
    
    '''r = 4
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

plotcount = 1
totimgs = 0
plt.figure(figsize=(11,12))
framesq = []

'''    
images = glob.glob('C:\\Anaconda3_64\\CarND-Vehicle-Detection\\test_images\\*.jpg')
#C:\Anaconda3_64\CarND-Vehicle-Detection\test_images
totimgs = len(images)
for filename in images:
    image = mpimg.imread(filename)
    process_image(image)
'''

output_video = 'marked_video_visual.mp4'
input_video = 'test_video.mp4'
#input_video = 'project_video.mp4'

clip1 = VideoFileClip(input_video)
video_clip = clip1.fl_image(process_image)
video_clip.write_videofile(output_video, audio=False)

print("Done")        
    
###########################################################################



