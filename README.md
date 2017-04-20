# Udacity-Project5

## root Directory
| File        | Description   | 
|:-------------:|:-------------:| 
|CarDetector.py    | Car detection program| 
|CarDetectorFunctions.py  | Shared vehicle detection functions  |
|CarDetectorSandbox.py     | Used to test and for experimentation and to create image samples|
|CarDetectorTrainer.py     | Trains on sample data and saves the resulting training data to the train_pickle.p file to be used for vehicle detection.    |
|marked_video.mp4    | Resulting video after training on project_video.mp4  |
|train_pickle.p     | Holds the traing data |
|writeup_project5     | Summary of project and results  |

## output_images Directory
| File        | Description   | 
|:-------------:|:-------------:| 
|car-hog.jpg    | Car and HOG image     | 
|car-noncar.jpg  | Car and Non-car image  |
|heat1.jpg     | Heatmap sequence 1     |
|heat2.pg     | Heatmap sequence 2     |
|heat3.jpg     | Heatmap sequence 3     |
|heat4.jpg     | Heatmap sequence 4     |
|heat5.jpg     | Heatmap sequence 5    |
|heat6.jpg     | Heatmap sequence 6     |
|heatmap.jpg     | Original image overlayed with bounding boxes and corresponding heatmap    |
|labeled.jpg     | Original image overlayed with bounding boxes     |
|labeled2.jpg     | Original image overlayed with bounding boxes which is result of using heat1...6 heatmaps    |
|labeled3.jpg     | Image showing scipy.ndimage.measurements.label() function as a result of using heat1...6 heatmaps    |
|notcar-hog.jpg     | Non-Car and HOG image    |
|sliding.jpg     | Original image overlayed with sliding windows    |
|sliding2.jpg     | Original image overlayed with sliding windows    |
