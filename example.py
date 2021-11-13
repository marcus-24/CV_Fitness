import cv2 as cv
import os
import json
import matplotlib.pyplot as plt

""" Code Summary
Code Example from Geeksforgeeks
https://www.geeksforgeeks.org/python-opencv-pose-estimation/
"""

# %% Load model
model_dir = os.path.join('pose', 'mpi')  # model directory
model_arch = os.path.join(model_dir, 'pose_deploy_linevec.prototxt')  # model architecture settings
model_weights = os.path.join(model_dir, 'pose_iter_160000.caffemodel')  # model weights

net = cv.dnn.readNetFromCaffe(model_arch, model_weights)

# %% Defining body parts from model
body_parts_fname = open(os.path.join(model_dir, 'body_parts.json'))
BODY_PARTS = json.load(body_parts_fname)

# %% Read image
img_fname = os.path.join('images', 'usain bolt.jpg')
image = cv.imread(img_fname)

# %% Specify the input image dimensions
in_height, in_width = image.shape[:2]

# %% Prepare the frame to be fed to the network
inpBlob = cv.dnn.blobFromImage(image,
                               scalefactor=1.0 / 255,
                               size=(in_width, in_height),
                               mean=(0, 0, 0),
                               swapRB=False,
                               crop=False)

# %% Set the prepared object as the input blob of the network
net.setInput(inpBlob)

output = net.forward()


out_height = output.shape[2]
out_width = output.shape[3]
# Empty list to store the detected keypoints
points = []
threshold = 0.05

for idx in range(len(BODY_PARTS)):
    # confidence map of corresponding body's part.
    prob_map = output[0, idx, :, :]

    # Find global maxima of the prob_map.
    _, prob, _, point = cv.minMaxLoc(prob_map)

    # Scale the point to fit on the original image
    x_coord = int((in_width * point[0]) / out_width)
    y_coord = int((in_height * point[1]) / out_height)

    cv.circle(image,
              center=(x_coord, y_coord),
              radius=15,
              color=(0, 255, 255),
              thickness=-1,
              lineType=cv.FILLED)

    cv.putText(image,
               text=f'{idx}',
               org=(x_coord, y_coord),
               fontFace=cv.FONT_HERSHEY_SIMPLEX,
               fontScale=1.4,
               color=(0, 0, 255),
               thickness=3,
               lineType=cv.LINE_AA)

    # Add the point to the list if the probability is greater than the threshold
    points.append((x_coord, y_coord) if prob > threshold else None)

plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.title("Output-Keypoints")
plt.show()
