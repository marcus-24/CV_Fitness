# %%

# standard imports
import os
import cv2 as cv
import matplotlib.pyplot as plt
import mediapipe as mp
import pandas as pd
import itertools

# local imports
from posetools.loaddefaults import landmark_names

# Code Summary
# Modified example from the following pages:
# https://colab.research.google.com/drive/1uCuA6We9T5r0WljspEHWPHXCT_2bMKUy
# https://google.github.io/mediapipe/solutions/pose_classification.html

# TODO: Use dash module for MVP of exercise instructions

# %% Read image
img_fname = os.path.join('images', 'usain bolt.jpg')
image = cv.imread(img_fname)
img_height, img_width, _ = image.shape

# %% Load model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# %% Run MediaPipe Pose and draw pose landmarks
#TODO: Use opencv video capture to tag videos
# https://google.github.io/mediapipe/solutions/pose.html#python-solution-api
# https://appdividend.com/2022/03/19/python-cv2-videocapture/
#TODO: How are you going to sync the videos
n_frames = 1


with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=0) as pose:
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # convert to image to RGB
    results = pose.process(image_rgb)  # find landmarks in image

    # draw landmarks on new image
    annotated_image = image_rgb.copy()
    mp_drawing.draw_landmarks(annotated_image,
                              results.pose_landmarks,
                              mp_pose.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    # save landmarks
    current_landmarks = [[lmk.x, lmk.y, lmk.z] for lmk in results.pose_landmarks.landmark]  # TODO: Map outputs to landmarks in https://google.github.io/mediapipe/solutions/pose.html#pose-landmark-model-blazepose-ghum-3d
    # Store a 2D lists for data. Then put in dataframe. This is faster than appending/accessing dataframe on the fly

data = list(itertools.chain.from_iterable(current_landmarks)) # TODO need to change when multiple frames
columns = list(itertools.chain.from_iterable([[f'{n}_x', f'{n}_y', f'{n}_z'] for _, n in landmark_names.items()]))
pose_landmarks = pd.DataFrame([data], columns=columns)
# %% Plot results
plt.imshow(annotated_image)
plt.title("Output-Keypoints")


# %% Plot pose landmarks in 3-D plot
mp_drawing.plot_landmarks(results.pose_world_landmarks,
                          mp_pose.POSE_CONNECTIONS)


plt.show()
