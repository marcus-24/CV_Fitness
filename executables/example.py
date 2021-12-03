# %%
import os
import cv2 as cv
import matplotlib.pyplot as plt
import mediapipe as mp

# Code Summary
# Modified example from the following pages:
# https://colab.research.google.com/drive/1uCuA6We9T5r0WljspEHWPHXCT_2bMKUy

# %% Read image
img_fname = os.path.join('images', 'usain bolt.jpg')
image = cv.imread(img_fname)
img_height, img_width, _ = image.shape

# %% Load model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# %% Run MediaPipe Pose and draw pose landmarks
with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=0) as pose:
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # convert to image to RGB
    results = pose.process(image_rgb)  # find landmarks in image

    # draw landmarks on new image
    annotated_image = image_rgb.copy()
    mp_drawing.draw_landmarks(annotated_image,
                              results.pose_landmarks,
                              mp_pose.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

# %% Plot results
plt.imshow(annotated_image)
plt.title("Output-Keypoints")
plt.show()
