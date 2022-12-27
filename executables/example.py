# %%

# standard imports
import os
import cv2 as cv
import mediapipe as mp

# local imports
from posetools.loaddefaults import landmark_names

# Code Summary
# Modified example from the following pages:
# https://colab.research.google.com/drive/1uCuA6We9T5r0WljspEHWPHXCT_2bMKUy
# https://google.github.io/mediapipe/solutions/pose_classification.html

# TODO: Use dash module for MVP of exercise instructions

# %% Read image
vid_fname = os.path.join('videos', 'running.mp4') #from <a href="https://www.vecteezy.com/video/10012531-woman-running-ocean-beach-young-asian-female-exercising-outdoors-running-seashore-concept-of-healthy-running-and-outdoor-exercise-active-sporty-athlete-jogging-summer-active">

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
with mp_pose.Pose(
     min_detection_confidence=0.15,
     min_tracking_confidence=0.5) as pose:

    cap = cv.VideoCapture(vid_fname)
    if not cap.isOpened():
        print('cant open file')

    while (cap.isOpened()):
        success, frame = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break

        scale_percent = 25
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_frame = cv.resize(frame, dim, interpolation=cv.INTER_AREA)

        # To improve performance, optionally mark the frame as not writeable to
        # pass by reference.
        resized_frame.flags.writeable = False
        resized_frame = cv.cvtColor(resized_frame, cv.COLOR_BGR2RGB)
        results = pose.process(resized_frame)

        # Draw the pose annotation on the frame.
        resized_frame.flags.writeable = True
        resized_frame = cv.cvtColor(resized_frame, cv.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(resized_frame,
                                  results.pose_landmarks,
                                  mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # Flip the frame horizontally for a selfie-view display.
        cv.imshow('MediaPipe Pose', resized_frame)
        if cv.waitKey(1) & 0xFF == 27:
            break

cap.release()