import os
import json

landmark_fname = os.path.join('posetools', 'landmarknames.json')
with open(landmark_fname, 'r') as f:
    landmark_names = json.load(f)