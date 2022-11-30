import os
import cv2
import numpy as np
from tqdm import tqdm
from pose_estimation import pose_estimation

from joblib import Parallel, delayed

def process_video(file):
    # Read and process video file
    cap = cv2.VideoCapture(file)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    SEQUENCE_LENGTH = 20

    if frame_count < SEQUENCE_LENGTH:
        return None, None

    skip_frame_window = max(frame_count / SEQUENCE_LENGTH, 1)

    # Get name of video file
    video_name = file.split('/')[-1].split('.')[0]

    
    actions_dict = {
        'backhand': 0, 'backhand2h': 0, 'bslice': 0, 'bvolley' :0,
        'forehand': 1, 'foreflat': 1, 'foreopen': 1, 'fslice': 1, 'fvolley': 1,
        'serve': 2, 'serflat': 2, 'serkick': 2, 'serslice': 2,
        'smash': 3
    }
    pose_technique = 'Mediapipe'

    

    window = []

    for i in range(frame_count):

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i*skip_frame_window))
        ret, frame = cap.read()
        if not ret:
            break

        # Pose estimation
        pose = pose_estimation(frame, pose_technique)
        if pose is None:
            return None, None
        window.append(pose)
        
    cap.release()

    return window, actions_dict[video_name.split('_')[1]]


def data_preprocessing(input_path):
    """
    From input path, extract all the files, perform pose estimation and return data
    :param path: input path
    :param pose_estimation: pose estimation technique   
    :param labels: list of labels
    :return: data
    """

    # Obtain all files in input path recursively inside subdirectories
    files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(input_path) for f in filenames if os.path.splitext(f)[1] == '.avi']

    # Parallel for loop
    vals = Parallel(n_jobs=-1)(delayed(process_video)(file) for file in tqdm(files))

    sequences = []
    labels = []
    if vals:
        for seq, lbl in vals:
            if seq is not None and lbl is not None:
                sequences.append(seq)
                labels.append(lbl)


    sequences = np.array(sequences, dtype=np.float64)
    labels = np.array(labels)
    
    return sequences, labels
    


