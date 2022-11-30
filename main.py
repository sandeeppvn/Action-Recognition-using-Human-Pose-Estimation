import os
import argparse
from data_preparation import data_preprocessing
from action_recognition import action_recognition
import numpy as np

def main():
    """Main function
    This is an action recognition computer vision project using pose estimation
    """

    # Take video file input path from user
    parser = argparse.ArgumentParser(description='Action Recognition')
    parser.add_argument('--input_path', type=str, help='Path to video file', default='Data/VIDEO_RGB')
    parser.add_argument('--sequences', type=str, help='Path to sequences.npy', default='Data/sequences.npy')
    parser.add_argument('--labels', type=str, help='Path to labels.npy', default='Data/labels.npy')
    parser.add_argument('--pose_estimation', type=str, help='Pose Estimation technique: OpenPose, AlphaPose, PoseFlow, Mediapipe', default='Mediapipe')
    parser.add_argument('--action_recognition', type=str, help='Action Recognition technique: LSTM, CNNLSTM, ConvLSTM, ViT, ViViT', default='LSTM')
    args = parser.parse_args()

    # if not os.path.exists(args.sequences) or not os.path.exists(args.labels):
    if True:
        sequences, labels = data_preprocessing(args.input_path)
        np.save(args.sequences, sequences)
        np.save(args.labels, labels)
    else:
        sequences = np.load(args.sequences)
        labels = np.load(args.labels)
        
    # Action recognition
    action_recognition(sequences,labels,args.action_recognition)



if __name__ == '__main__':
    # Set working directory to Project
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()




