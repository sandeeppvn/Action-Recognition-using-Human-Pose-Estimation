import os
import argparse
from data_preparation import data_preprocessing
from action_recognition import action_recognition
from feature_extraction import feature_extraction
import numpy as np

def main():
    """Main function
    This is an action recognition computer vision project using pose estimation
    """

    # Take video file input path from user
    parser = argparse.ArgumentParser(description='Action Recognition')
    parser.add_argument('--input_path', type=str, help='Path to video file', default='Data/VIDEO_RGB')
    parser.add_argument('--sequences', type=str, help='Path to sequences.npy', default='Data/features.npy')
    parser.add_argument('--labels', type=str, help='Path to labels.npy', default='Data/labels_full.npy')
    parser.add_argument('--pose_estimation', type=str, help='Pose Estimation technique: Mediapipe', default='Mediapipe')
    parser.add_argument('--action_recognition', type=str, help='Action Recognition technique: LSTM', default='LSTM')
    parser.add_argument('--exp_name', type=str, help='Experiment name', default='lstm_full')
    parser.add_argument('--batch_size', type=int, help='Batch size', default=32)
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=50)
    parser.add_argument('--lr', type=float, help='Learning rate', default=0.001)
    parser.add_argument('--num_layers', type=int, help='Number of layers', default=2)
    parser.add_argument('--hidden_size', type=int, help='Hidden size', default=128)

    args = parser.parse_args()

    if not os.path.exists(args.sequences) or not os.path.exists(args.labels):
        sequences, labels = data_preprocessing(args.input_path)
        sequences = feature_extraction(sequences)

        np.save(args.sequences, sequences)
        np.save(args.labels, labels)
    else:
        sequences = np.load(args.sequences)
        labels = np.load(args.labels)

    # sequences = feature_extraction(sequences)
    

    params = {
        'exp_name': args.exp_name,
        'BATCH_SIZE': args.batch_size,
        'NUM_EPOCHS': args.epochs,
        'LEARNING_RATE': args.lr,
        'NUM_LAYERS': args.num_layers,
        'HIDDEN_SIZE': args.hidden_size
    }
        
    
    # Action recognition
    action_recognition(sequences,labels,args.action_recognition, params)



if __name__ == '__main__':
    # Set working directory to Project
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
