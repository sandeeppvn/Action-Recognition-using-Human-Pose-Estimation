import os
import argparse
from keras.models import load_model
from data_preparation import data_preprocessing
from feature_extraction import feature_extraction
import cv2
import matplotlib.pyplot as plt
from pose_estimation import pose_estimation

def predict(input_path, exp_name):
    """Predict function
    This function predicts the action of the input video
    """

    # Load model
    model = load_model(os.path.join('saved_models/', exp_name + '.h5'))

    # Load video
    sequences, _ = data_preprocessing(input_path)
    sequences = feature_extraction(sequences)

    # Predict
    prediction = model.predict(sequences)

    # Get argmax
    prediction = prediction.argmax(axis=1)

    actions_dict = {
        'Backhand': 0, 'Backhand Volley': 1, 'Forehand': 2, 'Forehand Volley': 3,
        'Service': 4, 'Smash': 4
    }

    # Get action name
    action_name = [key for key, value in actions_dict.items() if value == prediction[0]]

    # Clear console with control + l
    os.system('cls' if os.name == 'nt' else 'clear')

    # Load the video
    cap = cv2.VideoCapture(input_path)
    
    # While the video is playing, print the action predicted and the expected action and wait till the user presses q
    video_frames = []
    joints_to_use = [11,12,13,14,15,16,21,22,23,24,25,26,27,28,29,30,31,32]
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        
        pose_points = pose_estimation(frame, 'Mediapipe')

        if pose_points is not None:
            # Every 4 points are a pose
            for i in joints_to_use:
                x, y, z, visibility = pose_points[i*4:(i+1)*4]
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

        cv2.putText(frame, 'Expected: {}'.format(input_path.split('/')[-2]), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, 'Action Predicted: {}'.format(action_name[0]), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Append the frame to the video_frames list
        video_frames.append(frame)


        
    cap.release()
    cv2.destroyAllWindows()

    print('Expected: {}'.format(input_path.split('/')[-2]))
    print('Action Predicted: {}'.format(action_name[0]))

    # Save the video
    height, width, layers = video_frames[0].shape
    size = (width, height)
    out = cv2.VideoWriter('results/{}+output.avi'.format(exp_name+input_path.split('/')[-1].split('.')[0]), cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(video_frames)):
        out.write(video_frames[i])
    out.release()

    # Save the middle frame
    cv2.imwrite('results/{}+shot_prediction.jpg'.format(exp_name+input_path.split('/')[-1].split('.')[0]), video_frames[len(video_frames)//2])

    
    

   

def main():

    parser = argparse.ArgumentParser(description='Action Recognition')
    parser.add_argument('--input_path', type=str, help='Path to video file', default='Data/VIDEO_RGB/backhand_volley/p3_bvolley_s3.avi')
    parser.add_argument('--exp_name', type=str, help='Path to model', default='bilstm_features_final')

    args = parser.parse_args()
    predict(args.input_path, args.exp_name)


if __name__ == '__main__':
    # Set working directory to Project
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    main()

