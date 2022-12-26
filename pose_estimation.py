import cv2
import numpy as np
import matplotlib.pyplot as plt

def pose_estimation(frame, technique):
    """
    Pose estimation function
    :param frame: Frame from video
    :param technique: Pose estimation technique
    :return: Pose points
    """

    if technique == 'Mediapipe':
        return mediapipe(frame)
    else:
        raise ValueError('Invalid pose estimation technique')


def mediapipe(frame):
    """
    MediaPipe pose estimation function
    :param frame: Frame from video
    :return: Pose points
    """
    import mediapipe as mp
    mp_drawing = mp.solutions.drawing_utils  # type: ignore
    mp_pose = mp.solutions.pose  # type: ignore

    # For static images:
    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=0,
            min_detection_confidence=0.5) as pose:
        result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        

        if not result.pose_landmarks:
            return None
        
        pose_point = np.array([[res.x, res.y, res.z, res.visibility] for res in result.pose_landmarks.landmark]).flatten()
        
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=result.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
        )

        return pose_point


def main():
    video_path = 'Data/VIDEO_RGB/smash/p4_smash_s2.avi'
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 45)
    ret, frame = cap.read()
    while not ret:
        ret, frame = cap.read()
        
    poses = pose_estimation(frame, 'Mediapipe')

    if poses is not None:
        plt.imshow(frame)
        for i in range(0, len(poses), 4):
            if poses[i+3] > 0.5:
                plt.scatter(poses[i]*frame.shape[1], poses[i+1]*frame.shape[0], s=20, c='r', marker='o')
        plt.show()
    
    cap.release()

if __name__ == '__main__':
    main()


