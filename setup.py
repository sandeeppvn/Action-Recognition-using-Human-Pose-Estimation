# This sciprt will install all the required packages and libraries for the project. 
# It will also download the required models for pose estimation and action recognition.

import os
def main():
    # Install required packages
    print('Installing required packages...')
    packages = ["mediapipe", "torch", "opencv-python", "pandas", "matplotlib", "numpy", "tqdm", "scikit-learn", "scikit-image", "scipy", "seaborn", "transformers", "joblib"]
    for package in packages:
        print(f'Installing {package}...')
        os.system("pip install " + package)
    print('Done')

if __name__ == '__main__':
    main()