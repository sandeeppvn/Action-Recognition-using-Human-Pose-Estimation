# This sciprt will install all the required packages and libraries for the project. 
# It will also download the required models for pose estimation and action recognition.

import os
import site
def main():
    
    # Install required packages
    print('Installing required packages...')
    # packages = ["mediapipe", "tensorflow", "opencv-python", "pandas", "matplotlib", "numpy", "tqdm", "scikit-learn", "scikit-image", "scipy", "seaborn", "transformers", "joblib", "pyyaml", "h5py"]
    packages = [
        "opencv-python",
        "seaborn",
        "numpy",
        "matplotlib",
        "joblib",
        "scikit-learn",
        "tqdm",
        "pydot"
    ]
    for package in packages:
        print(f'Installing {package}...')
        os.system("pip install " + package + " --upgrade")
        
    print('Done with pip installs')

    # Get the operating system is macOS with M1 chip
    if os.uname().machine == 'arm64':
        print('Installing tensorflow-macos...')

        os.system("conda install apple -c apple tensorflow-deps")

        additonal_packages = ["tensorflow-macos", "tensorflow-metal", "mediapipe-silicon","protobuf"]
        for package in additonal_packages:
            print(f'Installing {package}...')
            os.system("pip install " + package + " --upgrade")

        site_packages_path = site.getsitepackages()[0]
        protobuf_path = os.path.join(site_packages_path, 'google/protobuf/internal/builder.py')
        # Move the builder.py file to current directory
        print(f'Moving {protobuf_path} to current directory')
        os.system(f'mv {protobuf_path} .')
        # Downgrade the protobuf version to 3.19.4
        os.system('pip install protobuf==3.19.4')
        # Move the builder.py file back to site-packages
        print(f'Moving builder.py to {protobuf_path}')
        os.system(f'mv builder.py {protobuf_path}')
        print('Move complete')

        os.system('brew install graphviz')


    
    else:
        print('Installing tensorflow...')
        additonal_packages = ["tensorflow", "mediapipe"]
        for package in additonal_packages:
            print(f'Installing {package}...')
            os.system("pip install " + package + " --upgrade")

    print('Done with Additional pip installs')

    

    

if __name__ == '__main__':
    main()