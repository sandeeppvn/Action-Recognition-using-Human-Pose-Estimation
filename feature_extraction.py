import numpy as np

def feature_extraction(sequence):
    """
    Feature extraction function
    :param sequence: Sequence of poses
    :return: Feature vector
    This function takes into account the positions in the sequence and calculates velocity and acceleration
    """
    
    # joints_to_use = [11,12,13,14,15,16,21,22,23,24,25,26,27,28,29,30,31,32]
    # Use all joints
    joints_to_use = [i for i in range(33)]

    s1,s2,_ = sequence.shape
    s3 = int(len(joints_to_use)*6)
    
    final = np.zeros((s1,s2,s3))
    # For each image in the sequence
    for i in range(len(sequence)):
        # For each frame in the sequence
        for j in range(len(sequence[i])):
            # For each pose in joints to use (Every 4 values are the x,y,z and visibility of a pose point)
            for k in joints_to_use:

                # Get the x,y,z and visibility of the pose point
                x, y, z, visibility = sequence[i][j][k*4], sequence[i][j][k*4+1], sequence[i][j][k*4+2], sequence[i][j][k*4+3]
                x_prev, y_prev, z_prev, visibility_prev = sequence[i][j-1][k*4], sequence[i][j-1][k*4+1], sequence[i][j-1][k*4+2], sequence[i][j-1][k*4+3]
                x_prev2, y_prev2, z_prev2, visibility_prev2 = sequence[i][j-2][k*4], sequence[i][j-2][k*4+1], sequence[i][j-2][k*4+2], sequence[i][j-2][k*4+3]

                # Calculate velocity and acceleration
                if j == 0:
                    # If it is the first frame, velocity and acceleration are 0
                    velocity = 0
                    acceleration = 0
                else:
                    # Calculate velocity and acceleration using differential calculus
                    velocity = np.sqrt((x - x_prev)**2 + (y - y_prev)**2 + (z - z_prev)**2)
                    
                    if j == 1:
                        acceleration = 0
                    else:
                        acceleration = np.sqrt((velocity - np.sqrt((x_prev - x_prev2)**2 + (y_prev - y_prev2)**2 + (z_prev - z_prev2)**2))**2)


                # Add the pose to final
                final[i][j][joints_to_use.index(k)*6] = x
                final[i][j][joints_to_use.index(k)*6+1] = y
                final[i][j][joints_to_use.index(k)*6+2] = z
                final[i][j][joints_to_use.index(k)*6+3] = velocity
                final[i][j][joints_to_use.index(k)*6+4] = acceleration
                final[i][j][joints_to_use.index(k)*6+5] = visibility


    return final


def main():
    """
    Main function
    """
    sequence = np.load('Data/sequences.npy')
    feature_extraction(sequence)

if __name__ == '__main__':
    main()

    