from LSTM import lstm

def action_recognition(X, y, action_recognition_technique, params):
    """
    Action recognition function
    :param pose_points: Pose points from pose estimation
    :param action_recognition_technique: Action recognition technique
    :return: Action
    """

    # Combine labels
    y = combine_labels(y)

    if action_recognition_technique == 'LSTM':
        lstm(X, y, params)

    else:
        raise ValueError('Invalid action recognition technique')

def combine_labels(y):
    # actions_dict = {
    #     'backhand': 0, 'backhand2h': 1, 'bslice': 2, 'bvolley' :3,
    #     'forehand': 4, 'foreflat': 5, 'foreopen': 6, 'fslice': 7, 'fvolley': 8,
    #     'serve': 9, 'serflat': 10, 'serkick': 11, 'serslice': 12,
    #     'smash': 13
    # }

    # y is a numpy array
    y[y < 3] = -6
    y[y == 3] = -5
    y[(y > 3) & (y < 8)] = -4
    y[y == 8] = -3
    y[(y > 8) & (y < 13)] = -2
    y[y == 13] = -1

    # Add 6 to all values
    y += 6  
    
    return y

