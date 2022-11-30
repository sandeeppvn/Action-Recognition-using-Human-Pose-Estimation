import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from LSTM import LSTM_keras
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

def action_recognition(X, y, action_recognition_technique):
    """
    Action recognition function
    :param pose_points: Pose points from pose estimation
    :param action_recognition_technique: Action recognition technique
    :return: Action
    """

    # Apply one-hot encoding to labels
    onehot_encoder = OneHotEncoder(sparse=False)
    y = onehot_encoder.fit_transform(np.array(y).reshape(-1, 1))
    

    if action_recognition_technique == 'LSTM':
        lstm(X, y)

    else:
        raise ValueError('Invalid action recognition technique')

def lstm(X, y):
    """
    LSTM action recognition function
    :param data: Pose points from pose estimation
    :return: Action
    """

    log_dir = 'logs'
    tb_callback = TensorBoard(log_dir=log_dir)

    NUM_CLASSES = 4

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

    # Hyper parameters
    NO_OF_SEQUENCES, SEQUENCE_LENGTH, INPUT_SIZE = X_train.shape
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001

    # Define model
    model = LSTM_keras(INPUT_SIZE, SEQUENCE_LENGTH, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'])

    # Train model
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(X_val, y_val), callbacks=[tb_callback])

    # Evaluate model and print results
    loss, accuracy = model.evaluate(X_test, y_test)
    print('Loss: ', loss)
    print('Accuracy: ', accuracy)



