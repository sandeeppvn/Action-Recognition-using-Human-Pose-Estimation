import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Define class for LSTM model using Keras
class LSTM_keras(tf.keras.Model):
    def __init__(self, input_size, sequence_length, hidden_size, num_layers, num_classes):
        super(LSTM_keras, self).__init__()
        self.lstm = LSTM(hidden_size, return_sequences=True, input_shape=(sequence_length, input_size), activation='relu')
        self.lstm2 = LSTM(hidden_size, return_sequences=True, activation='relu')
        self.lstm3 = LSTM(hidden_size, return_sequences=False, activation='relu')
        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(32, activation='relu')
        self.fc3 = Dense(num_classes, activation='softmax')
        self.num_layers = num_layers
        
    def call(self, x):
        lstm_result = self.lstm(x)
        if self.num_layers > 2:
            for i in range(self.num_layers-2):
                lstm_result = self.lstm2(lstm_result)
        lstm_result = self.lstm3(lstm_result)
        
        fc1_result = self.fc1(lstm_result)
        fc2_result = self.fc2(fc1_result)
        fc3_result = self.fc3(fc2_result)
        return fc3_result

        

    


    