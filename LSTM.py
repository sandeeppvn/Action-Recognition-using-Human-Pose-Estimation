import tensorflow as tf
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout, LayerNormalization
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, f1_score


def lstm(X, y, params):
    """
    LSTM action recognition function
    :param data: Pose points from pose estimation
    :return: Action
    """

    exp_name = params['exp_name']
    BATCH_SIZE = int(params['BATCH_SIZE'])
    NUM_EPOCHS = int(params['NUM_EPOCHS'])
    LEARNING_RATE = params['LEARNING_RATE']
    NUM_LAYERS = params['NUM_LAYERS']
    HIDDEN_SIZE = params['HIDDEN_SIZE']
    EARLY_STOPPING_PATIENCE = 10

    log_dir = 'logs/' + exp_name
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    else:
        os.system('rm -rf ' + log_dir)
        os.makedirs(log_dir)
        
    tb_callback = TensorBoard(log_dir=log_dir)

    NUM_CLASSES = len(np.unique(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.11, random_state=42, stratify=y_train, shuffle=True)

    # One hot encoding
    one_hot_encoder = OneHotEncoder(sparse=False)
    y_train = one_hot_encoder.fit_transform(y_train.reshape(-1, 1))
    y_val = one_hot_encoder.transform(y_val.reshape(-1, 1))
    y_test = one_hot_encoder.transform(y_test.reshape(-1, 1))


    # Hyper parameters
    NO_OF_SEQUENCES, SEQUENCE_LENGTH, INPUT_SIZE = X_train.shape

    # Define model
    model = Sequential()
    model.add(LayerNormalization(input_shape=(SEQUENCE_LENGTH, INPUT_SIZE)))
    model.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True), input_shape=(SEQUENCE_LENGTH, INPUT_SIZE)))

    if NUM_LAYERS >= 2:
        for i in range(NUM_LAYERS-2):
            model.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True)))
    
        model.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=False)))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1)

    #Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Loss function
    loss = tf.keras.losses.CategoricalCrossentropy()

    # Compile model
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Train model
    history = model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[tb_callback, early_stopping]
    )


    

    # Model Summary
    model.summary()

    # Save model and weights
    if not os.path.isdir('saved_models'):
        os.makedirs('saved_models')

    # Delete old model
    if os.path.isfile('saved_models/{}.h5'.format(exp_name)):
        os.remove('saved_models/{}.h5'.format(exp_name))

    # Save model as tensorflow model
    model.save('saved_models/{}.h5'.format(exp_name))

    # Load model
    model = tf.keras.models.load_model('saved_models/{}.h5'.format(exp_name))

    # Evaluate model and print results
    loss, accuracy = model.evaluate(X_test, y_test)
    print('Loss: ', loss)
    print('Accuracy: ', accuracy)

    # Plot model
    tf.keras.utils.plot_model(model, to_file='saved_models/{}.png'.format(exp_name), show_shapes=True)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)

    # F1 score
    f1 = f1_score(y_test, y_pred, average='macro')

    # Generate Accuracy and Loss plots
    history_dict = history.history
    sns.set_style('darkgrid')
    sns.lineplot(data=history_dict['accuracy'], label='Training Accuracy')
    sns.lineplot(data=history_dict['val_accuracy'], label='Validation Accuracy')
    sns.lineplot(data=history_dict['loss'], label='Training Loss')
    sns.lineplot(data=history_dict['val_loss'], label='Validation Loss')


    # Save plot
    if not os.path.isdir('plots'):
        os.makedirs('plots')

    # Delete old plot
    if os.path.isfile('plots/{}_plot.png'.format(exp_name)):
        os.remove('plots/{}_plot.png'.format(exp_name))
        os.remove('plots/{}_cm.png'.format(exp_name))

    # Save plot
    plt.savefig('plots/{}_plot.png'.format(exp_name))

    # Clear plot
    plt.clf()

    # Generate and save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')

    # Save plot
    plt.savefig('plots/{}_cm.png'.format(exp_name))


    # Create dictionary to store results
    results = {
        'exp_name': exp_name,
        'Train Accuracy': history_dict['accuracy'][-1],
        'Validation Accuracy': history_dict['val_accuracy'][-1],
        'Test Accuracy': accuracy,
        'Train Loss': history_dict['loss'][-1],
        'Validation Loss': history_dict['val_loss'][-1],
        'Test Loss': loss,
        'F1 Score': f1,
        'LEARNING_RATE': LEARNING_RATE,
        'BATCH_SIZE': BATCH_SIZE,
        'HIDDEN_SIZE': HIDDEN_SIZE,
        'NUM_LAYERS': NUM_LAYERS,
        'NUM_EPOCHS': NUM_EPOCHS,
        'Early Stopping Patience': EARLY_STOPPING_PATIENCE,
        'Optimizer': 'Adam',
        'Loss Function': 'Categorical Crossentropy',
        'Model': 'Bidirectional LSTM'
    }

    # Save results
    if not os.path.isdir('results'):
        os.makedirs('results')

    # Delete old results
    if os.path.isfile('results/{}.csv'.format(exp_name)):
        os.remove('results/{}.csv'.format(exp_name))

    # Save results
    pd.DataFrame(results, index=[0]).to_csv('results/{}.csv'.format(exp_name), index=False)

    # Clear session
    tf.keras.backend.clear_session()
    
    return model



        

    


    