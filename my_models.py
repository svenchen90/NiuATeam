from keras.models import Sequential
from keras.layers import Conv1D, Dense, LSTM, RepeatVector, TimeDistributed, Bidirectional, Flatten, Dropout, Reshape
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

def CNN_model(input_length, output_length):
    model = Sequential()
    
    model.add(Reshape((input_length, 1), input_shape=(input_length,)))
    Reshape((3, 4), )
    # layer 1
    model.add(Conv1D(64, 64, activation="relu", padding="same", strides=1))

    # layer 2
    model.add(Conv1D(32, 32, activation="relu", padding="same", strides=1))

    # layer 3
    model.add(Conv1D(16, 16, activation="relu", padding="same", strides=1))

    # layer 4
    model.add(Conv1D(16, 8, activation="relu", padding="same", strides=1))

    # layer 5
    model.add(Conv1D(16, 4, activation="relu", padding="same", strides=1))

    # layer 6
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    
    # output layer
    model.add(Dense(output_length))

    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model


def RNN_model(input_length, output_length):
    model = Sequential()
    
    
    model.add(Reshape((input_length, 1), input_shape=(input_length,)))
    # 1D Conv
    model.add(Conv1D(16, 4, activation="linear", input_shape=(input_length, 1), padding="same", strides=1))

    #Bi-directional LSTMs
    model.add(Bidirectional(LSTM(128, return_sequences=True, stateful=False), merge_mode='concat'))
    model.add(Bidirectional(LSTM(256, return_sequences=True, stateful=False), merge_mode='concat'))

    # Fully Connected Layers
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(1, activation='linear'))
    
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    
    # output layer
    model.add(Dense(output_length))
    
    model.compile(loss='mse', optimizer='adam',metrics=['mae'])
    model.summary()
    # plot_model(model, to_file='model.png', show_shapes=True)

    return model

