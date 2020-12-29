import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Reshape, Input, Dot
from tensorflow.keras.optimizers import Adam, SGD
import numpy as np
import random
from collections import deque

class Curiosity:
    def __init__(self, input_dimensions, output_dimensions, model_path=None):
        self.replay_mem = deque(maxlen=50000)
        self.min_replay_len = 1000
        self.batch_size = 64
        self.input_dimensions = input_dimensions
        self.output_dimensions = output_dimensions
        self.model = self.create_model(model_path)

    def create_model(self, model_path=None):
        if model_path:
            return load_model(model_path)

        model = Sequential()
        model.add(Dense(128, input_shape=self.input_dimensions))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(self.output_dimensions))
        model.add(Activation('linear'))

        model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['accuracy'])

        model.summary()
        return model

    def calc_reward(self, state, future_state):
        state = state.flatten()
        state = state.reshape(-1, state.shape[0])
        future_state = future_state.flatten()
        future_state = future_state.reshape(-1, future_state.shape[0])
        return self.model.evaluate(state, future_state, verbose=0)

    def train(self, state,  new_state):
        state = state.flatten()
        new_state = new_state.flatten()
        self.replay_mem.append((state, new_state))

        #if just started to play & replay mem not long enough
        #then don't train yet, play more
        if len(self.replay_mem) < self.min_replay_len:
            return

        #build batch from replay_mem
        batch = random.sample(self.replay_mem, self.batch_size)
        X = np.array([elem[0] for elem in batch])
        y = np.array([elem[1] for elem in batch])

        self.model.fit(X, y, batch_size=self.batch_size, shuffle=False, verbose=0)
