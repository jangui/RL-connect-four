import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import math
from collections import deque
import random

class Agent:
    def __init__(self, model_path=None):
        self.discount = 0.99
        self.replay_mem_size = 5000
        self.minimum_replay_len = 150
        self.batch_size = 64
        self.replay_memory = deque(maxlen=self.replay_mem_size)
        self.priority_memory = deque(maxlen=50)
        self.model = self.create_model(model_path)


    def create_model(self, model_path):
        if model_path:
            return load_model(model_path)

        model = Sequential()
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(7, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

        return model

    def get_action(self, state, curiosity_values=None):
        state = state.reshape((1, 6, 7))
        prediction_list = self.model.predict(state, batch_size=1)
        self.q_values = prediction_list[0] # get first and only prediction
        if type(curiosity_values) != type(None):
            self.q_values += curiosity_values
        action = np.argmax(self.q_values)
        return action

    def get_next_action(self):
        first_choice = np.argmax(self.q_values)
        self.q_values[first_choice] = -math.inf
        next_choice = np.argmax(self.q_values)
        if math.isinf(self.q_values[next_choice]):
            return None
        return next_choice

    def add_data(self, training_data):
        # add training data to model's replay memory
        # training data: [state, action, new_state, reward, done]
        #   state: original state of agent
        #   action: action taken at original state
        #   new_state: new state arrived at after taken action
        #   reward: reward recieved at new state
        #   done: game state (True/False if game finished or not)
        self.replay_memory.append(training_data)

    def add_priority_data(self, training_data):
        self.priority_memory.append(training_data)

    def train(self):
        """
        Function for training the network.
        Data must be added to the replay memory with the add_data method
        """

        # don't start training until minimum data points collected
        if len(self.replay_memory) < self.minimum_replay_len:
            return

        #build batch from replay_mem
        batch = random.sample(self.replay_memory, self.batch_size)

        if len(self.priority_memory) > 0:
            priority_experience = random.choice(self.priority_memory)
            batch[0] = priority_experience

        # used to fit model
        # X = state
        # y = q values of the state with the q value for action taken updated according to reward
        X, y = [], []

        # get all states in our batch
        states = [elem[0] for elem in batch]

        # get all q values for all states in training data list
        current_q_vals = self.model.predict(np.array(states), batch_size=self.batch_size)

        # for each training data in the list
        #   find new q value
        #   update q values for that state
        for i, (state, action, new_state, reward, done) in enumerate(batch):
            if done:
                new_q_val = reward
            else:
                new_state = new_state.reshape((1,6,7))
                max_future_q_val = max(self.model.predict(new_state, batch_size=1)[0])
                new_q_val = reward + self.discount * max_future_q_val

            # update current state's q values with the new q value for action taken
            current_q_vals[i][action] = new_q_val

            X.append(state)
            y.append(current_q_vals[i])

        # train! :)
        self.model.fit(np.array(X), np.array(y), batch_size=self.batch_size, verbose=0)


class Curiosity:
    def __init__(self, model_path=None):
        self.replay_mem_size = 5000
        self.minimum_replay_len = 150
        self.batch_size = 64
        self.replay_memory = deque(maxlen=self.replay_mem_size)
        self.model = self.create_model(model_path)

    def predict(self, state):
        return self.model.predict(state.reshape((1,6,7)), batch_size=1)

    def create_model(self, model_path):
        if model_path:
            return load_model(model_path)

        model = Sequential()
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(42, activation='tanh'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

        return model

    def calc_reward(self, state, new_state):
        state = state.reshape((1, 6, 7))
        new_state = new_state.reshape((1,42))
        return self.model.evaluate(state, new_state, batch_size=1, verbose=0)[0]


    def add_data(self, training_data):
        """
        add data for training
            training_data: [new_state, future_state]
                new_state: state of board after agent makes move
                future_state: state of board when rival makes move
        """
        self.replay_memory.append(training_data)

    def train(self):
        """
        train curiosity module
        data must be added to the replay memory with the add_data method
        """
        if len(self.replay_memory) < self.minimum_replay_len:
            return

        #build batch from replay_mem
        batch = random.sample(self.replay_memory, self.batch_size)

        X, y = [], []
        for (state, new_state) in batch:
            X.append(state)
            y.append(new_state.flatten())

        self.model.fit(np.array(X), np.array(y), batch_size=self.batch_size, verbose=0)


