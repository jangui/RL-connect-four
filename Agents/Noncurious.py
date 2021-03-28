import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import math
from collections import deque
import random

class Noncurious:
    def __init__(self, game, model_path=None):
        self.game = game
        self.discount = 0.99
        self.replay_mem_size = 5000
        self.minimum_replay_len = 150
        self.batch_size = 64
        self.epsilon = 1
        self.epsilon_decay = 0.99985
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

    def get_action(self, board):
        # random action
        if self.epsilon > random.random():
            self.epsilon *= self.epsilon_decay
            action = random.randint(0, 6)
            while not self.game.check_valid_move(action, board):
                action = random.randint(0, 6)
            return action

        # agent action
        state = board.reshape((1, 6, 7))
        prediction_list = self.model.predict(state, batch_size=1)
        q_values = prediction_list[0]
        action = np.argmax(q_values)
        while not self.game.check_valid_move(action, board):
            q_values[action] = -math.inf
            action = np.argmax(q_values)
            if math.isinf(action):
                return None
        return action

    def get_next_action(self):
        first_choice = np.argmax(self.q_values)
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

        # make sure every batch has a priority experience
        if len(self.priority_memory) > 0:
            priority_experience = random.choice(self.priority_memory)
            batch[0] = priority_experience

        # get q vals for states and new states
        states = [elem[0] for elem in batch]
        new_states = [elem[2] for elem in batch]
        q_vals = self.model.predict(np.array(states), batch_size=self.batch_size)
        future_q_vals = self.model.predict(np.array(new_states), batch_size=self.batch_size)

        # get q values and update based on reward
        new_q_vals = []
        for i, (state, action, new_state, reward, done) in enumerate(batch):
            if done:
                new_q_val = reward
            else:
                # take future reward into account
                max_future_q_val = np.max(future_q_vals[i])
                new_q_val = reward + self.discount * max_future_q_val

            # update q value for action taken
            q_vals[i][action] = new_q_val

        # fit states to updated q values
        self.model.fit(np.array(states), q_vals, batch_size=self.batch_size, verbose=0)
