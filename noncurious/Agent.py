import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np

class Agent:
    def __init__(self, model_path=None):
        self.model = self.create_model(model_path)
        self.discount = 0.99

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

    def get_action(self, state):
        state = np.reshape(state, (1, 6, 7))
        prediction_list = self.model.predict(state)
        q_values = prediction_list[0] # get first and only prediction
        action = np.argmax(q_values)
        return action

    def train(self, training_data_list):
        """
        Function for training the network.
        Arguemnts:
            training_data_list: a list of training data

        training data: [state, action, new_state, reward, done]
            state: original state of agent
            action: action taken at original state
            new_state: new state arrived at after taken action
            reward: reward recieved at new state
            done: game state (finished or not)

        """
        # used to fit model
        # X = state
        # y = q values of the state with the q value for action taken updated according to reward
        X, y = [], []

        # get all states in training data list
        states = [elem[0] for elem in training_data_list]

        # get all q values for all states in training data list
        current_q_vals = self.model.predict(states)

        # for each training data in the list
        #   find new q value
        #   update q values for that state
        for i, (state, action, new_state, reward, done) in enumerate(training_data_list):
            if done:
                new_q_val = reward
            else:
                max_future_q_val = max(self.model.predict((new_state)))
                new_q_val = reward + self.discount * max_future_q_val

            # update current state's q values with the new q value for action taken
            current_q_vals[i][action] = new_q_val

            X.append(state)
            y.append(current_q_vals[i])

        # train! :)
        self.model.fit(X, y, verbose=0)


class Curiosity:
    def __init__(self):
        return
