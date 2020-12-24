import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Reshape, Input, Dot
from tensorflow.keras.optimizers import Adam, SGD
from collections import deque
import numpy as np
import random

class Agent:
    def __init__(self, input_dimensions, output_dimensions, model_path=None):
        self.replay_mem_size = 50000
        self.batch_size = 64
        self.min_replay_len = 1000
        self.update_pred_model_period = 5
        self.epsilon = 1
        self.epsilon_decay = 0.999975
        self.min_epsilon = 0.1
        self.discount = 0.99
        self.success_margin = 1000
        self.input_dimensions = input_dimensions
        self.output_dimensions = output_dimensions
        self.replay_memory = deque(maxlen=self.replay_mem_size)
        self.model_update_counter = 0

        #main model that gets trained and predicts optimal action
        self.model = self.create_model(model_path)

        #Secondary model used to predict future Q values
        #makes predicting future Q vals more stable
        #more stable bcs multiple predictions from same reference point
        #model / reference point updated to match main model on chosen interval
        self.stable_pred_model = self.create_model()
        self.stable_pred_model.set_weights(self.model.get_weights())

    def create_model(self, model_path=None):
        if model_path:
            return load_model(model_path)

        model = Sequential()
        model.add(Dense(64, input_shape=self.input_dimensions))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(self.output_dimensions))
        model.add(Activation('linear'))

        model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['accuracy'])

        model.summary()
        return model

    def get_action(self, state):
        return np.argmax(self.model.predict(state.reshape(-1,state.shape[0])))

    def train(self, env_info):
        #env info: (state, action, new_state, reward, done)
        #add to replay memory
        self.replay_memory.append(env_info)

        #if just started to play & replay mem not long enough
        #then don't train yet, play more
        if len(self.replay_memory) < self.min_replay_len:
            return

        #build batch from replay_mem
        batch = random.sample(self.replay_memory, self.batch_size)
        #get output from network given state as input
        states = np.array([elem[0] for elem in batch])
        current_q_vals = self.model.predict(states)
        #predict future q (using other network) with new state
        new_states = np.array([elem[2] for elem in batch])
        future_q_vals = self.stable_pred_model.predict(new_states)
        #NOTE: its better to predict on full batch of states at once
        #   predicting gets vectorized :)

        X, y = [], []
        #populate X and y with state (input (X), & q vals (output (y))
        #must alter q vals in accordance with Q learning algorith
        #network will train to fit to qvals
        #this will fit the network towards states with better rewards
        #   (taking into account future rewards while doing so)

        for i, (state, action, new_state, reward, done) in enumerate(batch):
            #update q vals for action taken from state appropiately
            #if finished playing (win or lose), theres no future reward
            if done:
                current_q_vals[i][action] = reward
            else:
                #chose best action in new state
                optimal_future_q = np.max(future_q_vals[i])

                #Q-learning! :)
                current_q_vals[i][action] = reward + self.discount * optimal_future_q


            X.append(state)
            y.append(current_q_vals)

        self.model.fit(np.array(X), np.array(y), batch_size=self.batch_size, shuffle=False, verbose=0)

        #check if time to update prediction model
        #env_info[4]: done
        if env_info[4] and self.model_update_counter > self.update_pred_model_period:
            self.stable_pred_model.set_weights(self.model.get_weights())
            self.model_update_counter = 0
        elif env_info[4]:
            self.model_update_counter += 1
