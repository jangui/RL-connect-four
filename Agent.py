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
        model.add(Reshape((384,), input_shape=self.input_dimensions))
        model.add(Activation('relu'))
        model.add(Dropout(0.1))

        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.1))

        model.add(Dense(self.output_dimensions))
        model.add(Activation('linear'))

        model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['accuracy'])

        """
        optimizer = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
        input_layer = Input(shape=(6, 8, 8), name='board_layer')
        inter_layer_1 = Conv2D(1, (1, 1), data_format="channels_first")(input_layer)  # 1,8,8
        inter_layer_2 = Conv2D(1, (1, 1), data_format="channels_first")(input_layer)  # 1,8,8
        flat_1 = Reshape(target_shape=(1, 64))(inter_layer_1)
        flat_2 = Reshape(target_shape=(1, 64))(inter_layer_2)
        output_dot_layer = Dot(axes=1)([flat_1, flat_2])
        output_layer = Reshape(target_shape=(4096,))(output_dot_layer)
        model = Model(inputs=[input_layer], outputs=[output_layer])
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        """
        model.summary()
        return model

    def get_action(self, state):
        state = np.reshape(state, (1,6,8,8))
        prediction = self.model.predict(state)
        prediction = np.reshape(prediction[0], (64,64))
        move_from, move_to = np.unravel_index(prediction.argmax(), prediction.shape)
        return (move_from, move_to)

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
        states = np.array([elem[0].tolist() for elem in batch])
        current_q_vals = self.model.predict(states)
        #predict future q (using other network) with new state
        new_states = np.array([elem[2].tolist() for elem in batch])
        future_q_vals = self.stable_pred_model.predict(new_states)
        #NOTE: its better to predict on full batch of states at once
        #   predicting gets vectorized :)

        X, y = [], []
        #populate X and y with state (input (X), & q vals (output (y))
        #must alter q vals in accordance with Q learning algorith
        #network will train to fit to qvals
        #this will fit the network towards states with better rewards
        #   (taking into account future rewards while doing so)
        current_q_vals = np.reshape(current_q_vals, (len(batch), 64, 64))
        future_q_vals = np.reshape(future_q_vals, (len(batch), 64, 64))

        for i, (state, action, new_state, reward, done) in enumerate(batch):
            #update q vals for action taken from state appropiately
            #if finished playing (win or lose), theres no future reward
            if done:
                current_q_vals[i][action[0]][action[1]] = reward
            else:
                #chose best action in new state
                optimal_future_q = np.max(future_q_vals[i])

                #Q-learning! :)
                current_q_vals[i][action[0]][action[1]] = reward + self.discount * optimal_future_q


            X.append(state)
            y.append(np.reshape(current_q_vals[i], (4096,)))

        self.model.fit(np.array(X), np.array(y), batch_size=self.batch_size, shuffle=False, verbose=0)

        #check if time to update prediction model
        #env_info[4]: done
        if env_info[4] and self.model_update_counter > self.update_pred_model_period:
            self.stable_pred_model.set_weights(self.model.get_weights())
            self.model_update_counter = 0
        elif env_info[4]:
            self.model_update_counter += 1
