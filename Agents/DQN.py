import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import math
import random

class DQNAgent:
    def __init__(self, game, save_location = "./", model_name="dqn", model_path=None, training=True):
        self.game = game

        # training settings
        self.replay_mem_size = 10000
        self.replay_memory = deque(maxlen=self.replay_mem_size)
        self.minimum_replay_len = 3000
        self.batch_size = 64

        self.discount = 0.9
        self.lr = 0.00001 # learning rate

        self.epsilon = 1
        self.epsilon_decay = 0.999985
        self.epsilon_reintroduction = True
        self.epsilon_reintroduction_value = 0.5
        self.min_epsilon = 0.15

        self.training = training
        self.training_halt = False
        self.training_halt_percent_margin = 0.7

        self.target_model_update_period = 5
        self.model_update_counter = 0

        # stats settings
        self.display_stats = True
        self.aggregation_period = 300
        self.current_win_streak = 0
        self.best_win_streak = 0
        self.wins = 0
        self.win_streak = 0
        self.aggregate_wins = []
        self.aggregate_win_streaks = []

        # saving settings
        self.model_name = model_name
        self.save_location = save_location
        self.autosave_period = 1000
        self.good_model_win_threshold = 0.7

        # create models
        self.model = self.create_model(model_path)
        self.target_model = self.create_model(model_path)

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
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.lr))

        return model

    def get_action(self, board, player):
        # random action
        if self.epsilon > random.random() and self.training:

            # choose random action
            action = random.randint(0, 6)
            while not self.game.check_valid_move(action, board):
                action = random.randint(0, 6)
            return action

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

    def add_data(self, training_data):
        # add training data to model's replay memory
        # training data: [state, action, new_state, reward, done]
        #   state: original state of agent
        #   action: action taken at original state
        #   new_state: new state arrived at after taken action
        #   reward: reward recieved at new state
        #   done: game state (True/False if game finished or not)
        self.replay_memory.append(training_data)

    def train(self):
        """
        Function for training the network.
        Data must be added to the replay memory with the add_data method
        """
        if self.training_halt:
            return

        # don't start training until minimum data points collected
        if len(self.replay_memory) < self.minimum_replay_len:
            return

        #build batch from replay_mem
        batch = random.sample(self.replay_memory, self.batch_size)

        # get q vals for states and new states
        states = [elem[0] for elem in batch]
        new_states = [elem[2] for elem in batch]
        q_vals = self.model.predict(np.array(states), batch_size=self.batch_size)
        future_q_vals = self.target_model.predict(np.array(new_states), batch_size=self.batch_size)

        # get q values and update based on reward
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

    def post_episode(self, episode):
        self.update_target_model()
        self.decay_epsilon()
        self.halt_training()

        if episode % self.aggregation_period == 0:
            # update aggr stats
            self.aggregate_wins.append(self.wins)
            self.aggregate_win_streaks.append(self.best_win_streak)

            # display aggr stats
            if self.display_stats:
                self.display_aggregate_stats()

            self.save_good_model(episode)
            self.reset_aggregate_stats()

        # auto save
        if episode % self.autosave_period == 0:
            self.autosave(episode)

    def halt_training(self):
        if self.wins > self.training_halt_percent_margin * self.aggregation_period:
            self.training_halt = True
        else:
            self.training_halt = False

    def decay_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
        else:
            if self.epsilon_reintroduction:
                self.epsilon = self.epsilon_reintroduction_value

    def update_target_model(self):
        if len(self.replay_memory) < self.minimum_replay_len:
            return

        if self.model_update_counter > self.target_model_update_period:
            self.target_model.set_weights(self.model.get_weights())
            self.model_update_counter = 0
        else:
            self.model_update_counter += 1

    def save_good_model(self, episode):
        if self.wins >= self.aggregation_period * self.good_model_win_threshold:
            save_name = f"{self.model_name}-{episode}-w{self.wins}-s{self.best_win_streak}"
            self.model.save(self.save_location+save_name)

    def autosave(self, episode):
        save_name = f"{self.model_name}-autosave{episode}"
        self.model.save(self.save_location+save_name)

    def display_aggregate_stats(self):
        print(f"DQN {self.model_name}")
        print(f"\tAggregate Wins: {self.wins}")
        print(f"\tBest Win Strream: {self.win_streak}")
        print(f"Epsilon: {self.epsilon}")

    def reset_aggregate_stats(self):
        self.wins = 0
        self.win_streak = 0

    def won(self):
        self.wins += 1
        self.current_win_streak += 1
        if self.current_win_streak > self.best_win_streak:
            self.best_win_streak = self.current_win_streak

    def lost(self):
        self.current_win_streak = 0

    def plot_aggr_results(self):
        plt.plot((self.aggregate_wins, self.aggregate_win_streaks), len(self.aggregate_wins))
        plt.title(f"")
        plt.show()

