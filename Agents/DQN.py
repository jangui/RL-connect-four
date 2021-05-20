import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
from copy import deepcopy
import math
import random

class DQNAgent:
    def __init__(self, game, max_depth=3, save_location = "./", model_name="dqn", model_path=None, training=True):
        self.game = game
        self.max_depth = max_depth

        # training settings
        #self.replay_mem_size = 20000
        self.replay_mem_size = 5000
        self.replay_memory = deque(maxlen=self.replay_mem_size)
        self.minimum_replay_len = 64
        #self.minimum_replay_len = 5000
        #self.batch_size = 128
        self.batch_size = 64

        self.discount = 0.9
        #self.lr = 0.00001 # learning rate
        self.lr = 0.0001 # learning rate

        self.epsilon = 1
        #self.epsilon_decay = 0.999985
        self.epsilon_decay = 0.9999
        self.epsilon_reintroduction = False
        self.epsilon_reintroduction_value = 0.5
        self.min_epsilon = 0.15

        self.training = training
        self.training_halt = False
        self.training_halt_percent_margin = 0.7

        self.target_model_update_period = 5
        self.model_update_counter = 0

        self.reward = 1
        self.reward_discount = 0.825

        self.debug = 0

        # stats settings
        self.stats = False
        self.aggregation_period = 300
        self.current_win_streak = 0
        self.best_win_streak = 0
        self.wins = 0
        self.win_streak = 0
        self.aggregate_wins = []
        self.aggregate_win_streaks = []
        self.episode = 1

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
        """
        # random action
        if self.epsilon > np.random.random() and self.training:

            # choose random action
            action = np.random.randint(0, 7)
            while not self.game.check_valid_move(action, board):
                action = np.random.randint(0, 7)
            return action
        """


        # get action from network
        _, action = self.maximize(board, player, 0, -math.inf, math.inf)
        return action

    def get_q_values(self, board, player):
        # network doesn't know what player it is
        # board gets fixed to always be in same prespective
        state = deepcopy(board)
        state = state.reshape((1, 6, 7))
        if player == -1:
            state *= -1

        batch_of_predictions = self.model.predict(state, batch_size=1)
        q_values = batch_of_predictions[0]

        if self.debug:
            print("\n", board)
            print(f"{self.model_name} q vals: {q_values}\n")

        return q_values

    def maximize(self, board, player, depth, alpha, beta):
        if self.game.check_win(board, test=True):
            return None, None
        if self.game.is_full(board):
            return None, None
        if depth >= self.max_depth-1:
            # return utility & action of action with highest q value that is a valid move
            q_values = self.get_q_values(board, player)
            action = np.argmax(q_values)
            while not self.game.check_valid_move(action, board):
                q_values[action] = -math.inf
                action = np.argmax(q_values)
            utility = q_values[action]
            return utility, action
        else:
            max_util = -math.inf
            max_util_action = None
            q_values = None
            for action in range(7):
                # check if action valid
                if not self.game.check_valid_move(action, board):
                    continue
                new_state = self.game.move(action, board, test=True, action = player)
                min_util, _ = self.minimize(new_state, -player, depth+1, alpha, beta)

                # if node returned None then it is a terminal state
                # thus we need to get its q value its parent (this node)
                if min_util == None:
                    if type(q_values) == type(None):
                        q_values = self.get_q_values(board, player)
                    min_util = q_values[action]

                if min_util >= max_util:
                    max_util = min_util
                    max_util_action = action
                if min_util >= alpha:
                    alpha = min_util
                if alpha >= beta:
                    break
            return max_util, max_util_action

    def minimize(self, board, player, depth, alpha, beta):
        if self.game.check_win(board, test=True):
            return math.inf, None
        if self.game.is_full(board):
            return None, None
        if depth >= self.max_depth-1:
            # return utility & action of action with lowest q value that is a valid move
            q_values = -self.get_q_values(board, player)
            action = np.argmin(q_values)
            while not self.game.check_valid_move(action, board):
                q_values[action] = math.inf
                action = np.argmin(q_values)
            utility = q_values[action]
            return utility, action
        else:
            min_util = math.inf
            min_util_action = None
            q_values = None
            for action in range(7):
                # check if action valid
                if not self.game.check_valid_move(action, board):
                    continue
                new_state = self.game.move(action, board, test=True, action = player)
                max_util, _ = self.maximize(new_state, -player, depth+1, alpha, beta)

                # if node returned None then it is a terminal state
                # thus we need to get its q value its parent (this node)
                if max_util == None:
                    if type(q_values) == type(None):
                        q_values = self.get_q_values(board, player)
                    max_util = q_values[action]

                if max_util <= min_util:
                    min_util = max_util
                    min_util_action = action
                if max_util <= beta:
                    beta = max_util
                if beta <= alpha:
                    break
            return min_util, min_util_action

    def add_data(self, training_data, winner, win_type):
        # add training data to model's replay memory
        # training data: [ ..., [state, action, new_state, reward, done], ... ]
        #   state: original state of agent
        #   action: action taken at original state
        #   new_state: new state arrived at after taken action
        #   reward: reward recieved at new state
        #   done: game state (True/False if game finished or not)
        #   player: 1 or -1, 1 = first to move, -1 = second
        reversed_data = deepcopy(training_data)
        reversed_data.reverse()

        if self.debug:
            print(f"\n{self.model_name}")

        reward = self.reward
        """
        if win_type == 2:
            # minimal reward for vertical wins
            reward /= 1000
        """
        for env_info in reversed_data:
            reward_index = 3
            env_info[reward_index] += reward
            if winner == 1:
                # agent doent know what team he's on
                # board must always be in prespective of same team
                state_ind = 0
                new_state_ind = 2
                env_info[state_ind] *= -1
                env_info[new_state_ind] *= -1

            if self.debug:
                for data in env_info:
                    print(data)
                print()

            self.replay_memory.append(env_info)

            # decrease reward in proportion to actions leading to win
            #reward = round(reward * self.reward_discount, 8)

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

    def post_episode(self):
        self.update_target_model()
        #self.decay_epsilon()
        #self.halt_training()

        if self.stats:
            if self.episode % self.aggregation_period == 0:
                # update aggr stats
                self.aggregate_wins.append(self.wins)
                self.aggregate_win_streaks.append(self.best_win_streak)

                self.display_aggregate_stats()

                if self.debug:
                    self.plot_results()

                self.save_good_model(self.episode)
                self.reset_aggregate_stats()

        # auto save
        if self.episode % self.autosave_period == 0:
            self.autosave(self.episode)

        self.episode += 1

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
        print(f"\tBest Win Streak: {self.best_win_streak}")
        print(f"Epsilon: {self.epsilon}")

    def reset_aggregate_stats(self):
        self.wins = 0
        self.current_win_streak = 0
        self.best_win_streak = 0

    def won(self):
        self.wins += 1
        self.current_win_streak += 1
        if self.current_win_streak > self.best_win_streak:
            self.best_win_streak = self.current_win_streak

    def lost(self):
        self.current_win_streak = 0

    def plot_results(self):
        plt.plot(self.aggregate_wins)
        plt.plot(self.aggregate_win_streaks)
        plt.title(f"{self.model_name} Aggregate Results")
        plt.xlabel(f"Training Time ({self.aggregation_period}'s of Games)")
        plt.ylabel("Wins")
        plt.legend(["Aggregate Wins", "Best Win Streak"])
        plt.show()
        plt.close()
        plt.clf()

    def verbose(self):
        self.debug = (self.debug + 1) % 2

