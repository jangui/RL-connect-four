import numpy as np

class RandomAgent:
    def __init__(self, game):
        self.game = game
        self.model_name = "random_agent"

    def get_action(self, state, player):
        action = np.random.randint(0, 7)
        while not self.game.check_valid_move(action, state):
            action = np.random.randint(0, 7)
        return action

    # functions needed to train but not used by this agent

    def add_data(self, training_data, winner, win_type):
        pass

    def train(self):
        pass

    def post_episode(self):
        pass

    def won(self):
        pass

    def lost(self):
        pass

    def verbose(self):
        pass

    def plot_results(self):
        pass
