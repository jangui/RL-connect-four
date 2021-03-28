import random

class RandomAgent:
    def __init__(self, game):
        self.game = game

    def get_action(self, state):
        action = random.randint(0, 6)
        while not self.game.check_valid_move(action, state):
            action = random.randint(0, 6)
        return action
