from random import randint

class RandomAgent:
    def get_action(self, state):
        return randint(0, 6)
