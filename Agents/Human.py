
class Human:
    def __init__(self, game):
        self.game = game
        self.model_name = "human"

    def get_action(self, board, player):
        action = -1
        while action == -1:
            input_action = input("> ")
            try:
                action = int(input_action)
            except ValueError:
                print("invalid action")
                continue

            if not self.game.check_valid_move(action, board):
                action = -1
                print("invalid action")
        return action

    # functions needed to train but not used by this agent

    def add_data(self, training_data, winner, win_type):
        pass
    def train(self):
        pass
    def post_episode(self, episode):
        pass
    def won(self):
        pass
    def lost(self):
        pass
    def verbose(self):
        pass
    def plot_results(self):
        pass
