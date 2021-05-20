import random
import signal
import sys
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

from Agents.DQN import DQNAgent
from Agents.Minimax import Minimax, MinimaxDilute
from Agents.RandomAgent import RandomAgent
from Connect4 import Connect4

def main():
    game = Connect4()
    dqn = DQNAgent(game,
                    model_name="model1",
                    save_location="./models/no_reward_discount/",
                    model_path=None,
                    )
    dqn2 = DQNAgent(game,
                    model_name="model2",
                    save_location="./models/no_reward_discount/",
                    model_path=None,
                    )
    dqn3 = DQNAgent(game,
                    model_name="model3",
                    save_location="./models/no_reward_discount/",
                    model_path=None,
                    )
    """
    dqn4 = DQNAgent(game,
                    model_name="model4",
                    save_location="./models/no_epsilon_5/",
                    model_path=None,
                    )
    dqn5 = DQNAgent(game,
                    model_name="model5",
                    save_location="./models/no_epsilon_5/",
                    model_path=None,
                    )
    """
    #minimax = MinimaxDilute(game, max_depth=3, dilute=0.10)
    #agents = [dqn, dqn2, dqn3, dqn4, dqn5]
    agents = [dqn, dqn2, dqn3]
    players = [None, None]
    players[0] = agents[0]
    players[1] = agents[1]
    render = False
    episodes = 1000000
    render_period = 250

    # SIGQUIT (CTRL-/) signal handler
    # this will toggle between verbose player output or not
    def verbose(signal, frame):
        for agent in agents:
            agent.verbose()
    signal.signal(signal.SIGQUIT, verbose)

    # SIGINT (CTRL-C) signal handler
    # plot results before exiting
    def finish(signal, frame):
        for agent in agents:
            agent.plot_results()
        sys.exit(1)
    signal.signal(signal.SIGINT, finish)

    for episode in range(1, episodes+1):
        print(f"Game #{episode} ", end='')
        if len(agents) > 2:
            print(f"{players[0].model_name} vs. {players[1].model_name} ", end='')

        game.reset()
        training_data = [[],[]]
        player = 0
        winner = -1
        while not game.done:
            reward = 0

            state = deepcopy(game.board)
            action = players[player].get_action(state, game.player)
            new_state = deepcopy(game.move(action))

            if render and ((episode % render_period) == 0):
                print(new_state)
                print(f"action taken: {action}")

            win_type = game.check_win()

            env_info = [state, action, new_state, reward, game.done]
            training_data[player].append(env_info)

            if win_type != 0:
                # player won
                winner = player
                players[winner].won()
                loser = (winner + 1 ) % 2
                players[loser].lost()

                for p in players:
                    p.add_data(training_data[winner], winner, win_type)

            players[player].train()

            player = ( player + 1) % 2

        ### episode finished ###

        if  winner != -1:
            print(f"Winner: {players[winner].model_name} ", end='')
        else:
            print("Tie! ", end='')
        print(f"Moves: {game.moves}");

        # handle post episode events
        # stats, autosaving, etc
        for p in players:
            p.post_episode()

        # change current players
        if len(agents) > len(players):
            # rotate agents
            np.random.shuffle(agents)
            players[0] = agents[0]
            players[1] = agents[1]
        else:
            # swap player's side
            players.reverse()

    for p in players:
        p.plot_results()

if __name__ == "__main__":
    main()
