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
                    model_name="wiz",
                    save_location="./models/wiz/",
                    model_path=None,
                    )
    minimax = MinimaxDilute(game, max_depth=3, dilute=0.10)
    agents = [dqn, minimax]
    players = agents
    render = True
    episodes = 50000
    render_period = 250

    winner = None
    move_count_hist = []

    # SIGQUIT (CTRL-/) signal handler
    # this will toggle between verbose player output or not
    def verbose(signal, frame):
        for p in players:
            p.verbose()
    signal.signal(signal.SIGQUIT, verbose)

    # SIGINT (CTRL-C) signal handler
    # plot results before exiting
    def finish(signal, frame):
        for p in players:
            p.plot_results()
        plot_avg_moves(move_count_hist)
        sys.exit(1)
    signal.signal(signal.SIGINT, finish)

    for episode in range(1, episodes+1):
        print(f"Game #{episode} ", end='')
        if len(players) > 2:
            print(f"{players[0].model_name} vs. {players[1].model_name} ", end='')

        game.reset()
        training_data = [[],[]]
        player = 0
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

        if type(winner) != type(None):
            print(f"Winner: {players[winner].model_name} ", end='')
        print(f"Moves: {game.moves}");
        move_count_hist.append(game.moves)

        # handle post episode events
        # stats, autosaving, etc
        for p in players:
            p.post_episode(episode)

        # change current players
        if len(agents) > len(players):
            # rotate agents
            agent1 = random.randint(0,len(agents)-1)
            agent2 = random.randint(0,len(agents)-1)
            while agent2 == agent1:
                agent2 = random.randint(0,len(agents)-1)
            players[0] = agents[agent1]
            players[1] = agents[agent2]
        else:
            # swap player's side
            players.reverse()

    for p in players:
        p.plot_results()

    plot_avg_moves(move_count_hist)

def plot_avg_moves(move_count_hist):
    if len(move_count_hist) < 20:
        return
    rolling_avg = [0]
    rolling_avg[0] = int(np.mean(move_count_hist[:19]))
    for i in range(19, len(move_count_hist)):
        avg = int(np.mean(move_count_hist[:i+1]))
        rolling_avg.append(avg)
    plt.plot(rolling_avg)
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
