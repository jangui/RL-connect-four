from Agents.DQN import DQNAgent
from Agents.Minimax import Minimax, MinimaxDilute
from Agents.RandomAgent import RandomAgent
from copy import deepcopy
from Connect4 import Connect4
import random

def main():
    game = Connect4()
    dqn = DQNAgent(game,
                    model_name="athena",
                    save_location="./models/athena/",
                    model_path=None,
                    )
    minimax = MinimaxDilute(game, max_depth=2, dilute=0.25)

    agents = [dqn, minimax]
    players = agents

    episodes = 50000
    render_period = 125
    render = True

    winner = None
    for episode in range(1, episodes+1):
        print(f"Game #{episode} ", end='')

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
                handle_win(players, winner)
                add_training_data(training_data[winner], players, win_type)

            players[player].train()

            player = ( player + 1) % 2

        ### episode finished ###

        if type(winner) != type(None):
            print(f"Winner: {players[winner]} ", end='')
        print(f"Moves: {game.moves}");

        # handle post episode events
        # stats, autosaving, etc
        for player in players:
            player.post_episode(episode)

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

def handle_win(players, winner):
    players[winner].won()
    loser = (winner + 1 ) % 2
    players[loser].lost()

def add_training_data(data, players, win_type):
    reward  = 1

    # less reward on vertical wins
    if win_type == 2:
        reward = 0.001

    for env_info in data:
        reward_index = 3
        env_info[reward_index] += reward
        players[0].add_data(env_info)
        players[1].add_data(env_info)

if __name__ == "__main__":
    main()
