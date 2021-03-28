from Agents.Noncurious import Noncurious
from Agents.Minimax import Minimax
from Agents.RandomAgent import RandomAgent
from copy import deepcopy
from Connect4 import Connect4
import random

def main():
    game = Connect4()
    model_path = None
    agents = [Noncurious(game, model_path=model_path), Minimax(game, max_depth=1, player=-1)]
    noncurious = 0

    episodes = 32000
    autosave_period = 1000
    render_period = 50
    render = True

    for episode in range(1, episodes+1):
        print(f"Noncurious Game #{episode}")
        game.reset()
        winner = -1

        training_data = []

        agent = 0
        winner = None
        while not game.done:
            reward = 0
            state = deepcopy(game.board)

            action = agents[agent].get_action(state)

            new_state = deepcopy(game.move(action))

            if render and ((episode % render_period) == 0):
                print(new_state)
                print(f"action taken: {action}")

            # check if agent won
            win_type = game.check_win()
            if win_type != 0:
                winner = agent

            # save data for training
            if agent == noncurious:
                env_info = [state, action, new_state, reward, game.done]
                training_data.append(env_info)

            # swap to next agent
            agent = (agent + 1) % 2

        # after game is finished
        # if there wasnt a tie, train model
        if win_type != 0:
            train_model(win_type, winner, noncurious, agents, training_data)

        # autosave
        if episode % autosave_period == 0:
            agents[noncurious].model.save(f"./models/autosave/noncurious{episode}")
    return

def train_model(win_type, winner, agent, agents, training_data):
    # if we won add reward to all moves leading to win
    if winner == agent:
        print("agent won!")
        reward = 1
        if win_type == 2:
            # less reward for vertical wins
            reward = 0.001
        else:
            print("non vertical win!")
            reward = 1

        for i in range(len(training_data)):
            training_data[i][3] += reward
            # add data to model's replay mem for training
            agents[agent].add_data(training_data[i])
        agents[agent].train()
        return

    # if we lost, penalize losing move
    training_data[-1][3] = -1
    # add data to replay mem and train
    agents[agent].add_data(training_data[-1])
    agents[agent].add_priority_data(training_data[-1])
    agents[agent].train()

if __name__ == "__main__":
    main()
