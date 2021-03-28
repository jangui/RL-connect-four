from Agents.Curious import CuriousAgent, Curiosity
from Agents.Minimax import Minimax
from Agents.RandomAgent import RandomAgent
from Connect4 import Connect4
from copy import deepcopy
import numpy as np

def main():
    game = Connect4()
    agents = [CuriousAgent(game), Minimax(game, max_depth=1, player=-1)]
    curious_agent = 0
    rival = 1
    curiosity = Curiosity()

    episodes = 40000
    autosave_period = 1000
    render_period = 50
    render = True

    for episode in range(1, episodes+1):
        print(f"Curiosity Game #{episode}")
        game.reset()

        training_data = []
        curiosity_data = []

        agent = 0
        winner = None

        while not game.done:
            reward = 0.0
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

            if agent == rival:
                curiosity_data.append([state, new_state]) # used for rewards
                curiosity.add_data([state, new_state]) # used for training
            elif agent == curious_agent:
                env_info = [state, action, new_state, reward, game.done]
                training_data.append(env_info)

            # swap to next agent
            agent = (agent + 1) % 2

        # add curiosity reward to training data
        add_curiosity_reward(curious_agent, curiosity, curiosity_data, training_data)
        # train agent
        train_model(win_type, winner, curious_agent, agents, training_data)
        # train curiosity module
        curiosity.train()

        # autosave
        if episode % autosave_period == 0:
            agents[curious_agent].model.save(f"./models/autosave/curiousAgent{episode}")
            curiosity.model.save(f"./models/autosave/curiosity{episode}")
    return

def add_curiosity_reward(agent, curiosity, curiosity_data, training_data):
    for i in range(len(curiosity_data)):
        state = curiosity_data[i][0]
        new_state = curiosity_data[i][1]
        # calc curiosity reward
        curiosity_reward = curiosity.model.evaluate(state.reshape((1,6,7)),
                                                    new_state.reshape((1,42)),
                                                    verbose=0
                                                    )[0]
        if agent == 0:
            training_data[i][3] += curiosity_reward
        elif agent == 1:
            if i != 0:
                training_data[i-1][3] += curiosity_reward
    return

def train_model(win_type, winner, agent, agents, training_data):
    # if we won add reward to all moves leading to win
    if winner == agent:
        print("agent won!")
        if win_type == 2:
            # less reward for vertical wins
            reward = 0.001
        else:
            print("non vertical win!")
            reward = 1
        reward = 1

        for i in range(len(training_data)):
            training_data[i][3] += reward
            # add data to model's replay mem for training
            agents[agent].add_data(training_data[i])
    elif winner == (agent + 1) % 2:
        # if we lost, penalize losing move
        training_data[-1][3] = -1
        # add data to replay mem and train
        for env_info in training_data:
            agents[agent].add_data(env_info)
        # add losing move to priority_data
        agents[agent].add_priority_data(training_data[-1])
    else:
        for env_info in training_data:
            agents[agent].add_data(env_info)

    agents[agent].train()
    return

if __name__ == "__main__":
    main()
