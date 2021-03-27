from Agent import Agent, Curiosity
from .. import Connect4
import numpy as np
import sys

def main():
    agents = [Agent(), Agent()]
    curiosity = [Curiosity(), Curiosity()]
    game = Connect4()

    try:
        model_save_loc = sys.argv[1]
    except:
        model_save_loc = "./models/autosave/"

    try:
        env_reward = int(sys.argv[2])
    except:
        env_reward = 1

    episodes = 40000
    autosave_period = 1000
    render_period = 50
    render = True

    for episode in range(1, episodes+1):
        print(f"Curiosity Game #{episode}")
        game.reset()
        agent = 0
        winner = None

        training_data = [[], []]

        while not game.done:
            reward = 0.0

            # preform action
            state = game.board
            curiosity_values = get_curiosity_values(agent, agents, curiosity, game)
            action = agents[agent].get_action(state, curiosity_values)
            while not game.check_valid_move(action):
                # get next prefered action from agent
                action = agents[agent].get_next_action()

            new_state = game.move(action)

            if render and ((episode % render_period) == 0):
                print(new_state)
                print(f"action taken: {action}")

            # check if agent won
            if game.check_win():
                winner = agent

            # save data for training
            env_info = [state, action, new_state, reward, game.done]
            training_data[agent].append(env_info)

            # swap to next agent
            agent = (agent + 1) % 2

        # add curiosity reward to training data
        add_curiosity_reward(curiosity, training_data)

        # after game is finished train agents
        train_models(winner, agents, training_data)

        # train each curiosity module
        curiosity[0].train()
        curiosity[1].train()

        # autosave
        if episode % autosave_period == 0:
            agents[0].model.save(model_save_loc + f"red{episode}")
            curiosity[0].model.save(model_save_loc + f"red_curiosity{episode}")
            agents[1].model.save(model_save_loc + f"yellow{episode}")
            curiosity[1].model.save(model_save_loc + f"yellow_curiosity{episode}")
    return

def add_curiosity_reward(curiosity, training_data):
    agent2_won = True
    # if 1st agent won, they played an extra move
    if len(training_data[0]) != len(training_data[1]):
        agent2_won = False

    # for each agent
    for i in range(len(curiosity)):
        rival = (i + 1 ) % 2

        # for each element in each each agent's training data
        # note we iterate over 2nd agent's training_data length
        # 2nd agent's training_data is either same length or one less
        # bcs winning move doesnt need curiosity reward, this handles
        # if the 1st agent won
        print(f"agent {i}")
        for j in range(len(training_data[1])):


            # if first agent, rival response is the same index
            if i == 0:
                data_ind = j

            # if 2nd agent, rival's response is index + 1
            elif i == 1:
                data_ind = j+1

                # if 2nd agent won, skip last training data point
                if agent2_won and j == (len(training_data[1]) - 1):
                    break

            # use rivals state and new state for curiosity rewards and traing
            state = training_data[rival][data_ind][0]
            new_state = training_data[rival][data_ind][2]

            # add data for training (training done later)
            curiosity[i].add_data([state, new_state])

            # calc curiosity reward
            curiosity_reward = curiosity[i].model.evaluate(state.reshape((1,6,7)),
                                                            new_state.reshape((1,42)),
                                                            verbose=0
                                                            )[0]


            print("curiosity reward: ", end="")
            print(curiosity_reward)
            print()

            # add curiosity reward to each agent
            training_data[i][j][3] += curiosity_reward


def get_curiosity_values(agent, agents, curiosity, game):
    # remember to train as well
    rival = agents[(agent + 1) % 2]
    curiosity_values = np.zeros((7))
    # get curiosity value for each possible action
    for i in range(7):
        state = game.move(i, test=True)
        if game.is_full(state) or game.check_win(state, test=True):
            curiosity_values[i] = 0
            continue
        rival_action = rival.get_action(state)
        new_state = game.move(rival_action, test=True)
        curiosity[agent].add_data([state, new_state])
        curiosity_value = curiosity[agent].calc_reward(state, new_state)
        curiosity_values[i] = curiosity_value
    return curiosity_values

def train_models(winner, agents, training_data):
    # if no winner, train solely based on curiosity reward
    if winner == None:
        for agent in range(2):
            for data in training_data[agent]:
                agents[agent].add_data(data)
            agents[agent].train()
        return

    # train winner
    data = training_data[winner]

    # add reward to every move that lead to win
    reward_ind = 3
    for i in range(len(data)):
        data[i][reward_ind] += 1
        # add data to model's replay mem for training
        agents[winner].add_data(data[i])
    agents[winner].train()

    # train loser
    loser = (winner + 1) % 2
    data = training_data[loser]
    # penalize losing move
    data[-1][reward_ind] -= 1
    # add data to replay mem and train
    agents[loser].add_data(data[-1])
    agents[loser].add_priority_data(data[-1])
    agents[loser].train()
    return

if __name__ == "__main__":
    main()
