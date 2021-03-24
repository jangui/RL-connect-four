from Agent import Agent, Curiosity
from Game import Game
import sys

def main():
    try:
        env_reward = float(sys.argv[1])
    except:
        env_reward = 1.0

    try:
        save_loc = "./models/" + sys.arg[2]
    except:
        save_loc = "./models/autosave/"

    agents = [Agent(), Agent()]
    curiosity = [Curiosity(), Curiosity()]

    game = Game()

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
            action = agents[agent].get_action(state)
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

            # calculate curiousity reward
            c_reward = curiosity_reward(game, new_state, agent, agents, curiosity)
            reward += c_reward

            # save data for training
            env_info = [state, action, new_state, reward, game.done]
            training_data[agent].append(env_info)

            # swap to next agent
            agent = (agent + 1) % 2

        # after game is finished train agents
        train_models(winner, agents, training_data, env_reward)

        # train each curiosity module
        # training data was added when curiosity reward was calculated
        curiosity[0].train()
        curiosity[1].train()

        # autosave
        if episode % autosave_period == 0:
            agents[0].model.save(save_loc + f"red{episode}")
            curiosity[0].model.save(save_loc + f"red_curiosity{episode}")
            agents[1].model.save(save_loc + f"yellow{episode}")
            curiosity[1].model.save(save_loc + f"yellow_curiosity{episode}")
    return

def train_models(winner, agents, training_data, env_reward):
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
        data[i][reward_ind] += env_reward
        # add data to model's replay mem for training
        agents[winner].add_data(data[i])
    agents[winner].train()

    # train loser
    loser = (winner + 1) % 2
    data = training_data[loser]
    # penalize losing move
    data[-1][reward_ind] -= env_reward
    # add data to replay mem and train
    agents[loser].add_data(data[-1])
    agents[loser].train()
    return

def curiosity_reward(game, new_state, agent, agents, curiosity):
    # calc curiosity reward
    if not game.done:
        rival = agents[(agent + 1) % 2]

        # check if rival future action is valid
        rival_future_action = rival.get_action(new_state)
        while not game.check_valid_move(rival_future_action):
            # get next prefered action from rival
            rival_future_action = rival.get_next_action()

        future_state = game.move(rival_future_action, test=True)
        reward = curiosity[agent].calc_reward(new_state, future_state)

        # add data for training
        training_data = [new_state, future_state]
        curiosity[agent].add_data(training_data)

    else:
        return 0

    # if game can continue, calc future curiosity reward
    if not game.is_full(future_state) and not game.check_win(future_state, test=True):

        # check if future action is valid
        future_action = agents[agent].get_action(future_state)
        while not game.check_valid_move(future_action):
            # get next prefered action from agent
            future_action = agents[agent].get_next_action()

        future_future_state = game.move(future_action, future_state, test=True)
        future_reward = curiosity[agent].calc_reward(future_state, future_future_state)
        reward += future_reward * agents[agent].discount

    return reward

if __name__ == "__main__":
    main()
