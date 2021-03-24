from Agent import Agent, Curiosity
from Game import Game

def main():
    agents = [Agent(), Agent()]
    curiosity = [Curiosity(), Curiosity()]

    game = Game()

    episodes = 40000
    autosave_period = 1000
    render_period = 50
    render = True

    for episode in range(1, episodes+1):
        print(f"Pure Curiosity Game #{episode}")
        game.reset()
        agent = 0

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
            game.check_win()

            if render and ((episode % render_period) == 0):
                print(new_state)
                print(f"action taken: {action}")

            # calculate curiousity reward
            c_reward = curiosity_reward(game, new_state, agent, agents, curiosity)
            reward += c_reward

            # save data for training
            env_info = [state, action, new_state, reward, game.done]
            training_data[agent].append(env_info)

            # swap to next agent
            agent = (agent + 1) % 2

        # train models
        for agent in range(2):
            for data in training_data[agent]:
                agents[agent].add_data(data)
            agents[agent].train()
            curiosity[agent].train()

        # autosave
        if episode % autosave_period == 0:
            agents[0].model.save(f"./models/purecurious/autosave/red{episode}")
            curiosity[0].model.save(f"./models/purecurious/autosave/red_curiosity{episode}")
            agents[1].model.save(f"./models/purecurious/autosave/yellow{episode}")
            curiosity[1].model.save(f"./models/purecurious/autosave/yellow_curiosity{episode}")
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
