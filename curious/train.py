from Agent import Agent, Curiosity
from Game import Game

def main():
    agents = [Agent(), Agent()]
    curiosity = [Curiosity(), Curiosity()]
    training_data = [[], []]
    curiosity_training_data = [[], []]

    game = Game()

    episodes = 40000
    render_period = 100
    render = True

    for episode in range(1, episodes+1):
        game.reset()
        agent = 0
        winner = None

        training_data = [[], []]
        curiosity_training_data = [[], []]

        if render and ((episode % render_period) == 0):
            print(f"Game #{episode}")

        while not game.done:
            reward = 0

            # preform action
            state = game.board
            action = agents[agent].get_action(state)
            while not game.check_valid_move(action):
                # get next prefered action from agent
                action = agents[agent].get_next_action()
                if type(action) == type(None):
                    print("error: no valid action but game ongoing")
                    return

            new_state = game.move(action)

            if render and ((episode % render_period) == 0):
                print(new_state)
                print(f"action taken: {action}")

            # check if agent won
            if game.check_win():
                winner = agent

            # calculate curiousity reward
            #TODO fix
            """
            c_reward = curiosity_reward(game, new_state, agent, agents, curiosity, curiosity_training_data)
            reward += c_reward
            """

            # save data for training
            env_info = [state.reshape((1,6,7)), action, new_state.reshape((1,6,7)), reward, game.done]
            training_data[agent].append(env_info)

            # swap to next agent
            agent = (agent + 1) % 2

        # TODO delete
        if game.done and type(winner) == type(None):
            print("tie!")
            print("final board:")
            print(game.board)

        # after game is finished
        # if there wasnt a tie, train models
        if type(winner) != type(None):
            train_models(winner, agents, training_data)

        #TODO fix
        """
        # train each curiosity module
        curiosity[0].train(curiosity_training_data[0])
        curiosity[1].train(curiosity_training_data[1])
        """

def train_models(winner, agents, training_data):
    # train winner
    data = training_data[winner]

    # add reward to every move that lead to win
    reward_ind = 3
    for i in range(len(data)):
        data[i][reward_ind] += 1
    agents[winner].train(training_data[winner])

    # train loser
    loser = (winner + 1) % 2
    data = training_data[loser]
    # penalize losing move
    data[-1][reward_ind] = -1
    agents[loser].train([data[-1]])


def curiosity_reward(game, new_state, agent, agents, curiosity, curiosity_training_data):
    # calc curiosity reward
    if not game.done:
        rival = agents[(agent + 1) % 2]

        # check if rival future action is valid
        rival_future_action = rival.get_action(new_state)
        while not game.check_valid_move(rival_future_action):
            # get next prefered action from rival
            rival_future_action = rival.get_next_action()
            if type(rival_future_action) == type(None):
                print("error: no valid rival action but game ongoing")

        future_state = game.move(rival_future_action, test=True)
        reward = curiosity[agent].calc_reward(new_state, future_state)

        # add data for training
        training_data = [new_state, future_state]
        curiosity_training_data[agent].append(training_data)

    else:
        return 0

    # if game can continue, calc future curiosity reward
    if not game.is_full(future_state) and not game.check_win(future_state):

        # check if future action is valid
        future_action = agents[agent].get_action(future_state)
        while not game.check_valid_move(future_action):
            # get next prefered action from agent
            future_action = agents[agent].get_next_action()
            if type(future_action) == type(None):
                print("error: no valid future action but game ongoing")

        future_future_state = game.move(future_action, future_state, test=True)
        future_reward = curiosity[agent].calc_reward(future_state, future_future_state)
        reward += future_reward * agents[agent].discount

    return reward

if __name__ == "__main__":
    main()
