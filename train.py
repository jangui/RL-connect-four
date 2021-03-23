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

    for episode in range(episodes+1):
        game.reset()
        agent = 0
        winner = None

        training_data = [[], []]
        curiosity_training_data = [[], []]

        while not game.done:
            reward = 0

            # preform action
            state = game.board
            action = agents[agent].get_action(state)
            while not game.check_valid_move(action):
                # get next prefered action from agent
                action = agents[agent].get_next_action(state)

            new_state = game.move(action)

            if render and render_period % episode == 0:
                print(new_state)
                print(f"action taken: {action}")

            # check if agent won
            if game.check_win():
                winner = agent

            # calculate curiousity reward
            curiosity = curiosity_reward(game, new_state, agent, agents, curiosity, curiosity_training_data)
            reward += curiosity

            # save data for training
            env_info = (state, action, new_state, reward, game.done)
            training_data[agent].append(env_info)

            # swap to next agent
            agent = (agent + 1) % 2

        # after game is finished
        # if there wasnt a tie, train models
        if winner not none:
            train_models(winner, agents, training_data)

        # train each curiosity module
        curiosity[0].train(curiosity_training_data[0])
        curiosity[1].train(curiosity_training_data[1])

def train_models(winner, agents, training_data)
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
    agents[loser].train([data[-1])


def curiosity_reward(game, new_state, agent, agents, curiosity)
    if not game.done:
        # calc curiosity reward
        rival = agents[(agent + 1) % 2]
        rival_future_action = rival.get_action(new_state)
        future_state = game.move(rival_future_action, test=True)
        reward = curiosity[agent].calc_reward(new_state, future_state)

        # add data for training
        training_data = [new_state, future_state]
        curiosity_training_data[agent].append(training_data)
    else:
        return 0

    # future curiosity reward
    if not game.is_full(future_state) and not game.check_win(future_state):
        # if game can continue, calc future curiosity reward
        future_action = agents[agent].get_action(future_state)
        future_future_state = game.move(future_action, future_state, test=True)
        future_reward = curiosity[agent].calc_reward(future_state, future_future_state)
        reward += future_reward * agents[agent].discount

    return reward

if __name__ == "__main__":
    main()
