from Agent import Agent
from Game import Game
import random

def main():
    agents = [Agent(), Agent()]
    game = Game()

    episodes = 40000
    autosave_period = 1000
    render_period = 50
    render = True

    epsilon = 1
    epsilon_decay = 0.99985

    for episode in range(episodes+1):
        print(f"Noncurious Game #{episode}")
        game.reset()
        agent = 0
        winner = None

        training_data = [[], []]

        while not game.done:
            reward = 0
            state = game.board

            # preform action
            if epsilon > random.random():
                # random action
                action = random.randint(0, 6)
                while not game.check_valid_move(action):
                    action = random.randint(0, 6)
            else: # agent chosen action
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

            # save data for training
            env_info = [state, action, new_state, reward, game.done]
            training_data[agent].append(env_info)

            # swap to next agent
            agent = (agent + 1) % 2

            # decay epsilon
            if agent == 0:
                epsilon *= epsilon_decay

        # after game is finished
        # if there wasnt a tie, train models
        if type(winner) != type(None):
            train_models(winner, agents, training_data)

        # autosave
        if episode % autosave_period == 0:
            agents[0].model.save(f"./models/autosave/red{episode}")
            agents[1].model.save(f"./models/autosave/yellow{episode}")
    return

def train_models(winner, agents, training_data):
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
    data[-1][reward_ind] = -1
    # add data to replay mem and train
    agents[loser].add_data(data[-1])
    agents[loser].add_priority_data(data[-1])
    agents[loser].train()


if __name__ == "__main__":
    main()
