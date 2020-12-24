import random
from tqdm import tqdm
import numpy as np

from Settings import Settings
from Agent import Agent
from Game import Game

def main():
    game = Game() # connect 4

    num_actions = 7

    episode_rewards = []
    rewards_rolling_avg = []
    aggr_stats_lst = []

    model_path = None
    agent = Agent(input_dimensions=game.board.shape, output_dimensions=num_actions)
    agent2 = Agent(input_dimensions=game.board.shape, output_dimensions=num_actions)

    episodes = 10000

    for episode in tqdm(range(1, episodes+1), ascii=True, unit='episode'):
        episode_reward = 0
        done = False
        game.reset()
        state = game.board

        #each loop is a chess game
        while not done:
            if agent.epsilon > random.random():
                #preform random action
                #while epsilon is high more random actions will be taken
                action = random.randint(0, num_actions-1)
            else:
                #preform action based off network prediction
                #as episilon decays this will be the usual option
                action = agent.get_action(state)

            #take action and get data back from env
            reward = game.move(action)
            new_state = game.board
            done = game.done

            #train agent
            env_info = (state, action, new_state, reward, done)
            agent.train(env_info)

            #render
            if s.render and (episode % s.render_period == 0):
                print(game)

            state = new_state
            episode_reward += reward

        #decay epsilon
        if agent.epsilon > agent.min_epsilon:
            agent.epsilon *= agent.epsilon_decay
            agent.epsilon = max(agent.epsilon, agent.min_epsilon)


    return

if __name__ == "__main__":
    main()
