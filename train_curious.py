import random
import numpy as np
from tensorflow import convert_to_tensor, reshape

from Agent import Agent
from Curiosity import Curiosity
from Game import Game

def main():
    game = Game()

    curiosity_dim = game.board.shape[0] * game.board.shape[1]

    agent = Agent(game.board.shape, game.num_actions, model_path=None)
    curiosity = Curiosity((curiosity_dim,), curiosity_dim, model_path=None)
    agent2 = Agent(game.board.shape, game.num_actions, model_path=None)
    curiosity2 = Curiosity((curiosity_dim,), curiosity_dim, model_path=None)

    episodes = 100000
    render_period = 200
    render = True

    for episode in range(episodes):
        if render and (episode % render_period == 0):
            print(f"Curiosity Game #{episode}")
        reward = 0
        game.reset()
        state = game.board

        while not game.done:
            # first agent selects move
            action = agent.get_action(state)

            # agent makes move and collects new info about environment
            reward, win  = game.move(action)
            done = game.done
            new_state = game.board


            # calculate the curiosity reward
            # only if game has not been won or lost yet
            if not done:
                # find opponents response
                future_move = agent2.get_action(new_state)
                future_state, future_win = game.test_move(future_move)

                # calculate curiosity reward
                curiosity_reward = curiosity.calc_reward(new_state, future_state)

                # if opponents next move is winning penalize beacuse we didn't stop the win
                if future_win:
                    reward = -999

                reward += curiosity_reward[0]

                # train curiosity module
                curiosity.train(new_state, future_state)

            # train agent
            env_info = (state, action, new_state, reward, done)
            agent.train(env_info)

            # render
            if render and (episode % render_period == 0):
                print(game, "\n")
                if done and win:
                    print("red wins")
                elif done:
                    print("invalid move by red:", action)

            # opponent
            if not done:
                # opponent gets action for new state
                state = new_state
                action = agent2.get_action(state)

                # opponent makes move and gets new env info
                reward, win = game.move(action)
                done = game.done
                new_state = game.board

                # get next state of board to calculate curiousity based reward
                if not done:
                    future_move = agent.get_action(new_state)
                    future_state, future_win = game.test_move(future_move)

                    # calculate curiosity reward
                    curiosity_reward = curiosity2.calc_reward(new_state, future_state)

                    # if opponents next move is winning penalize beacuse we didn't stop the win
                    if future_win:
                        reward = -999

                    reward += curiosity_reward[0]

                    # train curiosity module
                    curiosity2.train(new_state, future_state)

                # train opponent agent
                env_info = (state, action, new_state, reward, done)
                agent2.train(env_info)

                # render
                if render and (episode % render_period == 0):
                    print(game, "\n")
                    if done and win:
                        print("yellow wins")
                    elif done:
                        print("invalid move by yellow:", action)


            state = new_state

        if episode % 1000 == 0:
            agent.model.save(f"./curiosity/red{episode}autosave")
            agent2.model.save(f"./curiosity/yellow{episode}autosave")
            curiosity.model.save(f"./curiosity/red_curiosity{episode}autosave")
            curiosity.model.save(f"./curiosity/yellow_curiosity{episode}autosave")


if __name__ == "__main__":
    main()
