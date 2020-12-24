import random

from Agent import Agent
from Game import Game

def main():
    game = Game()
    agent = Agent(game.board.shape, game.num_actions, model_path=None)
    agent2 = Agent(game.board.shape, game.num_actions, model_path=None)

    episodes = 100000
    render_period = 1000
    render = False

    for episode in range(episodes):
        reward = 0
        game.reset()
        state = game.board

        while not game.done:
            if agent.epsilon > random.random():
                #preform random move
                action = random.randint(0, game.num_actions-1)
            else:
                #preform action based off network prediction
                #as episilon decays this will be the usual option
                action = agent.get_action(state)

            reward = game.move(action)
            done = game.done
            new_state = game.board

            # train
            env_info = (state, action, new_state, reward, done)
            agent.train(env_info)

            # render
            if render and (episode % render_period == 0):
                print(game)

            # opponent moves
            if not done:
                if agent2.epsilon > random.random():
                    #preform random move
                    action = random.randint(0, game.num_actions-1)
                else:
                    #preform action based off network prediction
                    #as episilon decays this will be the usual option
                    action = agent.get_action(state)

                reward = game.move(action)
                done = game.done
                new_state = game.board

                # train
                env_info = (state, action, new_state, reward, done)
                agent2.train(env_info)

                # render
                if render and (episode % render_period == 0):
                    print(game)

            state = new_state

            #decay epsilon
            if agent.epsilon > agent.min_epsilon:
                agent.epsilon *= agent.epsilon_decay
                agent.epsilon = max(agent.epsilon, agent.min_epsilon)

            if agent2.epsilon > agent2.min_epsilon:
                agent2.epsilon *= agent2.epsilon_decay
                agent2.epsilon = max(agent2.epsilon, agent2.min_epsilon)

        if episode % 1000 == 0:
            agent.model.save(f"./q/red{episode}autosave")
            agent2.model.save(f"./q/yellow{episode}autosave")


if __name__ == "__main__":
    main()
