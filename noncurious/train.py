from Agent import Agent, Curiosity
from Game import Game

non_curious = False

def main():
    agents = [Agent(), Agent()]

    if not non_curious:
        curiosity = [Curiosity(), Curiosity()]

    game = Game()

    episodes = 40000

    if non_curious:
        import random
        epsilon = 1
        epsilon_decay = 0.99975

    if episode in episodes:
        if non_curious:
            episilon *= epsilon_decay
        reward = 0
        game.reset()
        agent = 0

        while not game.done:
            state = game.board
            action = preform_action(state, agents[agent])
            new_state = game.board

            # check if agent won
            if game.check_win():
                reward = 1
                break

            # calculate curiousity reward
            if not non_curious:
                rival = (agent + 1) % 2
                future_action = agents[rival].get_action(new_state)
                future_state = game.move(future_action, test=True)
                reward += curiosity[agent].calc_reward(new_state, future_state)

            # save data for training
            training_data = (state, action, new_state, reward, game.done)

            # swap to next agent
            agent = (agent + 1) % 2

def preform_action(state, agent):
    if non_curious:
            # chose random action while epsilon is large
            if epsilon > random.random():
                action = random.randint(0, 6)
            else:
                # agent chosen action
                action = agent.get_action(state)
        else:
            # curious agents never chose random moves
            action = agent.get_action(state)

    game.move(action)

    return action




if __name__ == "__main__":
    main()
