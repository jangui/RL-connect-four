import random

from Agent import Agent
from Game import Game

def main():
    game = Game()
    input_dimensions = game.board.shape[0] * game.board.shape[1]
    agent = Agent((input_dimensions,), game.num_actions, model_path="./curiosity/yellow1.5")


    game.reset()
    state = game.board

    while not game.done:
        action = int(input("move: "))
        game.move(action)
        print(game)
        print("last move:", action)
        done = game.done


        # opponent moves
        if not done:
            action = agent.get_action(game.board.flatten())

            game.move(action)
            print(game)
            print("last move:", action)
            done = game.done


if __name__ == "__main__":
    main()