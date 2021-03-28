from Agents.Minimax import Minimax
from Agents.Noncurious import Noncurious
from Connect4 import Connect4

def main():
    game = Connect4()
    model_path = ""
    agent = Noncurious(game, training=False, model_path=model_path)

    while not game.done:
        action = agent.get_action(game.board)
        game.move(action)

        if game.is_full() or game.check_win():
            break

        print()
        print(game.board)
        print("rival action:", action)

        action = int(input("Action: "))
        while not game.check_valid_move(action):
            action = int(input("Action: "))
        game.move(action)

        game.check_win()

    print(game.board)

    return

if __name__ == "__main__":
    main()
