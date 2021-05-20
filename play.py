from Agents.Minimax import MinimaxDilute
from Agents.DQN import DQNAgent
from Agents.Human import Human
from Connect4 import Connect4
import matplotlib.pyplot as plt

def main():
    game = Connect4()

    agent_model_path = "./models/no_reward_discount/model3-autosave13000"
    agent = DQNAgent(game, training=False, max_depth=3, model_path=agent_model_path)

    #minimax1 = MinimaxDilute(game, max_depth=3, dilute=0.25)
    #minimax2 = MinimaxDilute(game, max_depth=3, dilute=0.0)

    human = Human(game)

    players = [agent, human]

    render = True
    games = 2

    for i in range(games):
        print("Game:", i)

        game.reset()
        player = 0
        winner = -1
        while not game.done:
            current_player = players[player]
            action = current_player.get_action(game.board, game.player)
            game.move(action)

            if render:
                print(game.board)
                print(f"{players[player].model_name}'s move: {action}")
                print()

            if game.check_win():
                winner = player
                break
            if game.is_full():
                break

            player = (player + 1) % 2

        if render:
            if winner != -1:
                print("Winner:", players[winner].model_name)
            else:
                print("Tie!")
            print()

        players.reverse()

if __name__ == "__main__":
    main()
