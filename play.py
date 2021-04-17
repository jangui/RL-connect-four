from Agents.Minimax import MinimaxDilute
from Agents.DQN import DQNAgent
from Connect4 import Connect4
import matplotlib.pyplot as plt

def main():
    game = Connect4()
    #agent = DQNAgent(game, training=False, model_path=model_path)
    minimax1 = MinimaxDilute(game, max_depth=2, dilute=0.1)
    minimax2 = MinimaxDilute(game, max_depth=3, dilute=0.3)

    players = [[minimax1, 0, 0], [minimax2, 0, 0]]

    render = False
    games = 1000
    for i in range(games):
        print("Game:", i)

        game.reset()
        player = 0
        while not game.done:
            current_player = players[player][0]
            action = current_player.get_action(game.board, game.player)
            game.move(action)

            if render:
                print(game.board)

            if game.check_win():
                # add win
                players[player][player+1] += 1
                break
            if game.is_full():
                break

            player = (player + 1) % 2

        if render:
            print()

        players.reverse()

    print(players)
    return
    x = ['first move', 'second move']
    y = [players[0][1], players[0][2]]
    plt.bar(x, y, label='minimax')
    plt.show()
    y = [players[q][1], players[1][2]]
    plt.bar(x, y, label='minimax2')
    plt.show()

if __name__ == "__main__":
    main()
