from Agents.Minimax import MinimaxDilute
from Agents.DQN import DQNAgent
from Agents.Human import Human
from Connect4 import Connect4
import matplotlib.pyplot as plt

def main():
    game = Connect4()
    #random_agent = RandomAgent(game)
    minimax1 = MinimaxDilute(game, max_depth=3, dilute=0.25)
    minimax2 = MinimaxDilute(game, max_depth=3, dilute=0.0)
    #human = Human(game)


    agent_model_path = "./models/no_reward_discount/model2-autosave13000"
    agent = DQNAgent(game, training=False, max_depth=3, model_path=agent_model_path)

    players = [[agent, 0, 0], [minimax2, 0, 0]]

    render = False
    games = 1

    for i in range(games):

        game.reset()
        player = 0
        winner = -1
        while not game.done:
            current_player = players[player][0]
            action = current_player.get_action(game.board, game.player)
            game.move(action)

            if render:
                print(game.board)
                print(f"{players[player][0].model_name}'s move: {action}")
                print()

            if game.check_win():
                # add win
                players[player][player+1] += 1
                winner = player
                break
            if game.is_full():
                break

            player = (player + 1) % 2

        if render:
            if winner != -1:
                print("Winner:", players[winner][0].model_name)
            print()

        players.reverse()

    results(players[0], games)
    results(players[1], games)

def results(player, total_games):
    first_to_move_win_percent = round((player[1] / total_games) * 100, 2)
    second_to_move_win_percent = round((player[2] / total_games) * 100, 2)
    total_win_percent = first_to_move_win_percent + second_to_move_win_percent
    print(player[0].model_name)
    print(f"\twins - {total_win_percent}%")
    print(f"\tfist to move wins - {first_to_move_win_percent}%")
    print(f"\tsecond to move - {second_to_move_win_percent}%\n")
if __name__ == "__main__":
    main()
