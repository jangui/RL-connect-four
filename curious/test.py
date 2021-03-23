from Agent import Agent
from Game import Game

game = Game()
agent = Agent()
print("#"*100)
print()
print()

agent.get_action(game.board)

print("#"*100)
print()
print()

agent.test_train()
