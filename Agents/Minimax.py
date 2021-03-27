import math

class Minimax:
    def __init__(self, game, player=1):
        self.game = game
        self.player = player

    def calc_utility(self, action, state):
        utility = 0
        board = self.game.move(action, test=True)

        vertical_utility = self.calc_vert_utility(board)
        if math.isinf(vertical_utility):
            return math.inf
        utility += vertical_utility

        horizontal_utility = self.calc_hor_utility(board)
        if math.isinf(horizontal_utility):
            return math.inf
        utility += horizontal_utility

    def calc_vert_utility(board):
        connected = 0
        # count consecutive vertical pieces
        # start count from bottom to top
        for c in range(-1, 6, -1):
            for r in range(-1, 5, -1):

                # if a zero, there no pieces above
                # thus no need to keep counting
                if board[r][c] == 0:
                    break

                # if we reach this row but it doesn't have our piece
                # then we can't connect 4 vertically on this column
                if r == 3:
                    if board[r][c] != self.player:
                        break

                # count connected pieces
                if board[r][c] == self.player:
                    connected += 1
                else:
                    connected = 0

        if connected >= 4:
            return math.inf
        return connected

    def calc_hor_utility(board):
        # there are 4 sections we can connect 4 in each row
        # utility = the number of pieces in one each of those sections
        # if there is a rival piece in one of those sections,
        # then that section has 0 utilty
        utility = 0
        rival = -self.player
        for r in range(6):
            for c in range(4):
                section_utility = 0
                for i in range(4):
                    if board[r][i] == player:
                        section_utility += 1
                    elif board[r][i] == rival:
                        section_utility == 0
                        break
                if section_utility == 4:
                    return math.inf
                utility += section_utility
        return utility



        # count consecutive diagonal pieces (negative slope)
        c = 0
        for r in range(4):
            connected = 0
            while r < 6 and c < 7:
                if board[r][c] == self.player:
                    connected += 1
                else:
                    connected = 0

                if connected >= 4:
                    return math.inf
                r += 1
                c += 1
            utility += connected

        r = 0
        for c in range(4):
            connected = 0
            while r < 6 and c < 7:
                if board[r][c] == self.player:
                    connected += 1
                else:
                    connected = 0

                if connected >= 4:
                    return math.inf
                r += 1
                c += 1
            utility += connected

        # count consecutive diagonal pieces (positive slope)
        for r in range(3, 6):
            for c in range(4):
                if board[r][c]

