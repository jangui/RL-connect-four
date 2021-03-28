import math

class Minimax:
    def __init__(self, game, player=1):
        self.game = game
        self.player = player

    def calc_utility(self, board):
        utility = 0

        vertical_utility = self.calc_vert_utility(board)
        print("vert:",vertical_utility)
        if math.isinf(vertical_utility):
            return math.inf
        utility += vertical_utility

        horizontal_utility = self.calc_hor_utility(board)
        print("hor:",horizontal_utility)
        if math.isinf(horizontal_utility):
            return math.inf
        utility += horizontal_utility

        diagonal_utility = self.calc_diag_utility(board)
        print("diag:",diagonal_utility)
        if math.isinf(diagonal_utility):
            return math.inf
        return utility + diagonal_utility

    def calc_vert_utility(self, board):
        utility = 0
        # count consecutive vertical pieces
        # start count from bottom to top
        for c in range(6, -1, -1):
            connected = 0
            for r in range(5, -1, -1):

                # if a zero, there no pieces above
                # thus no need to keep counting
                if board[r][c] == 0:
                    break

                # if we reach this row but it doesn't have our piece
                # then we can't connect 4 vertically on this column
                if r == 3:
                    if board[r][c] != self.player:
                        connected = 0
                        break

                # count connected pieces
                if board[r][c] == self.player:
                    connected += 1
                else:
                    connected = 0

            if connected >= 4:
                return math.inf
            utility += connected
        return utility

    def calc_hor_utility(self, board):
        # there are 4 sections we can connect 4 in each row
        # utility = the number of pieces in one each of those sections
        # if there is a rival piece in one of those sections,
        # then that section has 0 utilty
        utility = 0
        rival = -self.player
        for r in range(6):
            section_utilities = [0,0,0,0]
            for c in range(7):
                piece = board[r][c]

                if piece == 0:
                    continue

                if c == 0:
                    sec_start_range = 0
                    sec_end_range = 1
                elif c == 1:
                    sec_start_range = 0
                    sec_end_range = 2
                elif c == 2:
                    sec_start_range = 0
                    sec_end_range = 3
                elif c == 3:
                    sec_start_range = 0
                    sec_end_range = 4
                elif c == 4:
                    sec_start_range = 1
                    sec_end_range = 4
                elif c == 5:
                    sec_start_range = 2
                    sec_end_range = 4
                elif c == 6:
                    sec_start_range = 3
                    sec_end_range = 4

                for i in range(sec_start_range,sec_end_range):
                    if section_utilities[i] == -1:
                        continue
                    if piece == rival:
                        section_utilities[i] = -1
                    elif piece == self.player:
                        section_utilities[i] += 1

            for section_util in section_utilities:
                if section_util == 4:
                    return math.inf
                if section_util == -1:
                    continue
                utility += section_util
        return utility

    def calc_diag_utility(self, board):
        utility = 0
        rival = -self.player
        # calc connections diagonally bottom left to top right
        for r in range(3, 6):
            c = 0
            connected = 0
            while r > -1 and c < 7:
                if board[r][c] == 0:
                    break

                # if rival on row index 3 we can't
                # connect 4 along this diagonal
                if r <= 3 and board[r][c] == rival:
                    connected = 0
                    break

                if board[r][c] == self.player:
                    connected += 1
                else:
                    connected = 0
                r -= 1
                c += 1

            if connected >= 4:
                return math.inf
            utility += connected

        for c in range(1, 4):
            r = 5
            connected = 0
            while r > -1 and c < 7:
                if board[r][c] == 0:
                    break

                # if rival on column index 3 we can't
                # connect 4 along this diagonal
                if c >= 3 and board[r][c] == rival:
                    connected = 0
                    break

                if board[r][c] == self.player:
                    connected += 1
                else:
                    connected = 0
                r -= 1
                c += 1

            if connected >= 4:
                return math.inf
            utility += connected

        # calc connections diagonally bottom right to top left
        for c in range(3, 6):
            r = 5
            connected = 0
            while r > -1 and c > -1:
                if board[r][c] == 0:
                    break

                # if rival on column index 3 we can't
                # connect 4 along this diagonal
                if c <= 3 and board[r][c] == rival:
                    connected = 0
                    break

                if board[r][c] == self.player:
                    connected += 1
                else:
                    connected = 0
                r -= 1
                c -= 1
            utility += connected

        for r in range(3, 6):
            c = 6
            connected = 0
            while r > -1 and c > -1:
                if board[r][c] == 0:
                    break

                # if rival on row index 3 we can't
                # connect 4 along this diagonal
                if r <= 3 and board[r][c] == rival:
                    connected = 0
                    break

                if board[r][c] == self.player:
                    connected += 1
                else:
                    connected = 0
                r -= 1
                c -= 1
            utility += connected
        return utility


