import numpy as np
import copy

class Connect4:
    def __init__(self):
        self.num_actions = 7
        self.reset()

    def reset(self):
        self.done = False
        self.board = np.zeros((6,7))
        self.turn = "red"

    def check_valid_move(self, column):
        # invalid column
        if column > 6 or column < 0:
            return False

        # column full
        if self.board[0][column] != 0:
            return False

        return True

    # check if board is full
    def is_full(self, board=None):
        if type(board) == type(None):
            board = self.board

        for c in range(7):
            if board[0][c] == 0:
                return False
        return True

    def move(self, column, board=None, test=False):
        if type(board) == type(None):
            board = self.board

        if self.turn == "red":
            action = 1
            if not test:
                self.turn = "yellow"
        else:
            action = -1
            if not test:
                self.turn = "red"

        # check where to place piece in column (because gravity effects connect 4)
        for i in range(6):
            pos = board[5-i][column]
            # make move in 'highest' empty spot
            if pos == 0:
                if test: # don't modify actual game state
                    board_copy = copy.deepcopy(board)
                    board_copy[5-i][column] = action
                    return board_copy
                board[5-i][column] = action
                self.board = board
                break

        # end game if board filled
        if self.is_full() and not test:
            self.done = True
        return self.board

    def check_win(self, board=None, test=False):
        if type(board) == type(None):
            board = self.board

        # check if last piece caused a win
        if self.check_win_row(board):
            if not test:
                self.done = True
            return True
        if self.check_win_column(board):
            if not test:
                self.done = True
            return True
        if self.check_win_diagonal(board):
            if not test:
                self.done = True
            return True
        return False

    def check_win_row(self, board):
        for r in range(6):
            for c in range(4):
                if board[r][c] == 0:
                    continue
                if board[r][c] == board[r][c+1] == board[r][c+2] == board[r][c+3]:
                    return True
        return False

    def check_win_column(self, board):
        for c in range(7):
            for r in range(3):
                if board[r][c] == 0:
                    continue
                if board[r][c] == board[r+1][c] == board[r+2][c] == board[r+3][c]:
                    return True
        return False

    def check_win_diagonal(self, board):
        # check diagonal top left to bottom right
        for r in range(3):
            for c in range(4):
                if board[r][c] == 0:
                    continue
                if board[r][c] == board[r+1][c+1] == board[r+2][c+2] == board[r+3][c+3]:
                    return True

        # check diagonal bottom left to top right
        for r in range(5, 2, -1):
            for c in range(3):
                if board[r][c] == 0:
                    continue
                if board[r][c] == board[r-1][c+1] == board[r-2][c+2] == board[r-3][c+3]:
                    return True
        return False

    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(self.board)

