import numpy as np
import copy

class Game:
    def __init__(self):
        self.num_actions = 7
        self.reset()

    def reset(self):
        self.done = False
        self.board = np.zeros((6,7))
        self.turn = "red"
        self.moves = 0

    def move(self, column):
        win = False
        if column > 6 or column < 0:
            self.done = True
            return -999, win # invalid move

        turn = self.turn
        if self.turn == "red":
            action = 1
            self.turn = "yellow"
        else:
            action = -1
            self.turn = "red"

        # check where to place piece in column
        for i in range(6):
            pos = self.board[5-i][column]
            # make move if empty spot
            if pos == 0:
                self.board[5-i][column] = action
                # check if move was a winning move
                reward = self.check_win(5-i, column, action, turn)
                if reward == 999:
                    win = True
                self.moves += 1
                if self.moves == 42: # end game if board filled
                    self.done = True
                return reward, win


        self.done = True
        return -999, win # invalid move

    # make move but don't actually modify board or game state
    # return copy of board with test move done
    def test_move(self, column):
        win = False
        board = copy.deepcopy(self.board)
        if column > 6 or column < 0:
            return board, win # invalid move, return un changed board

        turn = self.turn
        if self.turn == "red":
            action = 1
        else:
            action = -1

        # check where to place piece in column
        for i in range(6):
            pos = board[5-i][column]
            # make move if empty spot
            if pos == 0:
                board[5-i][column] = action
                reward = self.test_win(5-i, column, action, turn)
                if reward == 999:
                    win = True
                return board, win

        return board, win # invalid move, return un changed board

    # same as check win without modifying any game state
    def test_win(self, row, column, action, turn):
        # check if last piece caused a win
        if self.check_win_row(row, column, action):
            return 999
        if self.check_win_column(row, column, action):
            return 999
        if self.check_win_diagonal(row, column, action):
            return 999
        return 0

    def check_win(self, row, column, action, turn):
        # check if last piece caused a win
        if self.check_win_row(row, column, action):
            self.done = True
            return 999
        if self.check_win_column(row, column, action):
            self.done = True
            return 999
        if self.check_win_diagonal(row, column, action):
            self.done = True
            return 999
        return 0

    def check_win_row(self, row, column, action):
        # count how many same pieces to left and right
        count = 0
        for i in range(1, 7):
            if column - i < 0:
                break
            if self.board[row][column-i] !=  action:
                break
            count += 1

        for i in range(1, 7):
            if column + i > 6:
                break
            if self.board[row][column+i] !=  action:
                break
            count += 1

        if count >= 3:
            return True

        return False

    def check_win_column(self, row, column, action):
        # count how mant same pieces above and below
        count = 0
        for i in range(1, 6):
            if row - i < 0:
                break
            if self.board[row-i][column] !=  action:
                break
            count += 1

        for i in range(1, 6):
            if row + i > 5:
                break
            if self.board[row+i][column] !=  action:
                break
            count += 1

        if count >= 3:
            return True

        return False

    def check_win_diagonal(self, row, column, action):
        # count how many same pieces diagonally -
        count = 0
        for i in range(1, 7):
            if row + i > 5 or column + i > 6:
                break
            if self.board[row+i][column+i] != action:
                break
            count += 1

        for i in range(1, 7):
            if row - i < 0 or column - i < 0:
                break
            if self.board[row-i][column-i] != action:
                break
            count += 1


        if count >= 3:
            return True

        # count how many same pieces diagonally -
        count = 0

        for i in range(1, 7):
            if row + i > 5 or column - i < 0:
                break
            if self.board[row+i][column-i] != action:
                break
            count += 1

        for i in range(1, 7):
            if row - i < 0 or column + i > 6:
                break
            if self.board[row-i][column+i] != action:
                break
            count += 1

        if count >= 3:
            return True

        return False

    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(self.board)

