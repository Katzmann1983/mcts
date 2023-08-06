from abc import ABC, abstractmethod
import random

class ConnectFourBoard(Node):
    def __init__(self, board=None, turn=True):
        self.board = board or [[None] * 7 for _ in range(6)]
        self.turn = turn

    def find_children(self):
        children = set()
        for col in range(7):
            if self.is_valid_move(col):
                child_board = [row[:] for row in self.board]
                child_board = self.make_move(child_board, col)
                children.add(ConnectFourBoard(child_board, not self.turn))
        return children

    def find_random_child(self):
        valid_moves = [col for col in range(7) if self.is_valid_move(col)]
        random_col = random.choice(valid_moves)
        child_board = [row[:] for row in self.board]
        child_board = self.make_move(child_board, random_col)
        return ConnectFourBoard(child_board, not self.turn)

    def is_terminal(self):
        return self.is_draw() or self.is_winner('X') or self.is_winner('O')

    def reward(self):
        if self.is_winner('X'):
            return 1 if self.turn else 0
        elif self.is_winner('O'):
            return 0 if self.turn else 1
        else:
            return 0.5

    def is_valid_move(self, col):
        return self.board[0][col] is None

    def make_move(self, board, col):
        for row in range(5, -1, -1):
            if board[row][col] is None:
                board[row][col] = 'X' if self.turn else 'O'
                break
        return board

    def is_winner(self, player):
        for row in range(6):
            for col in range(4):
                if self.board[row][col] == player and self.board[row][col + 1] == player and self.board[row][col + 2] == player and self.board[row][col + 3] == player:
                    return True
        for row in range(3):
            for col in range(7):
                if self.board[row][col] == player and self.board[row + 1][col] == player and self.board[row + 2][col] == player and self.board[row + 3][col] == player:
                    return True
        for row in range(3):
            for col in range(4):
                if self.board[row][col] == player and self.board[row + 1][col + 1] == player and self.board[row + 2][col + 2] == player and self.board[row + 3][col + 3] == player:
                    return True
        for row in range(3):
            for col in range(3, 7):
                if self.board[row][col] == player and self.board[row + 1][col - 1] == player and self.board[row + 2][col - 2] == player and self.board[row + 3][col - 3] == player:
                    return True
        return False

    def is_draw(self):
        return all(self.board[0][col] is not None for col in range(7))

    def __hash__(self):
        return hash(str(self.board))

    def __eq__(self, other):
        return self.board == other.board
