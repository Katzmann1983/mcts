import random
from monte_carlo_tree_search import MCTS, Node


class TicTacToeBoard(Node):
    def __init__(self, board=None, turn=True, tup=None, **kwargs):
        self.board = board or [[" "] * 3 for _ in range(3)]
        self.turn = turn
        if tup:
            # (True, None, False, False, True, True, None, True, False)
            for i, t in enumerate(tup):
                sign = " "
                if t is not None:
                    sign = "X" if t else "O"
                row = i % 3
                col = i // 3
                self.board[row][col] = sign

    def find_children(self):
        children = set()
        for row in range(3):
            for col in range(3):
                if self.board[row][col] == " ":
                    child_board = [row[:] for row in self.board]
                    child_board[row][col] = "X" if self.turn else "O"
                    children.add(TicTacToeBoard(child_board, not self.turn))
        return children

    def find_random_child(self):
        valid_moves = []
        for row in range(3):
            for col in range(3):
                if self.board[row][col] == " ":
                    valid_moves.append((row, col))
        random_row, random_col = random.choice(valid_moves)
        child_board = [row[:] for row in self.board]
        child_board[random_row][random_col] = "X" if self.turn else "O"
        return TicTacToeBoard(child_board, not self.turn)

    def is_terminal(self):
        return self.is_winner("X") or self.is_winner("O") or self.is_fully_played()

    def reward(self):
        """Rewards depend on who is the current player"""
        if self.is_winner("X"):
            return 1
        elif self.is_winner("O"):
            return 0
        else:
            return 0.5

    def is_winner(self, player):
        # Check rows, columns, and diagonals for a win
        for i in range(3):
            if all(self.board[i][j] == player for j in range(3)):
                return True
            if all(self.board[j][i] == player for j in range(3)):
                return True
        if all(self.board[i][i] == player for i in range(3)):
            return True
        if all(self.board[i][2 - i] == player for i in range(3)):
            return True
        return False

    def is_fully_played(self):
        return all(self.board[i][j] != " " for i in range(3) for j in range(3))

    def __hash__(self):
        return hash(str(self.board))

    def __eq__(self, other):
        return self.board == other.board

    def to_pretty_string(self):
        # to_char = lambda v: ("X" if v is True else ("O" if v is False else " "))
        rows = [[(self.board[row][col]) for col in range(3)] for row in range(3)]
        return (
            "\n  1 2 3\n"
            + "\n".join(str(i + 1) + " " + " ".join(row) for i, row in enumerate(rows))
            + "\n"
        )

    def __str__(self) -> str:
        return self.to_pretty_string()

    @property
    def winner(self):
        if self.is_winner("X"):
            return True
        elif self.is_winner("O"):
            return False
        else:
            return None

    @property
    def tup(self):
        board_to_tup = {"X": True, "O": False, " ": None}
        return [
            board_to_tup[self.board[row][col]] for col in range(3) for row in range(3)
        ]


def new_tic_tac_toe_board():
    return TicTacToeBoard()


def play_game():
    tree = MCTS()
    board = new_tic_tac_toe_board()
    print(board.to_pretty_string())
    while True:
        row_col = input("enter row,col: ")
        row, col = map(int, row_col.split(","))
        if board.board[row - 1][col - 1] != " ":
            raise RuntimeError("Invalid move")
        board.board[row - 1][col - 1] = "X"
        board.turn = not board.turn
        print(board.to_pretty_string())
        if board.is_terminal():
            break
        # You can train as you go, or only at the beginning.
        # Here, we train as we go, doing fifty rollouts each turn.
        for _ in range(100):
            tree.do_rollout(board)
        board = tree.choose(board)
        print(board)
        if board.is_terminal():
            break


if __name__ == "__main__":
    play_game()
