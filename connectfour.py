import random
from monte_carlo_tree_search import MCTS, Node


class ConnectFourGame(Node):
    NR_COLS = 7
    NR_ROWS = 6

    def __init__(self, board=None, turn=True):
        self.board = board or [[" "] * self.NR_COLS for _ in range(self.NR_ROWS)]
        self.turn = turn

    def find_children(self):
        children = set()
        for col in range(self.NR_COLS):
            if self.is_valid_move(col):
                child_board = [row[:] for row in self.board]
                child_board = self.make_move(child_board, col)
                children.add(ConnectFourGame(child_board, not self.turn))
        return children

    def find_random_child(self):
        valid_moves = [col for col in range(self.NR_COLS) if self.is_valid_move(col)]
        random_col = random.choice(valid_moves)
        child_board = [row[:] for row in self.board]
        child_board = self.make_move(child_board, random_col)
        return ConnectFourGame(child_board, not self.turn)

    def is_terminal(self):
        return self.is_draw() or self.is_winner("X") or self.is_winner("O")

    def reward(self):
        if self.is_winner("X"):
            return 1
        elif self.is_winner("O"):
            return 0
        else:
            return 0.5

    def is_valid_move(self, col):
        return self.board[0][col] == " "

    def make_move(self, board, col):
        for row in range(5, -1, -1):
            if board[row][col] == " ":
                board[row][col] = "X" if self.turn else "O"
                break
        return board

    def is_winner(self, player):
        for row in range(self.NR_ROWS):
            for col in range(self.NR_COLS):
                # Horizontal check
                if col + 3 < self.NR_COLS and all(
                    self.board[row][col + i] == player for i in range(4)
                ):
                    return True
                # Vertical check
                if row + 3 < self.NR_ROWS and all(
                    self.board[row + i][col] == player for i in range(4)
                ):
                    return True
                # Diagonal check (down-right)
                if (
                    col + 3 < self.NR_COLS
                    and row + 3 < self.NR_ROWS
                    and all(self.board[row + i][col + i] == player for i in range(4))
                ):
                    return True
                # Diagonal check (down-left)
                if (
                    col - 3 >= 0
                    and row + 3 < self.NR_ROWS
                    and all(self.board[row + i][col - i] == player for i in range(4))
                ):
                    return True
        return False

    def is_draw(self):
        return all(self.board[0][col] != " " for col in range(self.NR_COLS))

    def __hash__(self):
        return hash(str(self.board))

    def __eq__(self, other):
        return self.board == other.board

    def to_pretty_string(self):
        # to_char = lambda v: ("X" if v is True else ("O" if v is False else " "))
        rows = [
            [(self.board[row][col]) for col in range(self.NR_COLS)]
            for row in range(self.NR_ROWS)
        ]
        return (
            "\n  "
            + " ".join([str(i + 1) for i in range(self.NR_COLS)])
            + "\n"
            + "\n".join(str(i + 1) + " " + " ".join(row) for i, row in enumerate(rows))
            + "\n"
        )

    def __str__(self) -> str:
        return self.to_pretty_string()


def play_game():
    tree = MCTS()
    game = ConnectFourGame()
    print(game.to_pretty_string())
    while True:
        col = input("enter col: ")
        col = int(col)
        if not game.is_valid_move(col):
            raise RuntimeError("Invalid move")
        game = ConnectFourGame(game.make_move(game.board, col - 1), turn=False)
        print(game.to_pretty_string())
        if game.is_terminal():
            break
        # You can train as you go, or only at the beginning.
        # Here, we train as we go, doing fifty rollouts each turn.
        for _ in range(10000):
            tree.do_rollout(game)
        game = tree.choose(game)
        print(game)
        if game.is_terminal():
            break


def profile_tree():
    import cProfile

    profiler = cProfile.Profile()
    profiler.enable()
    tree = MCTS()
    game = ConnectFourGame()
    for _ in range(100):
        tree.do_rollout(game)
    profiler.disable()
    profiler.print_stats()


if __name__ == "__main__":
    # play_game()
    profile_tree()
