# from tictactoe import TicTacToeBoard
from monte_carlo_tree_search import MCTS
from TicTacToeChat import TicTacToeBoard


def test_correct_winner():
    board = TicTacToeBoard(
        board=[["X", "X", "X"], [" ", " ", " "], [" ", " ", " "]],
        turn=True,
    )
    assert board.is_winner("X")


def test_all_leafs_found():
    # Create a MCTS and check that after a few 100 iterations, it is completely drawn
    board = TicTacToeBoard(
        board=[["X", " ", "O"], ["O", "X", "X"], [" ", "X", "O"]],
        turn=True,
    )
    tree = MCTS()
    for _ in range(100):
        tree.do_rollout(board)
    tree.do_rollout(board)
    assert len(tree.children) == 4

    board = TicTacToeBoard(
        board=[["X", " ", "O"], [" ", "X", "X"], [" ", "X", "O"]],
        turn=False,
    )
    tree = MCTS()
    for _ in range(100):
        tree.do_rollout(board)
    tree.do_rollout(board)
    assert len(tree.children) == 11


def test_leafs_consistent():
    # Create a MCTS and check that every terminal state is consisten (Win/Loss/Drawn)
    board = TicTacToeBoard(
        board=[["X", " ", "O"], [" ", "X", " "], [" ", " ", "O"]],
        turn=False,
    )
    tree = MCTS()
    for _ in range(100):
        tree.do_rollout(board)
        for child in tree.children:
            if child.is_terminal():
                if child.is_draw():
                    assert tree.Q[child] / tree.N[child] == 0.5
                elif child.is_winner("X"):
                    assert tree.Q[child] / tree.N[child] == 0
                else:
                    assert tree.Q[child] / tree.N[child] == 1


def test_leafs_consistent2():
    # Create a MCTS and check that every terminal state is consisten (Win/Loss/Drawn)
    board = TicTacToeBoard(
        board=[[" ", " ", " "], [" ", " ", " "], [" ", " ", " "]],
        turn=True,
    )
    tree = MCTS()
    for _ in range(1000):
        tree.do_rollout(board)
    for child in tree.N:
        if child.is_terminal():
            if tree.N[child] > 0:
                if child.is_draw():
                    assert tree.Q[child] / tree.N[child] == 0.5
                elif child.is_winner("X"):
                    assert tree.Q[child] / tree.N[child] == 0
                else:
                    assert tree.Q[child] / tree.N[child] == 1


def test_all_runs_accounted():
    # The game graph is not a tree, especially the same game state can be reached on multiple paths.
    # Only for the top level holds that the children should be equal to all runs - one for setup
    from random import seed

    seed(42)
    board = TicTacToeBoard(
        board=[["X", " ", "O"], [" ", "X", " "], [" ", " ", "O"]],
        turn=False,
    )
    tree = MCTS()
    for i in range(100):
        tree.do_rollout(board)
        if i > 0:
            assert tree.N[board] - 1 == i
    assert sum([tree.N[c] for c in tree.children[board]]) == tree.N[board] - 1


def test_correct_choice_simple():
    # Create a MCTS and check that given two simple choices
    # Check that it chooses the winning move
    from random import seed

    seed(42)
    board = TicTacToeBoard(
        board=[["O", "O", " "], ["X", "X", "O"], [" ", "X", "X"]],
        turn=False,
    )
    tree = MCTS()
    for _ in range(100):
        tree.do_rollout(board)  # Add one child
        # assert tree._select(board)[-1] not in tree.children[board] # Check that the next one chosen is new
    choice = tree.choose(board)
    assert choice.board[0][2] is not None and choice.board[0][2] == "O"


def test_correct_choice_simple_inverse():
    # Create a MCTS and check that given three simple choices
    # it chooses the winning one
    from random import seed

    seed(42)
    board = TicTacToeBoard(
        board=[["X", "X", " "], ["O", "O", " "], [" ", "O", "O"]],
        turn=True,
    )
    tree = MCTS()
    for _ in range(100):
        tree.do_rollout(board)  # Add one child
        # assert tree._select(board)[-1] not in tree.children[board] # Check that the next one chosen is new
    choice = tree.choose(board)
    assert choice.board[0][2] == "X"


def test_find_winning_move():
    # Check if True chooses
    board = TicTacToeBoard(
        board=[[" ", "O", " "], [" ", "X", " "], [" ", " ", " "]],
        turn=True,
    )
    tree = MCTS()
    for _ in range(627 * 2):
        tree.do_rollout(board)
    tree.do_rollout(board)
    assert len(tree.terminal) == 627
    # Check that all childs are terminated correctly:
    for child in tree.children[board]:
        if child.board[0][0] == "X":
            assert tree.terminal[child] == 0.0
        elif child.board[0][2] == "X":
            assert tree.terminal[child] == 0.0
        elif child.board[1][0] == "X":
            assert tree.terminal[child] == 0.0
        elif child.board[1][2] == "X":
            assert tree.terminal[child] == 0.0
        elif child.board[2][0] == "X":
            assert tree.terminal[child] == 0.0
        elif child.board[2][1] == "X":
            assert tree.terminal[child] == 0.5
        elif child.board[2][2] == "X":
            assert tree.terminal[child] == 0.0
        else:
            assert 1 != 0


def test_find_winning_move2():
    # Check if True chooses
    board = TicTacToeBoard(
        board=[[" ", "O", " "], [" ", "X", " "], [" ", "X", " "]],
        turn=False,
    )
    tree = MCTS()
    for _ in range(239 * 2):
        tree.do_rollout(board)
    tree.do_rollout(board)
    assert len(tree.terminal) == 239
    # Check that all childs are terminated correctly:
    for child in tree.children[board]:
        if child.board[0][0] == "O":
            assert tree.terminal[child] == 0.5
        elif child.board[0][2] == "O":
            assert tree.terminal[child] == 0.5
        elif child.board[1][0] == "O":
            assert tree.terminal[child] == 0.0
        elif child.board[1][2] == "O":
            assert tree.terminal[child] == 0.0
        elif child.board[2][0] == "O":
            assert tree.terminal[child] == 0.5
        elif child.board[2][2] == "O":
            assert tree.terminal[child] == 0.5
        else:
            assert 1 != 0


def test_find_winning_move3():
    # Check if True chooses
    board = TicTacToeBoard(
        board=[[" ", "O", " "], [" ", "X", "O"], [" ", "X", " "]],
        turn=True,
    )
    tree = MCTS()
    for _ in range(85 * 2):
        tree.do_rollout(board)
    tree.do_rollout(board)
    assert len(tree.terminal) == 85
    # Check that all childs are terminated correctly:
    print([tree.terminal[child] for child in tree.children[board]])
    for child in tree.children[board]:
        if child.board[0][0] == "X":
            assert tree.terminal[child] == 0.0
        elif child.board[0][2] == "X":
            assert tree.terminal[child] == 0.0
        elif child.board[1][0] == "X":
            assert tree.terminal[child] == 0.0
        elif child.board[2][2] == "X":
            assert tree.terminal[child] == 1.0
        elif child.board[2][0] == "X":
            assert tree.terminal[child] == 1.0
        else:
            assert 1 != 0


def test_find_winning_move4():
    # No matter what O does, X still wins
    board = TicTacToeBoard(
        board=[[" ", "O", " "], [" ", "X", "O"], ["X", "X", " "]],
        turn=False,
    )
    tree = MCTS()
    for _ in range(26 * 2):
        tree.do_rollout(board)
    tree.do_rollout(board)
    assert len(tree.terminal) == 26
    # Check that all childs are terminated correctly:
    print([tree.terminal[child] for child in tree.children[board]])
    print([tree.Q[child] / tree.N[child] for child in tree.children[board]])
    for child in tree.children[board]:
        if child.board[0][0] == "O":
            assert tree.terminal[child] == 1.0
        elif child.board[0][2] == "O":
            assert tree.terminal[child] == 1.0
        elif child.board[1][0] == "O":
            assert tree.terminal[child] == 1.0
        elif child.board[2][2] == "O":
            assert tree.terminal[child] == 1.0
        else:
            assert 1 != 0


def test_find_winning_move5():
    # X must win
    board = TicTacToeBoard(
        board=[[" ", "O", " "], ["O", "X", "O"], ["X", "X", " "]],
        turn=True,
    )
    tree = MCTS()
    for _ in range(4 * 2):
        tree.do_rollout(board)
    tree.do_rollout(board)
    assert len(tree.terminal) == 4
    # Check that all childs are terminated correctly:
    print([tree.terminal[child] for child in tree.children[board]])
    print([tree.Q[child] / tree.N[child] for child in tree.children[board]])
    for child in tree.children[board]:
        if child.board[0][0] == "X":
            assert tree.terminal[child] == 1.0
        elif child.board[0][2] == "X":
            assert tree.terminal[child] == 1.0
        elif child.board[2][2] == "X":
            assert tree.terminal[child] == 1.0
        else:
            assert 1 != 0


def test_find_winning_move6():
    # X must win
    board = TicTacToeBoard(
        board=[["X", "O", " "], ["O", "X", "O"], ["X", "X", " "]],
        turn=False,
    )
    tree = MCTS()
    for _ in range(5 * 2):
        tree.do_rollout(board)
    tree.do_rollout(board)
    assert len(tree.terminal) == 3
    # Check that all childs are terminated correctly:
    print([tree.terminal[child] for child in tree.children[board]])
    print([tree.Q[child] / tree.N[child] for child in tree.children[board]])
    for child in tree.children[board]:
        if child.board[0][2] == "O":
            assert tree.terminal[child] == 1.0
        elif child.board[2][2] == "O":
            assert tree.terminal[child] == 1.0
        else:
            assert 1 != 0


def test_find_end_of_game():
    from random import seed

    seed(42)
    # Solve, why it sometimes starts really bad:
    board = TicTacToeBoard(
        board=[[" ", " ", " "], [" ", "X", " "], [" ", " ", " "]],
        turn=False,
    )
    tree = MCTS()
    for _ in range(10000):
        tree.do_rollout(board)
    for k in tree.terminal:
        assert k in tree.children
    for c in tree.children:
        assert c in tree.children
    assert len(tree.terminal) == 1837
    # TODO: Should not explore terminal nodes unnecessay often
    for node in tree.N:
        if node.is_terminal():
            assert tree.terminal[node] == tree.Q[node] / tree.N[node]


def test_find_good_moves():
    from random import seed

    seed(42)
    # Solve, why it sometimes starts really bad:
    board = TicTacToeBoard(
        board=[[" ", " ", " "], [" ", "X", " "], [" ", " ", " "]],
        turn=False,
    )
    tree = MCTS()
    for _ in range(1837 * 2):
        tree.do_rollout(board)
    tree.do_rollout(board)
    # Check that all childs are terminated correctly:
    [tree.terminal[child] for child in tree.children[board]]
    for child in tree.children[board]:
        if child.board[0][0] == "O":
            assert tree.terminal[child] == 0.5
        elif child.board[0][1] == "O":
            assert tree.terminal[child] == 0.0
        elif child.board[0][2] == "O":
            assert tree.terminal[child] == 0.5
        elif child.board[1][0] == "O":
            assert tree.terminal[child] == 0.0
        elif child.board[1][2] == "O":
            assert tree.terminal[child] == 0.0
        elif child.board[2][0] == "O":
            assert tree.terminal[child] == 0.5
        elif child.board[2][1] == "O":
            assert tree.terminal[child] == 0.0
        elif child.board[2][2] == "O":
            assert tree.terminal[child] == 0.5
        else:
            assert 1 != 0
    # TODO: Make this test not failing!
    selection = tree.choose(board)
    print(selection)
    draw_moves = [0, 2, 6, 8]
    draw_moves = [[1, 1], [1, 3], [3, 1], [3, 3]]
    # corner_selected = any([selection.tup[i] is not None and not selection.tup[i] for i in draw_moves])
    corner_selected = any(
        [
            selection.board[r][c] is not None and not selection.board[r][c]
            for ir, c in draw_moves
        ]
    )
    assert corner_selected, "None corner selected"
