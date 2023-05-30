from tictactoe import TicTacToeBoard, MCTS


def test_all_leafs_found():
    # Create a MCTS and check that after a few 100 iterations, it is completely drawn
    board = TicTacToeBoard(
        tup=(True, None, False, False, True, True, None, True, False),
        turn=True,
        winner=None,
        terminal=False,
    )
    tree = MCTS()
    for _ in range(100):
        tree.do_rollout(board)
    tree.do_rollout(board)
    assert len(tree.children) == 2

    board = TicTacToeBoard(
        tup=(True, None, False, None, True, True, None, True, False),
        turn=False,
        winner=None,
        terminal=False,
    )
    tree = MCTS()
    for _ in range(100):
        tree.do_rollout(board)
    tree.do_rollout(board)
    assert len(tree.children) == 5


def test_leafs_consistent():
    # Create a MCTS and check that every terminal state is consisten (Win/Loss/Drawn)
    board = TicTacToeBoard(
        tup=(True, None, False, None, True, True, None, True, False),
        turn=False,
        winner=None,
        terminal=False,
    )
    tree = MCTS()
    for _ in range(100):
        tree.do_rollout(board)
    tree.do_rollout(board)
    for child in tree.children:
        if child.is_terminal():
            if child.winner is None:
                assert tree.Q[child] / tree.N[child] == 0.5
            elif child.winner:
                assert tree.Q[child] / tree.N[child] == 1
            else:
                assert tree.Q[child] / tree.N[child] == 0


def test_correct_choice():
    # Create a MCTS and check that given seven simple choices, it checks all and chooses the right one
    board = TicTacToeBoard(
        tup=(True, True, None, None, None, None, None, None, None),
        turn=True,
        winner=None,
        terminal=False,
    )
    tree = MCTS()
    for _ in range(7):
        tree.do_rollout(board)  # Add one child
        assert tree._select(board)[-1] not in tree.children[board]
    choice = tree.choose(board)
    assert choice.tup[2]


def test_correct_winner():
    board = TicTacToeBoard(
        tup=(True, True, True, None, None, None, None, None, None),
        turn=True,
        winner=None,
        terminal=False,
    )
