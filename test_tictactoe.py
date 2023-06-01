from tictactoe import TicTacToeBoard, MCTS, _find_winner


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
    assert len(tree.children) == 4 

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
    assert len(tree.children) == 11


def test_leafs_consistent():
    # Create a MCTS and check that every terminal state is consisten (Win/Loss/Drawn)
    from random import seed
    seed(42)
    board = TicTacToeBoard(
        tup=(True, None, False, None, True, None, None, None, False),
        turn=False,
        winner=None,
        terminal=False,
    )
    tree = MCTS()
    for _ in range(100):
        tree.do_rollout(board)
        for child in tree.children:
            if child.is_terminal():
                if child.winner is None:
                    assert tree.Q[child] / tree.N[child] == 0.5
                elif child.winner:
                    assert tree.Q[child] / tree.N[child] == 0
                else:
                    assert tree.Q[child] / tree.N[child] == 1


def test_all_runs_accounted():
    # The game graph is not a tree, especially the same game state can be reached on multiple paths.
    # Only for the top level holds that the children should be equal to all runs - one for setup
    from random import seed
    seed(42)
    board = TicTacToeBoard(
        tup=(True, None, False, None, True, None, None, None, False),
        turn=False,
        winner=None,
        terminal=False,
    )
    tree = MCTS()
    for i in range(100):
        tree.do_rollout(board)
        if i > 0:
            assert tree.N[board]-1 == i
    assert sum([tree.N[c] for c in tree.children[board]]) == tree.N[board] - 1
        

def test_correct_choice_simple():
    # Create a MCTS and check that given two simple choices
    # Check that it chooses the winning one
    from random import seed
    seed(42)
    board = TicTacToeBoard(
        tup=(False, False, None, 
            True, True, False, 
            None, True, True),
        turn=False,
        winner=None,
        terminal=False,
    )
    tree = MCTS()
    for _ in range(100):
        tree.do_rollout(board)  # Add one child
        # assert tree._select(board)[-1] not in tree.children[board] # Check that the next one chosen is new
    choice = tree.choose(board)
    assert choice.tup[2] is not None and not choice.tup[2]


def test_correct_choice_simple_inverse():
    # Create a MCTS and check that given two simple choices
    # Check that it chooses the winning one
    from random import seed
    seed(42)
    board = TicTacToeBoard(
        tup=(True, True, None, 
            False, False, True, 
            None, False, False),
        turn=True,
        winner=None,
        terminal=False,
    )
    tree = MCTS()
    for _ in range(100):
        tree.do_rollout(board)  # Add one child
        # assert tree._select(board)[-1] not in tree.children[board] # Check that the next one chosen is new
    choice = tree.choose(board)
    assert choice.tup[2] is not None and choice.tup[2]


def test_correct_winner():
    board = TicTacToeBoard(
        tup=(True, True, True, None, None, None, None, None, None),
        turn=True,
        winner=None,
        terminal=False,
    )
    assert _find_winner(board.tup)


def test_find_good_moves():
    from random import seed
    seed(42)
    # Solve, why it sometimes starts really bad:
    board = TicTacToeBoard(
        tup=(None, None, None, None, True, None, None, None, None),
        turn=False,
        winner=None,
        terminal=False,
    )
    tree = MCTS()
    for _ in range(10000):
        tree.do_rollout(board)
    choice = tree.choose(board)
    print(choice)
    # TODO: Make this test not failing!
    draw_moves = [0,2,6,8]
    corner_selected = any([choice.tup[i] is not None and not choice.tup[i] for i in draw_moves])
    assert corner_selected, "None corner selected"


def test_visits_terminal_nodes_once():
    "TODO: Should not explore terminal nodes unnecessay often"
    from random import seed
    seed(42)
    board = TicTacToeBoard(
        tup=(False, False, None, 
            True, True, False, 
            None, True, True),
        turn=False,
        winner=None,
        terminal=False,
    )
    tree = MCTS()
    for _ in range(100):
        tree.do_rollout(board)  # Add one child
        assert tree._select(board)[-1] not in tree.children[board] # Check that the next one chosen is new
    choice = tree.choose(board)
    assert choice.tup[2] is not None and not choice.tup[2]