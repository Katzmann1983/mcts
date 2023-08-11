from monte_carlo_tree_search import MCTS
from TicTacToeChat import TicTacToeBoard


def get_choice(board, child):
    differences = 0
    index = -1
    for i, (b, c) in enumerate(zip(board.tup, child.tup)):
        if b != c:
            differences += 1
            index = i
    assert differences == 1
    return index


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
    # Create a MCTS and check that every terminal state is consistent (Win/Loss/Drawn)
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
                elif not child.winner:
                    assert tree.Q[child] / tree.N[child] == 1
                else:
                    assert False, "Should never happen"


def test_leafs_consistent2():
    # Create a MCTS and check that every terminal state is consistent (Win/Loss/Drawn)
    board = TicTacToeBoard(
        tup=(None, None, None, None, None, None, None, None, None),
        turn=True,
        winner=None,
        terminal=False,
    )
    tree = MCTS()
    for _ in range(1000):
        tree.do_rollout(board)
    for child in tree.N:
        if child.is_terminal():
            if tree.N[child] > 0:
                if child.winner is None:
                    assert tree.Q[child] / tree.N[child] == 0.5
                elif child.winner:
                    assert tree.Q[child] / tree.N[child] == 0
                else:
                    assert tree.Q[child] / tree.N[child] == 1
                assert tree.terminal[child] == tree.Q[child] / tree.N[child]
                if child.turn:
                    assert tree.Q[child] / tree.N[child] == 1
                else:
                    assert tree.Q[child] / tree.N[child] == 0


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
            assert tree.N[board] - 1 == i
    assert sum([tree.N[c] for c in tree.children[board]]) == tree.N[board] - 1


def test_correct_choice_simple():
    # Create a MCTS and check that given two simple choices
    # Check that it chooses the winning move
    from random import seed

    seed(42)
    board = TicTacToeBoard(
        tup=(False, False, None, True, True, False, None, True, True),
        turn=False,
        winner=None,
        terminal=False,
    )
    tree = MCTS()
    for _ in range(100):
        tree.do_rollout(board)  # Add one child
        # assert tree._select(board)[-1] not in tree.children[board] # Check that the next one chosen is new
    child = tree.choose(board)
    assert child.tup[2] is not None and not child.tup[2]
    assert get_choice(child, board) == 2


def test_correct_choice_simple_inverse():
    # Create a MCTS and check that given two simple choices
    # Check that it chooses the winning one
    from random import seed

    seed(42)
    board = TicTacToeBoard(
        tup=(True, True, None, False, False, True, None, False, False),
        turn=True,
        winner=None,
        terminal=False,
    )
    tree = MCTS()
    for _ in range(100):
        tree.do_rollout(board)  # Add one child
        # assert tree._select(board)[-1] not in tree.children[board] # Check that the next one chosen is new
    child = tree.choose(board)
    assert child.tup[2] is not None and child.tup[2]
    assert get_choice(child, board) == 2


def test_correct_winner():
    board = TicTacToeBoard(
        tup=(True, True, True, None, None, None, None, None, None),
        turn=True,
        winner=None,
        terminal=False,
    )
    assert board.winner


def test_terminal_nodes_visited_once():
    board = TicTacToeBoard(
        tup=(None, None, None, None, True, None, None, None, None),
        turn=False,
        winner=None,
        terminal=False,
    )
    tree = MCTS()
    for _ in range(10000):
        tree.do_rollout(board)  # Add one child
    for node in tree.N:
        if node.is_terminal():
            assert tree.N[node] == 1


def test_find_end_of_game():
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
    for k in tree.terminal:
        assert k in tree.children
    for c in tree.children:
        assert c in tree.children
    assert len(tree.terminal) == 1837


def test_find_winning_move1():
    # Check if False chooses boarder
    board = TicTacToeBoard(
        tup=(None, None, None, None, True, None, None, None, None),
        turn=False,
        winner=None,
        terminal=False,
    )
    tree = MCTS()
    for _ in range(1837 * 2):
        tree.do_rollout(board)
    assert len(tree.terminal) == 1837
    tree.do_rollout(board)
    assert len(tree.terminal) == 1837
    # Check that all childs are terminated correctly:
    print([tree.terminal[child] for child in tree.children[board]])
    print([tree.Q[child] / tree.N[child] for child in tree.children[board]])
    for child in tree.children[board]:
        choice = get_choice(board, child)
        if choice == 0:
            assert tree.terminal[child] == 0.5
        elif choice == 1:
            assert tree.terminal[child] == 0
        elif choice == 2:
            assert tree.terminal[child] == 0.5
        elif choice == 3:
            assert tree.terminal[child] == 0
        elif choice == 5:
            assert tree.terminal[child] == 0
        elif choice == 6:
            assert tree.terminal[child] == 0.5
        elif choice == 7:
            assert tree.terminal[child] == 0
        elif choice == 8:
            assert tree.terminal[child] == 0.5
        else:
            assert False, f"This choice {choice} should never happen!"
    # TODO: Make this test not failing!
    selection = tree.choose(board)
    choice = get_choice(board, selection)
    print(selection)
    draw_moves = [0, 2, 6, 8]
    assert choice in draw_moves, "Middle point chosen! Bad MTCS"


def test_find_winning_move2():
    # Check that true chooses winner
    board = TicTacToeBoard(
        tup=(None, False, None, None, True, None, None, None, None),
        turn=True,
        winner=None,
        terminal=False,
    )
    tree = MCTS()
    for _ in range(627 * 2):
        tree.do_rollout(board)
    tree.do_rollout(board)
    assert len(tree.terminal) == 627
    # Check that all childs are terminated correctly:
    print([tree.terminal[child] for child in tree.children[board]])
    print([tree.Q[child] / tree.N[child] for child in tree.children[board]])
    for child in tree.children[board]:
        choice = get_choice(board, child)
        if choice == 0:
            assert tree.terminal[child] == 0.0
        elif choice == 2:
            assert tree.terminal[child] == 0.0
        elif choice == 3:
            assert tree.terminal[child] == 0
        elif choice == 5:
            assert tree.terminal[child] == 0
        elif choice == 6:
            assert tree.terminal[child] == 0.0
        elif choice == 7:
            assert tree.terminal[child] == 0.5
        elif choice == 8:
            assert tree.terminal[child] == 0.0
        else:
            assert False, f"This choice {choice} should never happen!"


def test_find_winning_move3():
    # X must win
    board = TicTacToeBoard(
        tup=(None, False, None, None, True, None, None, True, None),
        turn=False,
        winner=None,
        terminal=False,
    )
    tree = MCTS()
    for _ in range(206 * 3):
        tree.do_rollout(board)
    tree.do_rollout(board)
    assert len(tree.terminal) == 239
    # Check that all childs are terminated correctly:
    print([tree.terminal[child] for child in tree.children[board]])
    print([tree.Q[child] / tree.N[child] for child in tree.children[board]])
    for child in tree.children[board]:
        choice = get_choice(board, child)
        if choice == 0:
            assert tree.terminal[child] == 0.5
        elif choice == 2:
            assert tree.terminal[child] == 0.5
        elif choice == 3:
            assert tree.terminal[child] == 0.0
        elif choice == 5:
            assert tree.terminal[child] == 0.0
        elif choice == 6:
            assert tree.terminal[child] == 0.5
        elif choice == 8:
            assert tree.terminal[child] == 0.5
        else:
            assert False, f"This choice {choice} should never happen!"


def test_find_winning_move4():
    board = TicTacToeBoard(
        tup=(None, False, False, None, True, None, None, True, None),
        turn=True,
        winner=None,
        terminal=False,
    )
    tree = MCTS()
    for _ in range(76 * 3):
        tree.do_rollout(board)
    tree.do_rollout(board)
    assert len(tree.terminal) == 76
    # Check that all childs are terminated correctly:
    print([tree.terminal[child] for child in tree.children[board]])
    print([tree.Q[child] / tree.N[child] for child in tree.children[board]])
    for child in tree.children[board]:
        choice = get_choice(board, child)
        if choice == 0:
            assert tree.terminal[child] == 0.5
        elif choice == 3:
            assert tree.terminal[child] == 1.0
        elif choice == 5:
            assert tree.terminal[child] == 1.0
        elif choice == 6:
            assert tree.terminal[child] == 1.0
        elif choice == 8:
            assert tree.terminal[child] == 1.0
        else:
            assert False, f"This choice {choice} should never happen!"


def test_find_winning_move5():
    # X must win
    board = TicTacToeBoard(
        tup=(True, False, False, None, True, None, None, True, None),
        turn=False,
        winner=None,
        terminal=False,
    )
    tree = MCTS()
    for _ in range(26 * 3):
        tree.do_rollout(board)
    tree.do_rollout(board)
    assert len(tree.terminal) == 31
    # Check that all childs are terminated correctly:
    print([tree.terminal[child] for child in tree.children[board]])
    print([tree.Q[child] / tree.N[child] for child in tree.children[board]])
    for child in tree.children[board]:
        choice = get_choice(board, child)
        if choice == 3:
            assert tree.terminal[child] == 0
        elif choice == 5:
            assert tree.terminal[child] == 0
        elif choice == 6:
            assert tree.terminal[child] == 0
        elif choice == 8:
            assert tree.terminal[child] == 0.5
        else:
            assert False, f"This choice {choice} should never happen!"


def test_find_winning_move6():
    board = TicTacToeBoard(
        tup=(True, False, False, None, True, None, None, True, False),
        turn=True,
        winner=None,
        terminal=False,
    )
    tree = MCTS()
    for _ in range(9 * 3):
        tree.do_rollout(board)
    tree.do_rollout(board)
    assert len(tree.terminal) == 12
    # Check that all childs are terminated correctly:
    print([tree.terminal[child] for child in tree.children[board]])
    print([tree.Q[child] / tree.N[child] for child in tree.children[board]])
    for child in tree.children[board]:
        choice = get_choice(board, child)
        if choice == 3:
            assert tree.terminal[child] == 1.0
        elif choice == 5:
            assert tree.terminal[child] == 0.5
        elif choice == 6:
            assert tree.terminal[child] == 1.0
        else:
            assert False, f"This choice {choice} should never happen!"


def test_find_winning_move7():
    board = TicTacToeBoard(
        tup=(True, False, False, True, True, None, None, True, False),
        turn=False,
        winner=None,
        terminal=False,
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
        choice = get_choice(board, child)
        if choice == 5:
            assert tree.terminal[child] == 1.0
        elif choice == 6:
            assert tree.terminal[child] == 0.0
        else:
            assert False, f"This choice {choice} should never happen!"


def test_find_winning_move8():
    board = TicTacToeBoard(
        tup=(True, False, False, True, True, None, False, True, False),
        turn=True,
        winner=None,
        terminal=False,
    )
    tree = MCTS()
    for _ in range(2 * 2):
        tree.do_rollout(board)
    tree.do_rollout(board)
    assert len(tree.terminal) == 2
    # Check that all childs are terminated correctly:
    print([tree.terminal[child] for child in tree.children[board]])
    print([tree.Q[child] / tree.N[child] for child in tree.children[board]])
    for child in tree.children[board]:
        choice = get_choice(board, child)
        if choice == 5:
            assert tree.terminal[child] == 0
        else:
            assert False, f"This choice {choice} should never happen!"
