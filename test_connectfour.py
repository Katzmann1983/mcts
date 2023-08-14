from connectfour import ConnectFourGame


def test_winner_states():
    new_game = ConnectFourGame()
    # Columns
    PLAYER = "X"
    for i in range(new_game.NR_COLS):
        new_board = new_game.make_move(new_game.board, i)
        new_board = new_game.make_move(new_board, i)
        new_board = new_game.make_move(new_board, i)
        new_board = new_game.make_move(new_board, i)
    simple_game = ConnectFourGame(new_board, PLAYER)
    new_board = new_game.make_move(new_game.board, 0)
    new_board = new_game.make_move(new_board, 1)
    new_board = new_game.make_move(new_board, 2)
    new_board = new_game.make_move(new_board, 3)
    assert simple_game.is_winner(player=PLAYER)
