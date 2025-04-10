import chess
import time
from bot import ChessBot
from board import ChessBoard


def print_position_evaluation(bot, board, position_fen=None):
    """Print detailed evaluation for a specific position"""
    if position_fen:
        board.set_fen(position_fen)

    print(f"\n===== POSITION EVALUATION =====")
    print(f"FEN: {board.fen()}")
    print(board)
    print("\n")

    # Evaluate each component
    material = bot.evaluate_material(board)
    position = bot.evaluate_piece_position(board)
    pawn_structure = bot.evaluate_pawn_structure(board)
    king_safety = bot.evaluate_king_safety(board)
    mobility = bot.evaluate_mobility(board)
    key_squares = bot.evaluate_key_square_control(board)

    # Total score
    total = bot.evaluate_position(board)

    # Print individual components
    print(f"Material: {material}")
    print(f"Position: {position}")
    print(f"Pawn Structure: {pawn_structure}")
    print(f"King Safety: {king_safety}")
    print(f"Mobility: {mobility}")
    print(f"Key Square Control: {key_squares}")
    print(f"TOTAL EVALUATION: {total}")

    # Also evaluate the position from black's perspective
    board.turn = not board.turn
    black_eval = -bot.evaluate_position(board)
    board.turn = not board.turn
    print(f"Evaluation from Black's perspective: {black_eval}")

    return total


def test_search_at_position(bot, board, position_fen=None, depth=4):
    """Test the search function at a specific position"""
    if position_fen:
        board.set_fen(position_fen)

    print(f"\n===== SEARCH TEST (Depth {depth}) =====")
    print(f"FEN: {board.fen()}")
    print(board)

    start_time = time.time()

    # Clear transposition tables before test
    bot.transposition_table = {}
    bot.killer_moves = {}

    # Run PVS search
    score, best_move = bot.pvs(board, depth, float('-inf'), float('inf'), board.turn)

    elapsed = time.time() - start_time

    print(f"Best move: {best_move}")
    print(f"Score: {score}")
    print(f"Time: {elapsed:.2f}s")

    # Test the move
    if best_move:
        print("\nTesting this move:")
        board.push(best_move)
        print(board)

        # Evaluate the resulting position
        opponent_eval = bot.evaluate_position(board)
        print(f"Position evaluation after move: {-opponent_eval} (from current player's perspective)")

        # Verify if score matches
        if abs(score - (-opponent_eval)) > 100:
            print(f"⚠️ WARNING: Search score ({score}) significantly differs from evaluation ({-opponent_eval})")

        # Restore board
        board.pop()

    return best_move, score


def test_specific_positions():
    """Test the bot on specific positions"""
    bot = ChessBot()

    test_positions = [
        # Starting position
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",

        # Position where g2g4 was played with inf score
        # Replace this with the actual position from your game
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",

        # Middle game position
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",

        # Complex position with many captures possible
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",

        # Endgame position
        "8/8/8/8/3k4/8/3P4/3K4 w - - 0 1"
    ]

    for i, fen in enumerate(test_positions):
        print(f"\n\n======== TEST POSITION {i + 1} ========")

        # Create a fresh board for each test
        board = chess.Board(fen)

        # Test components
        eval_score = print_position_evaluation(bot, board)

        # Test search
        move, score = test_search_at_position(bot, board, depth=3)

        # Test quiescence search
        q_score = bot.quiescence(board, float('-inf'), float('inf'), max_depth=3)
        print(f"\nQuiescence search score: {q_score}")

        # Try a higher depth search
        if i < 2:  # Only for first couple positions to save time
            move_d4, score_d4 = test_search_at_position(bot, board, depth=4)


def test_game_position(position_fen=None):
    """Test a specific game position with detailed diagnostics"""
    bot = ChessBot()

    if position_fen:
        board = chess.Board(position_fen)
    else:
        # Default to starting position
        board = chess.Board()

    # Evaluate position
    print_position_evaluation(bot, board)

    # Test move generation and ordering
    print("\n===== MOVE ORDERING =====")
    ordered_moves = bot.order_moves(board)
    for i, move in enumerate(ordered_moves[:10]):  # Print top 10 moves
        board.push(move)
        eval_after = -bot.evaluate_position(board)
        board.pop()
        print(f"{i + 1}. {move} -> Evaluation: {eval_after}")

    # Test incremental depths
    for depth in range(1, 6):
        test_search_at_position(bot, board, depth=depth)

    # Test with real get_move function (with time limit)
    print("\n===== TESTING get_move() =====")
    start_time = time.time()
    move = bot.get_move(ChessBoard(board))
    elapsed = time.time() - start_time
    print(f"Selected move: {move}")
    print(f"Time taken: {elapsed:.2f}s")


def debug_infinite_score(bot, board, position_fen=None):
    """Debug infinite scores by tracing the search path"""
    if position_fen:
        board.set_fen(position_fen)

    print(f"\n===== DEBUGGING INFINITE SCORE =====")
    print(f"FEN: {board.fen()}")
    print(board)

    # Hook into PVS with a debug wrapper
    original_pvs = bot.pvs

    def debug_pvs(board, depth, alpha, beta, maximizing_player, ply=0):
        # Print current state
        if depth >= 2:  # Only print for higher depths to reduce output
            print(f"Depth: {depth}, Ply: {ply}, Alpha: {alpha}, Beta: {beta}")

            if alpha == float('-inf') and beta == float('inf'):
                print("Full window search")
            else:
                print(f"Window: [{alpha}, {beta}]")

        # Call original PVS
        score, move = original_pvs(board, depth, alpha, beta, maximizing_player, ply)

        # Check for extreme scores
        if abs(score) > 9000 and depth > 1:
            print(f"⚠️ Extreme score {score} at depth {depth}, ply {ply}")
            if move:
                print(f"  Move: {move}")

        # Check for infinity
        if score == float('inf') or score == float('-inf'):
            print(f"⚠️ INFINITE SCORE at depth {depth}, ply {ply}")
            if move:
                print(f"  Move: {move}")

        return score, move

    # Replace the bot's PVS function temporarily
    bot.pvs = debug_pvs

    # Run search
    try:
        best_score, best_move = bot.pvs(board, 3, float('-inf'), float('inf'), board.turn)
        print(f"\nFinal result: Move {best_move} with score {best_score}")
    finally:
        # Restore original PVS
        bot.pvs = original_pvs


if __name__ == "__main__":
    # First, test standard positions
    test_specific_positions()

    # Test a specific position where g2g4 was selected with inf score
    # Replace this FEN with the actual position from your game
    problem_position = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    bot = ChessBot()
    board = chess.Board(problem_position)

    # Debug the position specifically
    print("\n\n======== DEBUGGING PROBLEM POSITION ========")
    debug_infinite_score(bot, board)

    # Get deeper analysis
    test_game_position(problem_position)