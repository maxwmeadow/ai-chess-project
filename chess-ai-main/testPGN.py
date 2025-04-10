import chess
from pgn_reader import PGNOpeningBook


def test_pgn_reader():
    # Create the opening book
    book = PGNOpeningBook(
        pgn_path="books/lichess_elite_2025-02.pgn",
        max_moves=15,
        min_frequency=2,
        max_positions=50000,
        cache_file="lichess_elite_test.openings"
    )

    # Print statistics
    book.print_stats()

    # Test a few common positions
    test_positions = [
        # Starting position
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        # King's pawn opening
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        # Sicilian Defense
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
        # Queen's Gambit
        "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq d3 0 2"
    ]

    for fen in test_positions:
        board = chess.Board(fen)
        print(f"\nPosition: {board.fen()}")

        # Get the best moves
        weighted_moves = book.get_weighted_moves(board)
        if weighted_moves:
            print("Top moves:")
            for move, weight in weighted_moves[:5]:  # Show top 5 moves
                print(f"  {board.san(move)}: {weight}")
        else:
            print("No moves found in book")

        # Get a single move
        move = book.get_move(board)
        if move:
            print(f"Selected move: {board.san(move)}")
        else:
            print("No move selected")


if __name__ == "__main__":
    test_pgn_reader()