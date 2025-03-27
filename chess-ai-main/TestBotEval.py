import chess
from bot import ChessBot

bot = ChessBot()

# Sample positions to test
positions = {
    "Isolated pawn": "8/8/8/3P1P1P/8/8/8/8 w - - 0 1",
    "Doubled pawns": "8/8/8/3PPP2/8/8/8/8 w - - 0 1",
    "Passed pawn": "8/8/3P4/8/8/8/8/8 w - - 0 1",
    "Connected passed pawns": "8/8/3P4/2P5/8/8/8/8 w - - 0 1"
}

for name, fen in positions.items():
    board = chess.Board(fen)

    print(f"\n{name}:")
    print(f"FEN: {fen}")

    # Specifically test pawn structure
    pawn_structure = bot.evaluate_pawn_structure(board)
    print(f"Pawn Structure Score: {pawn_structure}")
