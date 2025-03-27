from bot import ChessBot
import chess

bot = ChessBot()

test_cases = [
    {
        "name": "Simple development",
        "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "expected_moves": ["f1b5", "f1c4", "d2d4", "b1c3"]
    },
    {
        "name": "Tactics incoming",
        "fen": "rnbqkbnr/pp1ppppp/8/2p5/3P4/5N2/PPP2PPP/RNBQKB1R w KQkq c6 0 3",
        "expected_moves": ["d4c5", "c2c4", "c1f4"]
    },
    {
        "name": "Knight outpost",
        "fen": "r2q1rk1/pp1nbppp/3bpn2/2p5/2P5/2N1PN2/PPQ1BPPP/R1B2RK1 w - - 0 10",
        "expected_moves": ["d2d4", "b2b3", "b1d2", "c1d2"]
    },
    {
        "name": "Open file & tempo",
        "fen": "r1bq1rk1/ppp2ppp/2n2n2/2bp4/4P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7",
        "expected_moves": ["c3d5", "e4d5", "c1g5", "d1e2"]
    },
    {
        "name": "Attacking chance",
        "fen": "r1bq1rk1/pp1nbppp/2n1p3/3pP3/2pP4/2N2N2/PPQ1BPPP/R1B2RK1 w - - 0 10",
        "expected_moves": ["b1d2", "f1d1", "h2h4", "c1g5"]
    },
    {
        "name": "Semi-open c-file",
        "fen": "r2q1rk1/pp1b1ppp/2n1pn2/2bp4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 9",
        "expected_moves": ["a1c1", "d4c5", "c3a4", "d1e2"]
    }
]

passed = 0
total = len(test_cases)

for test in test_cases:
    board = chess.Board(test["fen"])
    score, move = bot.minimax(board, depth=3, alpha=float('-inf'), beta=float('inf'), maximizing_player=board.turn)
    move_uci = move.uci() if move else "None"

    if move_uci in test["expected_moves"]:
        result = "[✓]"
        passed += 1
    else:
        result = "[✗]"

    print(f"{result} {test['name']}: Best move {move_uci} (score: {score}) — expected: {test['expected_moves']}")

print(f"\nAccuracy: {passed}/{total} correct")
