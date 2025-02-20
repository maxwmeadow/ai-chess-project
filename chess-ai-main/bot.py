import chess
from chess.svg import piece

from board import ChessBoard

class ChessBot:
    def __init__(self):
        self.pawn_table = [
            0, 0, 0, 0, 0, 0, 0, 0,
            50, 50, 50, 35, 35, 50, 50, 50,
            10, 10, 20, 40, 40, 20, 10, 10,
            5, 5, 10, 25, 25, 10, 5, 5,
            0, 0, 0, 20, 20, 0, 0, 0,
            5, -5, -10, 0, 0, -10, -5, 5,
            5, 10, 10, -5, -5, 10, 10, 5,
            0, 0, 0, 0, 0, 0, 0, 0
        ]

        self.knight_table = [
            -50, -40, -30, -30, -30, -30, -40, -50,
            -40, -20, 0, 0, 0, 0, -20, -40,
            -30, 0, 10, 15, 15, 10, 0, -30,
            -30, 5, 15, 20, 20, 15, 5, -30,
            -30, 0, 15, 20, 20, 15, 0, -30,
            -30, 5, 10, 15, 15, 10, 5, -30,
            -40, -20, 0, 5, 5, 0, -20, -40,
            -50, -40, -30, -30, -30, -30, -40, -50
        ]

        self.bishop_table = [
            -20, -10, -10, -10, -10, -10, -10, -20,
            -10, 5, 10, 10, 10, 10, 5, -10,
            -10, 10, 15, 20, 20, 15, 10, -10,
            -10, 10, 20, 30, 30, 20, 10, -10,
            -10, 10, 20, 30, 30, 20, 10, -10,
            -10, 10, 15, 20, 20, 15, 10, -10,
            -10, 5, 10, 10, 10, 10, 5, -10,
            -20, -10, -10, -10, -10, -10, -10, -20
        ]

    def evaluate_position(self, board):

        """

        Evaluate the current position.

        Positive values favor white, negative values favor black.

        """

        if board.is_game_over():

            if board.is_checkmate():
                return -10000 if board.turn else 10000

            return 0  # Draw

        score = 0

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue

            piece_values = {
                chess.PAWN: 100,
                chess.KNIGHT: 320,
                chess.BISHOP: 330,
                chess.ROOK: 500,
                chess.QUEEN: 900,
                chess.KING: 20000
            }.get(piece.piece_type, 0)

            if piece.color == chess.WHITE:
                score += piece_values
                if piece.piece_type == chess.PAWN:
                    score += self.pawn_table[square]
                elif piece.piece_type == chess.KNIGHT:
                    score += self.knight_table[square]
                elif piece.piece_type == chess.BISHOP:
                    score += self.bishop_table[square]

            else:
                score -= piece_values
                if piece.piece_type == chess.PAWN:
                    score -= self.pawn_table[chess.square_mirror(square)]
                elif piece.piece_type == chess.KNIGHT:
                    score -= self.knight_table[chess.square_mirror(square)]
                elif piece.piece_type == chess.BISHOP:
                    score += self.bishop_table[chess.square_mirror(square)]

        legal_moves = len(list(board.legal_moves))
        if board.turn:
            score += legal_moves * 2
        else:
            score -= legal_moves * 2

        if not board.turn:
            for square in [chess.E4, chess.E5, chess.D4, chess.D5]:
                piece = board.piece_at(square)
                if piece is not None:
                    if piece.color == board.turn:
                        score += 200
                    else:
                        score -= 200

        return score

    def minimax(self, board, depth, alpha, beta, maximizing_player):

        """

        Minimax implementation.

        Returns (best_score, best_move)

        """

        if depth == 0 or board.is_game_over():
            return self.evaluate_position(board), None

        best_move = None

        if maximizing_player:

            max_eval = float('-inf')

            for move in board.legal_moves:

                board.push(move)

                eval, _ = self.minimax(board, depth - 1, alpha, beta, False)

                board.pop()

                if eval > max_eval:
                    max_eval = eval

                    best_move = move

                alpha = max(alpha, eval)
                if beta <= alpha:
                    break

            return max_eval, best_move

        else:

            min_eval = float('inf')

            for move in board.legal_moves:

                board.push(move)

                eval, _ = self.minimax(board, depth - 1, alpha, beta, True)

                board.pop()

                if eval < min_eval:
                    min_eval = eval

                    best_move = move

                beta = min(beta, eval)
                if beta <= alpha:
                    break

            return min_eval, best_move

    def get_move(self, board: ChessBoard):

        """

        Main method to select the best move.

        """
        if isinstance(board, ChessBoard):
            board = board.get_board_state()

        score, best_move = self.minimax(board, depth=3, alpha=float('-inf'), beta=float('inf'), maximizing_player=board.turn)

        print("Best move found was " + str(best_move) + " with a score of " + str(score))

        return best_move