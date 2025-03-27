import chess
from chess.svg import piece
import chess.polyglot

from board import ChessBoard
from opening_book import OpeningBook, PolyglotBook, create_simple_opening_book

def get_game_phase(board):
    piece_values = {
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
    }
    npm = 0

    for pt in piece_values:
        npm += len(board.pieces(pt, True)) * piece_values[pt]
        npm += len(board.pieces(pt, False)) * piece_values[pt]

    return min(npm, 256)

def interpolate(mg_score, eg_score, phase):
    return ((mg_score * phase) + (eg_score * (256 - phase))) // 256

class ChessBot:
    def __init__(self):

        self.opening_book = OpeningBook(create_simple_opening_book(), max_book_depth = 8)
        self.polyglot_book = PolyglotBook("books/gm2600.bin")

        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }

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

        self.rook_table = [
            0, 0, 5, 10, 10, 5, 0, 0,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            5, 10, 10, 10, 10, 10, 10, 5,
            0, 0, 0, 0, 0, 0, 0, 0
        ]

        self.queen_table = [
            -20, -10, -10, -5, -5, -10, -10, -20,
            -10, 0, 0, 0, 0, 0, 0, -10,
            -10, 0, 5, 5, 5, 5, 0, -10,
            -5, 0, 5, 5, 5, 5, 0, -5,
            0, 0, 5, 5, 5, 5, 0, -5,
            -10, 5, 5, 5, 5, 5, 0, -10,
            -10, 0, 5, 0, 0, 0, 0, -10,
            -20, -10, -10, -5, -5, -10, -10, -20
        ]

        self.king_table_mg = [
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -20, -30, -30, -40, -40, -30, -30, -20,
            -10, -20, -20, -20, -20, -20, -20, -10,
            20, 20, 0, 0, 0, 0, 20, 20,
            20, 30, 10, 0, 0, 10, 30, 20
        ]

        self.king_table_eg = [
            -50, -40, -30, -20, -20, -30, -40, -50,
            -30, -20, -10, 0, 0, -10, -20, -30,
            -30, -10, 20, 30, 30, 20, -10, -30,
            -30, -10, 30, 40, 40, 30, -10, -30,
            -30, -10, 30, 40, 40, 30, -10, -30,
            -30, -10, 20, 30, 30, 20, -10, -30,
            -30, -30, 0, 0, 0, 0, -30, -30,
            -50, -30, -30, -30, -30, -30, -30, -50
        ]

        self.position_scores = {
            chess.PAWN: self.pawn_table,
            chess.KNIGHT: self.knight_table,
            chess.BISHOP: self.bishop_table,
            chess.ROOK: self.rook_table,
            chess.QUEEN: self.queen_table,
            chess.KING: self.king_table_mg
        }

    def evaluate_piece_position(self, board):
        score_mg = 0
        score_eg = 0

        tables_mg = {
            chess.PAWN: self.pawn_table,
            chess.KNIGHT: self.knight_table,
            chess.BISHOP: self.bishop_table,
            chess.ROOK: self.rook_table,
            chess.QUEEN: self.queen_table,
            chess.KING: self.king_table_mg
        }

        tables_eg = dict(tables_mg)
        tables_eg[chess.KING] = self.king_table_eg

        for color in [True, False]:
            mult = 1 if color else -1

            for piece_type in tables_mg:
                for square in board.pieces(piece_type, color):
                    rank_square = square if color else chess.square_mirror(square)
                    score_mg += mult * tables_mg[piece_type][rank_square]
                    score_eg += mult * tables_eg[piece_type][rank_square]

        phase = get_game_phase(board)
        return interpolate(score_mg, score_eg, phase)

    def evaluate_material(self, board):

        score = 0

        # Basic material count

        for piece in self.piece_values:
            score += len(board.pieces(piece, True)) * self.piece_values[piece]

            score -= len(board.pieces(piece, False)) * self.piece_values[piece]

        # Bishop pair bonus

        if len(board.pieces(chess.BISHOP, True)) >= 2:
            score += 50

        if len(board.pieces(chess.BISHOP, False)) >= 2:
            score -= 50

        # Rook pair penalty
        if len(board.pieces(chess.ROOK, True)) >= 2:
            score -= 10

        if len(board.pieces(chess.ROOK, False)) >= 2:
            score += 10

        return score

    def evaluate_pawn_structure(self, board):
        white_score = 0
        black_score = 0

        for color in [True, False]:
            pawns = list(board.pieces(chess.PAWN, color))
            mult = 1 if color else -1
            score = 0

            #Pawn files
            pawn_files = set(chess.square_file(p) for p in pawns)

            #Isolated pawns penalty
            for pawn in pawns:
                file = chess.square_file(pawn)

                adjacent_files = [file - 1, file + 1]
                if not any(adj_file in pawn_files for adj_file in adjacent_files if 0 <= adj_file < 8):
                    score -= 15

            #Doubled pawns penalty
            file_counts = {}
            for pawn in pawns:
                file = chess.square_file(pawn)
                file_counts[file] = file_counts.get(file, 0) + 1

            for count in file_counts.values():
                if count > 1:
                    score -= 15 * (count - 1)

            #Passed pawn bonus
            for pawn in pawns:
                rank = chess.square_rank(pawn)
                file = chess.square_file(pawn)

                is_passed = True
                direction = 1 if color else -1

                for check_file in [file - 1, file, file + 1]:
                    if 0 <= check_file < 8:
                        for check_rank in range(rank + direction, 8 if color else -1, direction):
                            check_square = chess.square(check_file, check_rank)
                            blocking_piece = board.piece_at(check_square)
                            if blocking_piece and blocking_piece.piece_type == chess.PAWN and blocking_piece.color != color:
                                is_passed = False
                                break
                        if not is_passed:
                            break

                if is_passed:
                    promotion_bonus = 20 + abs(rank - (0 if color else 7)) * 5
                    score += promotion_bonus

                    #Connected passed pawn bonus
                    for other in pawns:
                        if other == pawn:
                            continue
                        other_file = chess.square_file(other)
                        other_rank = chess.square_rank(other)
                        if abs(other_file - file) == 1 and abs(other_rank - rank) <= 1:
                            score += 20
                            break

            if color:
                white_score += score
            else:
                black_score += score

        return white_score - black_score


    def evaluate_king_safety(self, board):
        score = 0

        for color in [True, False]:
            multiplier = 1 if color else -1
            enemy_color = not color
            king_square = board.king(color)
            if king_square is None:
                continue

            king_file = chess.square_file(king_square)
            king_rank = chess.square_rank(king_square)

            shield_score = 0
            open_file_penalty = 0

            for file in range(max(0, king_file - 1), min(8, king_file + 2)):
                shield_rank = king_rank + (1 if color else -1)
                if 0 <= shield_rank < 8:
                    shield_square = chess.square(file, shield_rank)
                    piece = board.piece_at(shield_square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        shield_score += 10

                # Check open file penalty
                has_friendly_pawn = any(
                    board.piece_at(chess.square(file, r)) == chess.Piece(chess.PAWN, color)
                    for r in range(8)
                )
                if not has_friendly_pawn:
                    open_file_penalty += 20

            shield_score = min(shield_score, 30)
            score += (shield_score - open_file_penalty) * multiplier

            # Tropism: enemy piece proximity to king
            for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                for sq in board.pieces(piece_type, enemy_color):
                    dist = chess.square_distance(king_square, sq)
                    if dist <= 3:
                        score -= (10 - 2 * dist) * multiplier

        return score


    def evaluate_position(self, board):

        if board.is_game_over():

            if board.is_checkmate():
                return -10000 if board.turn else 10000

            return 0  # Draw

        score = 0

        # Material and piece position evaluation

        score += self.evaluate_material(board)

        score += self.evaluate_piece_position(board)

        # Pawn structure

        score += self.evaluate_pawn_structure(board)

        # King safety

        score += self.evaluate_king_safety(board)

        score += self.evaluate_mobility(board)

        score += self.evaluate_key_square_control(board)

        return score

    def evaluate_mobility(self, board):
        weights = {
            chess.KNIGHT: 4,
            chess.BISHOP: 4,
            chess.ROOK: 2,
            chess.QUEEN: 1
        }

        for color in [True, False]:
            score = 0
            mult = 1 if color else -1
            mobility = 0

            for piece_type, weight in weights.items():
                for square in board.pieces(piece_type, color):
                    legal_moves = [move for move in board.legal_moves if move.from_square == square]
                    mobility += weight * len(legal_moves)

            score += mobility * mult

        return score

    def evaluate_key_square_control(self, board):
        key_squares = [
            chess.E4, chess.D4, chess.E5, chess.D5
        ]

        score = 0

        for square in key_squares:
            white_attackers = board.attackers(chess.WHITE, square)
            black_attackers = board.attackers(chess.BLACK, square)

            score += 5 * len(white_attackers)
            score -= 5 * len(black_attackers)

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

        book_move = self.opening_book.get_move(board)
        if book_move and book_move in board.legal_moves:
            print(f"[DEBUG] Book move played: {book_move}")
            return book_move

        polyglot_move = self.polyglot_book.get_move(board)
        if polyglot_move and polyglot_move in board.legal_moves:
            print(f"[DEBUG] Book move played (polyglot): {polyglot_move}")
            return polyglot_move

        # Otherwise fallback to search
        score, best_move = self.minimax(board, depth=3, alpha=float('-inf'), beta=float('inf'),
                                        maximizing_player=board.turn)
        print("Best move found was " + str(best_move) + " with a score of " + str(score))
        return best_move