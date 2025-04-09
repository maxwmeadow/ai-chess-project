import chess
from chess.svg import piece
import chess.polyglot
import time

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
        # Initialize the same resources as before
        self.opening_book = OpeningBook(create_simple_opening_book(), max_book_depth=8)
        self.polyglot_book = PolyglotBook("books/gm2600.bin")

        self.transposition_table = {}
        self.transposition_table_max_size = 1000000
        self.eval_cache = {}

        self.killer_moves = {}
        self.counter_moves = {}
        self.history_table = {}

        # Rest of the initialization remains the same
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
        phase = get_game_phase(board)

        # Dynamic pawn value - increases as game progresses toward endgame
        pawn_value = self.piece_values[chess.PAWN]
        if phase < 128:  # Endgame territory
            pawn_value = self.piece_values[chess.PAWN] + (256 - phase) // 4

        for piece in self.piece_values:
            if piece == chess.PAWN:
                score += len(board.pieces(piece, True)) * pawn_value
                score -= len(board.pieces(piece, False)) * pawn_value
            else:
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
            score = 0

            # Pawn files for isolated/doubled
            pawn_files = set(chess.square_file(p) for p in pawns)

            file_counts = {}
            for pawn in pawns:
                file = chess.square_file(pawn)
                file_counts[file] = file_counts.get(file, 0) + 1

            for pawn in pawns:
                rank = chess.square_rank(pawn)
                file = chess.square_file(pawn)
                pawn_score = 0

                # Advancement bonus
                advancement = rank if color else 7 - rank
                pawn_score += advancement * 5  # Progressive bonus

                # Isolated
                adjacent_files = [file - 1, file + 1]
                if not any(adj_file in pawn_files for adj_file in adjacent_files if 0 <= adj_file < 8):
                    pawn_score -= 15

                # Doubled
                count_on_file = file_counts[file]
                if count_on_file > 1:
                    penalty = 15 * (count_on_file - 1)
                    pawn_score -= penalty

                # Passed
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
                    passed_bonus = 20 + (rank if color else 7 - rank) * 10
                    pawn_score += passed_bonus

                    # Connected passed
                    for other in pawns:
                        if other == pawn:
                            continue
                        other_file = chess.square_file(other)
                        other_rank = chess.square_rank(other)
                        if abs(other_file - file) == 1 and abs(other_rank - rank) <= 1:
                            pawn_score += 20
                            break

                score += pawn_score

            if color:
                white_score += score
            else:
                black_score += score

        total = white_score - black_score
        return total

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

                has_friendly_pawn = any(
                    board.piece_at(chess.square(file, r)) == chess.Piece(chess.PAWN, color)
                    for r in range(8)
                )
                if not has_friendly_pawn:
                    open_file_penalty += 20

            capped_shield = min(shield_score, 30)
            partial = capped_shield - open_file_penalty
            score += partial * multiplier

            # Tropism
            tropism_penalty = 0
            for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                for sq in board.pieces(piece_type, enemy_color):
                    dist = chess.square_distance(king_square, sq)
                    if dist <= 3:
                        penalty = (10 - 2 * dist)
                        tropism_penalty += penalty

            score -= tropism_penalty * multiplier

        return score

    def evaluate_position(self, board):
        # Check position cache
        board_hash = chess.polyglot.zobrist_hash(board)
        if board_hash in self.eval_cache:
            return self.eval_cache[board_hash]

        if board.is_game_over():
            if board.is_checkmate():
                return -10000 if board.turn else 10000

            return 0  # Draw

        phase = get_game_phase(board)
        material = self.evaluate_material(board)
        position = self.evaluate_piece_position(board)

        if phase < 64:
            # Opening / Early Middlegame
            pawn_structure = 0
            king_safety = self.evaluate_king_safety(board)
            mobility = self.evaluate_mobility(board)
            key_squares = self.evaluate_key_square_control(board)
            attack_score = self.evaluate_attacks(board)
            coordination = self.evaluate_piece_coordination(board)

        elif phase < 128:
            # Middlegame / Early Endgame
            pawn_structure = self.evaluate_pawn_structure(board)
            king_safety = self.evaluate_king_safety(board)
            mobility = self.evaluate_mobility(board)
            key_squares = self.evaluate_key_square_control(board)
            attack_score = self.evaluate_attacks(board)
            coordination = self.evaluate_piece_coordination(board)

        else:
            # True Endgame
            pawn_structure = self.evaluate_pawn_structure(board)
            king_safety = 0  # King activity matters more than "safety" now
            mobility = self.evaluate_mobility(board)
            key_squares = self.evaluate_key_square_control(board)
            attack_score = 0  # Attacks don't matter much when few pieces left
            coordination = self.evaluate_piece_coordination(board)

        score = (
                material +
                position +
                pawn_structure +
                king_safety +
                mobility +
                key_squares +
                attack_score +
                coordination
        )


        self.eval_cache[board_hash] = score
        return score

    def evaluate_mobility(self, board):
        weights = {
            chess.KNIGHT: 5,
            chess.BISHOP: 5,
            chess.ROOK: 3,
            chess.QUEEN: 2
        }

        total_score = 0

        for color in [True, False]:
            mult = 1 if color else -1
            score = 0

            for piece_type, weight in weights.items():
                for square in board.pieces(piece_type, color):
                    # Get all legal moves for this piece
                    mobility_count = 0
                    for move in board.legal_moves:
                        if move.from_square == square:
                            mobility_count += 1

                    score += mobility_count * weight

            total_score += score * mult

        return total_score

    def evaluate_attacks(self, board):
        score = 0

        for color in [True, False]:
            mult = 1 if color else -1
            enemy_color = not color
            enemy_king_square = board.king(enemy_color)
            if enemy_king_square is None:
                continue

            attack_score = 0

            # Count attackers and their weight
            for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                attack_weight = {
                    chess.KNIGHT: 10,
                    chess.BISHOP: 10,
                    chess.ROOK: 15,
                    chess.QUEEN: 20
                }

                for square in board.pieces(piece_type, color):
                    # Check if this piece is attacking the king's vicinity
                    king_vicinity = [
                        s for s in chess.SQUARES
                        if chess.square_distance(s, enemy_king_square) <= 2
                    ]

                    for vicinity_square in king_vicinity:
                        if board.is_attacked_by(color, vicinity_square):
                            attack_score += attack_weight[piece_type]
                            break

            # Check if king is in check
            temp_board = board.copy()
            temp_board.turn = color
            if temp_board.is_check():
                attack_score += 50

            score += attack_score * mult

        return score

    def evaluate_piece_coordination(self, board):
        score = 0

        for color in [True, False]:
            mult = 1 if color else -1
            coordination_score = 0

            # Knights supporting each other
            knight_squares = list(board.pieces(chess.KNIGHT, color))
            for i, sq1 in enumerate(knight_squares):
                for sq2 in knight_squares[i + 1:]:
                    if chess.square_distance(sq1, sq2) <= 2:
                        coordination_score += 5

            # Bishops on same diagonal or adjacent diagonals
            bishop_squares = list(board.pieces(chess.BISHOP, color))
            for i, sq1 in enumerate(bishop_squares):
                for sq2 in bishop_squares[i + 1:]:
                    if (chess.square_rank(sq1) + chess.square_file(sq1) ==
                            chess.square_rank(sq2) + chess.square_file(sq2) or
                            chess.square_rank(sq1) - chess.square_file(sq1) ==
                            chess.square_rank(sq2) - chess.square_file(sq2)):
                        coordination_score += 10

            # Rooks on same file or rank
            rook_squares = list(board.pieces(chess.ROOK, color))
            for i, sq1 in enumerate(rook_squares):
                for sq2 in rook_squares[i + 1:]:
                    if (chess.square_file(sq1) == chess.square_file(sq2) or
                            chess.square_rank(sq1) == chess.square_rank(sq2)):
                        coordination_score += 15

            score += coordination_score * mult

        return score

    def evaluate_key_square_control(self, board):
        key_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        score = 0

        for square in key_squares:
            white_attackers = board.attackers(chess.WHITE, square)
            black_attackers = board.attackers(chess.BLACK, square)
            score += 5 * len(white_attackers)
            score -= 5 * len(black_attackers)

        return score

    def minimax(self, board, depth, alpha, beta, maximizing_player, ply=0):
        """Legacy minimax implementation - use PVS instead for better performance"""
        return self.pvs(board, depth, alpha, beta, maximizing_player, ply)

    def get_move(self, board: ChessBoard):
        """Main method to select the best move."""
        start_time = time.time()
        time_limit = 5
        time_for_move = min(time_limit * 0.9, time_limit - 0.1)

        if isinstance(board, ChessBoard):
            board = board.get_board_state()

        mate_move = self.check_for_immediate_mate(board)
        if mate_move:
            print("Found mate in 1!")
            return mate_move

        # Try book moves first
        book_move = self.opening_book.get_move(board)
        if book_move and book_move in board.legal_moves:
            print(f"[DEBUG] Book move played: {book_move}")
            return book_move

        polyglot_move = self.polyglot_book.get_move(board)
        if polyglot_move and polyglot_move in board.legal_moves:
            print(f"[DEBUG] Book move played (polyglot): {polyglot_move}")
            return polyglot_move

        # Reset tables for new search
        self.transposition_table = {}
        self.killer_moves = {}

        best_move = None
        max_depth = 15

        # Iterative deepening
        for depth in range(1, max_depth + 1):
            if time.time() - start_time > time_for_move:
                break

            score, move = self.minimax(board, depth, float('-inf'), float('inf'), 1 if board.turn else -1, 0)

            if move:
                best_move = move
                elapsed = time.time() - start_time
                print(f"Depth {depth} completed in {elapsed:.2f}s: {best_move} with score {score}")

            # Early exit conditions
            if abs(score) > 9000 and depth > 3:  # Mate found
                break
            if score > 300 and depth > 5 and (time.time() - start_time) > (time_for_move * 0.5):
                break

        # Fallback if no move found
        if not best_move and list(board.legal_moves):
            best_move = list(board.legal_moves)[0]

        return best_move


    def order_moves(self, board, quiescence=False):
        """Order moves for better alpha-beta pruning efficiency"""
        moves = list(board.legal_moves)
        scored_moves = []

        ply = len(board.move_stack)
        killer_move = self.killer_moves.get(ply)
        previous_move = board.move_stack[-1] if board.move_stack else None
        counter_move = None

        # Get counter move if available
        if previous_move:
            counter_key = (previous_move.from_square, previous_move.to_square)
            counter_move = self.counter_moves.get(counter_key)

        for move in moves:
            score = 0

            # PV move from transposition table gets highest priority
            board_hash = chess.polyglot.zobrist_hash(board)
            if board_hash in self.transposition_table:
                _, _, hash_move, _ = self.transposition_table[board_hash]
                if hash_move == move:
                    score += 10000

            # Counter move bonus
            if counter_move == move:
                score += 8500  # Between PV and killer moves

            # Castling bonus (for early/middle game)
            if board.is_castling(move):
                score += 500

            piece = board.piece_at(move.from_square)
            to_rank = chess.square_rank(move.to_square)
            if piece:
                if piece.piece_type in [chess.ROOK, chess.QUEEN]:
                    if (piece.color and to_rank == 6) or (not piece.color and to_rank == 1):
                        score += 50

            # Killer move
            if killer_move == move:
                score += 9000

            # History heuristic
            if (board.turn, move.from_square, move.to_square) in self.history_table:
                score += self.history_table[(board.turn, move.from_square, move.to_square)]

            # Prioritize captures by MVV-LVA (Most Valuable Victim - Least Valuable Aggressor)
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                aggressor = board.piece_at(move.from_square)

                if victim and aggressor:
                    victim_value = self.get_piece_value(victim)
                    aggressor_value = self.get_piece_value(aggressor)

                    # More refined MVV-LVA scoring
                    if victim_value >= aggressor_value:
                        score += 8000 + 10 * victim_value - aggressor_value
                    else:
                        # Potentially bad captures score lower than non-captures
                        score += 7000 + 10 * victim_value - aggressor_value

                if board.is_en_passant(move):
                    score += 100

            # Promotions
            if move.promotion:
                score += 7000 + self.get_piece_value(chess.Piece(move.promotion, True)) - 100

            # Checks
            if not quiescence:
                board.push(move)
                if board.is_check():
                    score += 30
                board.pop()

            # Center control
            piece = board.piece_at(move.from_square)
            if piece and piece.piece_type == chess.PAWN:
                if move.to_square in [chess.E4, chess.D4, chess.E5, chess.D5]:
                    score += 20

            # Use move string as stable tiebreaker
            scored_moves.append((score, str(move), move))

        # Sort moves by score in descending order
        scored_moves.sort(reverse=True)
        return [move for _, _, move in scored_moves]

    def get_piece_value(self, piece):
        """Get the value of a piece for move ordering"""
        if piece is None:
            return 0
        return self.piece_values.get(piece.piece_type, 0)

    def quiescence(self, board, alpha, beta, color, max_depth=3, current_depth=0):
        """Simplified quiescence search."""
        stand_pat = self.evaluate_position(board) * color

        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        if current_depth >= max_depth:
            return stand_pat

        captures = [move for move in board.legal_moves if board.is_capture(move)]
        captures = self.order_captures(board, captures)

        for move in captures:
            board.push(move)
            score = -self.quiescence(board, -beta, -alpha, -color, max_depth, current_depth + 1)
            board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    def order_captures(self, board, moves):
        """Simple ordering for captures in quiescence search."""
        scored_moves = []

        for move in moves:
            score = 0
            victim = board.piece_at(move.to_square)
            aggressor = board.piece_at(move.from_square)

            if victim and aggressor:
                victim_value = self.get_piece_value(victim)
                aggressor_value = self.get_piece_value(aggressor)
                score = 10 * victim_value - aggressor_value

            # Add the move string as a stable secondary sort key
            scored_moves.append((score, str(move), move))

        # Sort by score first, then by move string for stability
        scored_moves.sort(reverse=True)
        return [move for _, _, move in scored_moves]

    def update_history_heuristic(self, board, move, depth):
        """Update history table for successful moves"""
        self.history_table[(board.turn, move.from_square, move.to_square)] = \
            self.history_table.get((board.turn, move.from_square, move.to_square), 0) + depth * depth

    def update_killer_move(self, move, ply):
        """Update killer move table"""
        self.killer_moves[ply] = move

    def is_likely_good_capture(self, board, move):
        """Static Exchange Evaluation (SEE) approximation - determines if a capture is likely good"""
        if not board.is_capture(move):
            return True  # Not a capture, no need to evaluate

        victim = board.piece_at(move.to_square)
        aggressor = board.piece_at(move.from_square)

        if not victim or not aggressor:
            return True  # Edge case

        victim_value = self.get_piece_value(victim)
        aggressor_value = self.get_piece_value(aggressor)

        # En passant is always good
        if board.is_en_passant(move):
            return True

        # Simple MVV-LVA (Most Valuable Victim - Least Valuable Aggressor)
        if victim_value >= aggressor_value:
            return True  # Capturing equal or higher value piece

        # For lower value captures, check if the square is defended
        board.push(move)
        recapture_possible = bool(board.attackers(not board.turn, move.to_square))
        board.pop()

        # If victim is worth less and square is defended, likely a bad capture
        if recapture_possible:
            return False

        return True  # No recapture possible, so capture is probably good

    def is_endgame(self, board):
        return (len(board.pieces(chess.QUEEN, chess.WHITE)) +
                len(board.pieces(chess.QUEEN, chess.BLACK)) == 0 or
                get_game_phase(board) < 100)

    def manage_transposition_table(self):
        if len(self.transposition_table) > self.transposition_table_max_size:
            # Sort entries by depth (preserve deeper searches)
            entries = [(key, value[0]) for key, value in self.transposition_table.items()]
            entries.sort(key=lambda x: x[1])  # Sort by depth

            # Remove 25% of shallowest depth entries
            keys_to_remove = [entry[0] for entry in entries[:len(entries) // 4]]
            for key in keys_to_remove:
                del self.transposition_table[key]

    def pvs(self, board, depth, alpha, beta, color, ply=0):
        """Principal Variation Search (PVS) with alpha-beta pruning."""
        if board.is_game_over():
            if board.is_checkmate():
                return -10000 * color + ply * color, None
            return 0, None  # Draw

        # Transposition table lookup
        board_hash = chess.polyglot.zobrist_hash(board)
        if board_hash in self.transposition_table:
            entry_depth, entry_score, entry_move, entry_type = self.transposition_table[board_hash]
            if entry_depth >= depth:
                if entry_type == "exact":
                    return entry_score * color, entry_move

        if depth == 0:
            return self.quiescence(board, alpha, beta, color), None

        # Null move pruning with more conditions
        if depth >= 3 and not board.is_check() and not self.is_endgame(board):
            # Don't do null move if we have only major pieces left
            if (len(board.pieces(chess.QUEEN, board.turn)) +
                    len(board.pieces(chess.ROOK, board.turn)) > 0):

                null_move = chess.Move.null()
                board.push(null_move)

                # Dynamic reduction based on depth
                R = 2
                if depth > 6: R = 3

                null_score, _ = self.pvs(board, depth - 1 - R, -beta, -beta + 1, -color, ply + 1)
                null_score = -null_score

                board.pop()

                if null_score >= beta:
                    # Verification search with reduced depth
                    verify_depth = depth - R
                    if verify_depth > 0:
                        verify_score, _ = self.pvs(board, verify_depth, beta - 1, beta, color, ply)
                        if verify_score >= beta:
                            return beta, None
                    else:
                        return beta, None

        best_move = None
        moves = self.order_moves(board)

        first_move = True
        original_alpha = alpha

        for move_idx, move in enumerate(moves):
            board.push(move)

            reduction = 0

            if not first_move and depth >= 3 and move_idx >= 3 and not board.is_check():
                # Only reduce bad-looking moves after a few good ones
                reduction = 1

            new_depth = depth - 1 - reduction

            if first_move or reduction == 0:
                score, _ = self.pvs(board, new_depth, -beta, -alpha, -color, ply + 1)
                score = -score
                first_move = False
            else:
                # Reduced-depth null window search first
                score, _ = self.pvs(board, new_depth, -alpha - 1, -alpha, -color, ply + 1)
                score = -score

                # If score improved alpha, full re-search at normal depth
                if alpha < score < beta:
                    score, _ = self.pvs(board, depth - 1, -beta, -score, -color, ply + 1)
                    score = -score

            board.pop()

            if score > alpha:
                alpha = score
                best_move = move

            if alpha >= beta:
                self.update_killer_move(move, ply)
                break

            if score > alpha:
                alpha = score
                best_move = move

            if alpha >= beta:
                # Beta cutoff, update killer move
                self.update_killer_move(move, ply)
                break

        # Store in transposition table
        entry_type = "exact"
        if alpha <= original_alpha:
            entry_type = "upperbound"
        elif alpha >= beta:
            entry_type = "lowerbound"

        self.transposition_table[board_hash] = (depth, alpha * color, best_move, entry_type)

        return alpha, best_move

    def check_for_immediate_mate(self, board):
        """Check if there's a mate in 1 move"""
        for move in board.legal_moves:
            board.push(move)
            is_mate = board.is_checkmate()
            board.pop()
            if is_mate:
                return move
        return None

    def check_for_blunders(self, board, candidate_move, depth=2):
        """
        Check if a candidate move is a blunder by examining if it:
        1. Hangs a piece (can be captured without compensation)
        2. Walks into a fork or pin
        3. Misses a simple tactic

        Returns:
        - True if the move is likely a blunder
        - False if the move seems safe
        """
        if not candidate_move or not board.is_legal(candidate_move):
            return True  # Invalid move is considered a blunder

        original_board = board.copy()
        board = board.copy()  # Work with a copy to avoid modifying the original

        # Execute the candidate move
        board.push(candidate_move)

        # 1. Check if we just hung a piece (SEE - Static Exchange Evaluation)
        piece_moved = original_board.piece_at(candidate_move.from_square)
        to_square = candidate_move.to_square

        if piece_moved and board.is_attacked_by(not board.turn, to_square):
            # Our piece can be captured - check if it's adequately defended
            attacker_value = float('inf')
            for attacker_square in board.attackers(not board.turn, to_square):
                attacker = board.piece_at(attacker_square)
                if attacker:
                    attacker_value = min(attacker_value, self.get_piece_value(attacker))

            # Check if we can recapture
            can_recapture = False
            recapture_value = 0
            for defender_square in board.attackers(board.turn, to_square):
                defender = board.piece_at(defender_square)
                if defender:
                    can_recapture = True
                    recapture_value = max(recapture_value, self.get_piece_value(defender))

            # If we can't recapture or losing significant material, likely a blunder
            piece_value = self.get_piece_value(piece_moved)
            if not can_recapture and piece_value > 100:  # Allow pawn sacrifice
                return True
            if can_recapture and (piece_value - attacker_value + recapture_value) < -200:
                # Net material loss after exchange
                return True

        # 2. Check if our king is in check and we don't have good responses
        if board.is_check():
            legal_moves = list(board.legal_moves)
            if not legal_moves:  # Checkmate
                return True
            if len(legal_moves) == 1:  # Forced move, but not necessarily bad
                # Check if the forced move loses material
                check_board = board.copy()
                check_board.push(legal_moves[0])
                if self.evaluate_material(check_board) < self.evaluate_material(original_board) - 200:
                    return True

        # 3. Do a shallow search to see if the position after our move is significantly worse
        current_eval = self.evaluate_position(original_board)

        # Use negamax/PVS with reduced depth to check tactical sequences
        # We're looking for significant immediate drops in evaluation
        next_color = 1 if board.turn else -1
        tactical_score, _ = self.pvs(board, depth, float('-inf'), float('inf'), next_color, 0)
        tactical_score = tactical_score * -1  # Because we're now viewing from opponent's perspective

        # Compare evaluations: if there's a significant drop, it might be a blunder
        eval_drop = current_eval - tactical_score

        # Threshold based on piece values - loss of knight or more is significant
        return eval_drop > 300  # Consider bishop/knight value as threshold for blunder