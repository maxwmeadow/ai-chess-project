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

        self.opening_book = OpeningBook(create_simple_opening_book(), max_book_depth = 8)
        self.polyglot_book = PolyglotBook("books/gm2600.bin")

        self.transposition_table = {}
        self.transposition_table_max_size = 1000000
        self.eval_cache = {}

        self.killer_moves = {}
        self.counter_moves = {}

        self.history_table = {}

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

            #Tropism
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
        #Check position cache
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

        if abs(material) < 500 or phase > 128:
            pawn_structure = self.evaluate_pawn_structure(board)
            king_safety = self.evaluate_king_safety(board)
            mobility = self.evaluate_mobility(board)
            key_squares = self.evaluate_key_square_control(board)
        else:
            pawn_structure = king_safety = mobility = key_squares = 0

        score = (
            material +
            position +
            pawn_structure +
            king_safety +
            mobility +
            key_squares
        )

        self.eval_cache[board_hash] = score
        return score

    def evaluate_mobility(self, board):
        weights = {
            chess.KNIGHT: 4,
            chess.BISHOP: 4,
            chess.ROOK: 2,
            chess.QUEEN: 1
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

        """

        Main method to select the best move.

        """
        start_time = time.time()
        time_limit = 5

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

        self.transposition_table = {}
        self.killer_moves = {}

        best_move = None
        best_score = 0
        max_depth = 15

        # Initial full-width search for first few depths
        for depth in range(1, 3):
            if time.time() - start_time > time_limit:
                break
            best_score, best_move = self.pvs(board, depth, float('-inf'), float('inf'), board.turn)

            if best_move:
                elapsed = time.time() - start_time
                print(f"Depth {depth} completed in {elapsed:.2f}s: {best_move} with score {best_score}")

        # Use aspiration windows for deeper searches
        window = 50  # Initial window size
        for depth in range(3, max_depth + 1):
            if time.time() - start_time > time_limit:
                break

            alpha = best_score - window
            beta = best_score + window

            tries = 0
            research_needed = True

            while research_needed and tries < 3:  # Try widening window up to 3 times
                score, move = self.pvs(board, depth, alpha, beta, board.turn)

                if score <= alpha:  # Failed low, research with wider window
                    alpha = max(alpha - window * (tries + 1), -10000)
                    window *= 2
                    tries += 1
                    continue

                if score >= beta:  # Failed high, research with wider window
                    beta = min(beta + window * (tries + 1), 10000)
                    window *= 2
                    tries += 1
                    continue

                # Success - score within window
                best_score = score
                if move:
                    best_move = move
                research_needed = False

            elapsed = time.time() - start_time
            print(f"Depth {depth} completed in {elapsed:.2f}s: {best_move} with score {best_score}")

            # If we found a mate or nearly out of time, stop searching
            if abs(best_score) > 9000 or (time.time() - start_time) > (time_limit * 0.8):
                break

        # If no move was found, use the first legal move
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

            #Killer move
            if killer_move == move:
                score += 9000

            #History heuristic
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

            #Promotions
            if move.promotion:
                score += 7000 + self.get_piece_value(chess.Piece(move.promotion,True)) - 100

            # Checks
            if not quiescence:
                board.push(move)
                if board.is_check():
                    score += 500
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

    def quiescence(self, board, alpha, beta, depth=0, max_depth=5):
        """Quiescence search to handle tactical positions"""
        stand_pat = self.evaluate_position(board)

        if depth >= max_depth:
            return stand_pat

        if stand_pat >= beta:
            return beta

        if alpha < stand_pat:
            alpha = stand_pat

        # Only consider captures
        for move in self.order_moves(board, quiescence=True):
            if not board.is_capture(move):
                continue

            # Delta pruning - skip likely bad captures
            if not self.is_likely_good_capture(board, move):
                continue

            board.push(move)
            score = -self.quiescence(board, -beta, -alpha, depth + 1, max_depth)
            board.pop()

            if score >= beta:
                return beta

            if score > alpha:
                alpha = score

        return alpha

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

    def pvs(self, board, depth, alpha, beta, maximizing_player, ply=0):
        """Principal Variation Search - more efficient than standard minimax"""

        # Check for repetitions and fifty-move rule
        if board.is_repetition(2) or board.halfmove_clock >= 100:
            return 0, None

        # Transposition table lookup
        board_hash = chess.polyglot.zobrist_hash(board)
        if board_hash in self.transposition_table:
            stored_depth, stored_value, stored_move, flag = self.transposition_table.get(board_hash, (0, 0, None, 0))
            if stored_depth >= depth:
                if flag == 0:  # Exact score
                    return stored_value, stored_move
                elif flag == 1 and stored_value <= alpha:  # Upper bound
                    return alpha, stored_move
                elif flag == 2 and stored_value >= beta:  # Lower bound
                    return beta, stored_move

        if depth == 0:
            return self.quiescence(board, alpha, beta), None

        if board.is_game_over():
            if board.is_checkmate():
                return (-10000 + ply if board.turn else 10000 - ply), None  # Prefer shorter mates
            return 0, None  # Draw

        alpha_orig = alpha  # For transposition table flag
        best_move = None

        # Internal Iterative Deepening
        if depth >= 4 and best_move is None:
            _, iid_move = self.pvs(board, depth // 2, alpha, beta, maximizing_player, ply)
            if iid_move:
                self.transposition_table[board_hash] = (0, 0, iid_move, 0)

        # Null move pruning
        if depth >= 3 and not board.is_check() and not self.is_endgame(board):
            R = 3 + depth // 6

            board.push(chess.Move.null())
            score, _ = self.pvs(board, depth - 1 - R, -beta, -beta + 1, not maximizing_player, ply + 1)
            score = -score
            board.pop()

            if score >= beta:
                return beta, None

        # Futility pruning
        if depth <= 2 and not board.is_check():
            static_eval = self.evaluate_position(board)
            futility_margin = 100 * depth
            if maximizing_player and static_eval + futility_margin <= alpha:
                return static_eval, None
            elif not maximizing_player and static_eval - futility_margin >= beta:
                return static_eval, None

        moves = self.order_moves(board)
        if not moves:
            return self.evaluate_position(board), None

        best_score = float('-inf') if maximizing_player else float('inf')

        # First move (full window search)
        first_move = moves[0]
        board.push(first_move)

        if maximizing_player:
            score, _ = self.pvs(board, depth - 1, -beta, -alpha, False, ply + 1)
            score = -score
            board.pop()

            if score > best_score:
                best_score = score
                best_move = first_move

            if score > alpha:
                alpha = score

            if alpha >= beta:
                self.update_killer_move(first_move, ply)
                self.update_history_heuristic(board, first_move, depth)

                # --- FIX: Update counter move here ---
                if len(board.move_stack) > 0:
                    prev_move = board.move_stack[-1]
                    counter_key = (prev_move.from_square, prev_move.to_square)
                    self.counter_moves[counter_key] = first_move

                self.transposition_table[board_hash] = (depth, best_score, best_move, 2)  # Lower bound
                return best_score, best_move
        else:
            score, _ = self.pvs(board, depth - 1, -beta, -alpha, True, ply + 1)
            score = -score
            board.pop()

            if score < best_score:
                best_score = score
                best_move = first_move

            if score < beta:
                beta = score

            if alpha >= beta:
                self.update_killer_move(first_move, ply)
                self.update_history_heuristic(board, first_move, depth)

                # --- FIX: Update counter move here ---
                if len(board.move_stack) > 0:
                    prev_move = board.move_stack[-1]
                    counter_key = (prev_move.from_square, prev_move.to_square)
                    self.counter_moves[counter_key] = first_move

                self.transposition_table[board_hash] = (depth, best_score, best_move, 1)  # Upper bound
                return best_score, best_move

        # Rest of moves with LMR and null window
        for i, move in enumerate(moves[1:], 1):
            board.push(move)

            do_full_search = True
            if i >= 4 and depth >= 3 and not board.is_check() and not board.is_capture(move) and move.promotion is None:
                R = 1 + depth // 3 + min(i // 6, 3)
                score, _ = self.pvs(board, depth - 1 - R, -alpha - 1, -alpha, not maximizing_player, ply + 1)
                score = -score
                do_full_search = (score > alpha)

            if do_full_search:
                score, _ = self.pvs(board, depth - 1, -alpha - 1, -alpha, not maximizing_player, ply + 1)
                score = -score

                if alpha < score < beta:
                    score, _ = self.pvs(board, depth - 1, -beta, -alpha, not maximizing_player, ply + 1)
                    score = -score

            board.pop()

            if maximizing_player:
                if score > best_score:
                    best_score = score
                    best_move = move

                if score > alpha:
                    alpha = score

                if alpha >= beta:
                    self.update_killer_move(move, ply)
                    self.update_history_heuristic(board, move, depth)

                    # --- FIX: Update counter move here ---
                    if len(board.move_stack) > 0:
                        prev_move = board.move_stack[-1]
                        counter_key = (prev_move.from_square, prev_move.to_square)
                        self.counter_moves[counter_key] = move

                    break
            else:
                if score < best_score:
                    best_score = score
                    best_move = move

                if score < beta:
                    beta = score

                if alpha >= beta:
                    self.update_killer_move(move, ply)
                    self.update_history_heuristic(board, move, depth)

                    if len(board.move_stack) > 0:
                        prev_move = board.move_stack[-1]
                        counter_key = (prev_move.from_square, prev_move.to_square)
                        self.counter_moves[counter_key] = move

                    break

        # Store position in transposition table
        if best_score <= alpha_orig:
            flag = 1  # Upper bound
        elif best_score >= beta:
            flag = 2  # Lower bound
        else:
            flag = 0  # Exact score

        self.transposition_table[board_hash] = (depth, best_score, best_move, flag)
        return best_score, best_move

    def manage_transposition_table(self):
        if len(self.transposition_table) > self.transposition_table_max_size:
            # Simple approach: clear half the table
            keys = list(self.transposition_table.keys())
            for key in keys[:len(keys) // 2]:
                del self.transposition_table[key]