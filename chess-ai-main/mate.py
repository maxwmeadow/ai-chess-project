import threading
import time

import chess


class MateSolver:
    def __init__(self, chess_bot):
        self.chess_bot = chess_bot
        self.running = False
        self.thread = None
        self.mate_found = False
        self.mate_sequence = None
        self.current_board = None
        self.lock = threading.Lock()

    def start(self, board):
        """Start the mate solver thread with the current board"""
        with self.lock:
            self.current_board = board.copy()
            self.mate_found = False
            self.mate_sequence = None

        if self.thread is None or not self.thread.is_alive():
            self.running = True
            self.thread = threading.Thread(target=self._mate_search_worker)
            self.thread.daemon = True  # This makes the thread exit when main program exits
            self.thread.start()

    def stop(self):
        """Stop the mate solver thread"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.1)  # Give it a moment to finish

    def get_mate_move(self):
        """Return the first move of mating sequence if one was found"""
        with self.lock:
            if self.mate_found and self.mate_sequence:
                return self.mate_sequence[0]
            return None

    def _mate_search_worker(self):
        """Worker that continuously searches for mate sequences"""
        max_mate_depth = 10  # Look for mates up to 10 moves deep

        while self.running:
            with self.lock:
                if self.current_board is None:
                    time.sleep(0.01)
                    continue
                board = self.current_board.copy()

            # Search for mate with increasing depth
            for mate_depth in range(1, max_mate_depth + 1):
                if not self.running:
                    break

                # Try to find mate at this depth
                mate_move = self._search_mate_n(board, mate_depth)
                if mate_move:
                    with self.lock:
                        self.mate_found = True
                        self.mate_sequence = mate_move
                        break

            if self.mate_found:
                break
            time.sleep(0.01)  # Small pause between iterations to not hog CPU

    def _search_mate_n(self, board, depth):
        """Search for mate in n moves"""
        # Only consider checks and high-value captures for efficiency
        moves = []
        for move in board.legal_moves:
            board.push(move)
            is_check = board.is_check()
            board.pop()

            capture_value = 0
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                if victim:
                    capture_value = self.chess_bot.get_piece_value(victim)

            if is_check or capture_value >= self.chess_bot.piece_values[chess.BISHOP]:
                moves.append(move)

        # If no interesting moves, no mate
        if not moves:
            return None

        # --- New Part: Prioritize moves ---
        moves.sort(key=lambda move: (
            board.gives_check(move),
            board.is_capture(move),
            -self.chess_bot.get_piece_value(board.piece_at(move.to_square)) if board.is_capture(move) else 0
        ), reverse=True)
        # -----------------------------------

        # Try each move and look for forced mate
        for move in moves:
            board.push(move)

            if board.is_checkmate():
                board.pop()
                return [move]

            if board.is_stalemate() or board.is_insufficient_material():
                board.pop()
                continue

            # Recursively search for opponent's best defense
            if depth > 1:
                opponent_has_defense = False
                for defense_move in board.legal_moves:
                    board.push(defense_move)

                    if depth > 2:
                        our_mate = self._search_mate_n(board, depth - 2)
                        if not our_mate:
                            opponent_has_defense = True
                    else:
                        has_mate_in_one = False
                        for m in board.legal_moves:
                            board.push(m)
                            if board.is_checkmate():
                                has_mate_in_one = True
                            board.pop()
                            if has_mate_in_one:
                                break

                        if not has_mate_in_one:
                            opponent_has_defense = True

                    board.pop()
                    if opponent_has_defense:
                        break

                if not opponent_has_defense:
                    board.pop()
                    return [move]

            board.pop()

        return None