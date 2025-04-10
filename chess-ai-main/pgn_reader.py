import chess
import chess.pgn
import chess.polyglot
import collections
import random
import time
import os
import pickle


class PGNOpeningBook:
    """A class to read PGN files and use them as an opening book with caching support"""

    def __init__(self, pgn_path, max_moves=10, min_frequency=2, max_positions=100000,
                 cache_file=None, min_elo=0):
        """
        Initialize the PGN opening book.

        Args:
            pgn_path: Path to the PGN file
            max_moves: Maximum number of moves to consider from each game
            min_frequency: Minimum number of times a position must appear to be considered
            max_positions: Maximum number of positions to store in memory
            cache_file: Path to save/load processed data (default: derived from pgn_path)
        """
        self.pgn_path = pgn_path
        self.max_moves = max_moves
        self.min_frequency = min_frequency
        self.max_positions = max_positions
        self.min_elo = min_elo

        # Default cache file name based on pgn file
        if cache_file is None:
            cache_file = os.path.splitext(pgn_path)[0] + '.openings'
        self.cache_file = cache_file

        # Dictionary to store positions and their possible moves with frequency
        self.positions = collections.defaultdict(collections.Counter)

        # Try to load from cache first
        if self.load_cache():
            print(f"Loaded opening book from cache: {self.cache_file}")
        # Otherwise load from PGN if it exists
        elif os.path.exists(pgn_path):
            self.load_pgn()
            self.save_cache()  # Save for next time
        else:
            print(f"Warning: PGN file {pgn_path} not found")

    def load_cache(self):
        """Load processed positions from cache file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.positions = data['positions']
                    return True
            return False
        except Exception as e:
            print(f"Error loading cache: {e}")
            return False

    def save_cache(self):
        """Save processed positions to cache file"""
        try:
            with open(self.cache_file, 'wb') as f:
                data = {
                    'positions': self.positions,
                    'metadata': {
                        'pgn_file': self.pgn_path,
                        'max_moves': self.max_moves,
                        'min_frequency': self.min_frequency,
                        'positions_count': len(self.positions)
                    }
                }
                pickle.dump(data, f)
            print(f"Saved opening book to cache: {self.cache_file}")
            return True
        except Exception as e:
            print(f"Error saving cache: {e}")
            return False

    def load_pgn(self):
        """Load positions from the PGN file"""
        start_time = time.time()
        positions_loaded = 0
        games_processed = 0

        print(f"Loading PGN file: {self.pgn_path}")

        try:
            pgn = open(self.pgn_path, encoding='utf-8')

            # Process games until we reach max positions or end of file
            while positions_loaded < self.max_positions:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break

                if not self._meets_elo_requirement(game):
                    continue

                # Skip games with very few moves
                if game.end().ply() < 10:
                    continue

                # Process this game
                self._process_game(game)
                games_processed += 1
                positions_loaded = len(self.positions)

                # Status update every 1000 games
                if games_processed % 1000 == 0:
                    elapsed = time.time() - start_time
                    print(f"Processed {games_processed} games, {positions_loaded} positions in {elapsed:.1f} seconds")

            pgn.close()

            # Prune infrequent positions
            self._prune_positions()

            elapsed = time.time() - start_time
            print(f"Finished loading {games_processed} games, {len(self.positions)} positions in {elapsed:.1f} seconds")

        except Exception as e:
            print(f"Error loading PGN file: {e}")

    def _process_game(self, game):
        """Extract opening moves from a single game"""
        board = game.board()

        # Get game result for weighting
        result_score = self._get_result_score(game)

        # Get player Elo ratings to weight high-level games more
        elo_bonus = self._get_elo_bonus(game)

        # Follow the main line for max_moves or until the game ends
        move_count = 0
        for move in game.mainline_moves():
            # Stop if we've reached the max number of moves to consider
            if move_count >= self.max_moves:
                break

            # Get the position's Zobrist hash before the move
            pos_hash = chess.polyglot.zobrist_hash(board)

            # Store this move as a candidate for this position
            # Weight by result and player strength
            weight = 1 + result_score + elo_bonus
            self.positions[pos_hash][move] += weight

            # Make the move on our board
            board.push(move)
            move_count += 1

    def _get_result_score(self, game):
        """Convert game result to a score bonus for weighting moves"""
        result = game.headers.get("Result", "*")

        # Game still in progress or result unknown
        if result == "*":
            return 0

        # Draw
        if result == "1/2-1/2":
            return 0.5

        # White win - bonus for white moves
        if result == "1-0":
            return 1

        # Black win - bonus for black moves
        if result == "0-1":
            return 1

        return 0

    def _get_elo_bonus(self, game):
        """Calculate a bonus based on player Elo ratings"""
        try:
            white_elo = int(game.headers.get("WhiteElo", 0))
            black_elo = int(game.headers.get("BlackElo", 0))

            # Average Elo of the players
            avg_elo = (white_elo + black_elo) / 2

            # Bonus for strong players (2400+)
            if avg_elo >= 2600:
                return 3
            elif avg_elo >= 2400:
                return 2
            elif avg_elo >= 2200:
                return 1

        except (ValueError, TypeError):
            pass

        return 0

    def _prune_positions(self):
        """Remove positions that don't appear frequently enough"""
        positions_to_remove = []

        for pos_hash, moves in self.positions.items():
            # Remove moves that don't appear frequently enough
            for move, count in list(moves.items()):
                if count < self.min_frequency:
                    del moves[move]

            # If no moves left, remove the position
            if not moves:
                positions_to_remove.append(pos_hash)

        # Remove empty positions
        for pos_hash in positions_to_remove:
            del self.positions[pos_hash]

    def get_move(self, board):
        """Get a move from the opening book for the given position"""
        pos_hash = chess.polyglot.zobrist_hash(board)

        # Check if we have this position
        if pos_hash not in self.positions:
            return None

        moves = self.positions[pos_hash]
        if not moves:
            return None

        # Filter to only legal moves (in case of hash collision)
        legal_moves = {move: freq for move, freq in moves.items()
                       if move in board.legal_moves}

        if not legal_moves:
            return None

        # Select a move, weighted by frequency
        total = sum(legal_moves.values())
        threshold = random.random() * total

        current = 0
        for move, freq in sorted(legal_moves.items(), key=lambda x: x[1], reverse=True):
            current += freq
            if current >= threshold:
                return move

        # Fallback to the most common move
        return max(legal_moves.items(), key=lambda x: x[1])[0]

    def get_weighted_moves(self, board):
        """Get all moves from the opening book with their weights"""
        pos_hash = chess.polyglot.zobrist_hash(board)

        if pos_hash not in self.positions:
            return []

        moves = self.positions[pos_hash]
        legal_moves = {move: freq for move, freq in moves.items()
                       if move in board.legal_moves}

        return sorted(legal_moves.items(), key=lambda x: x[1], reverse=True)

    def print_stats(self):
        """Print statistics about the opening book"""
        if not self.positions:
            print("Opening book is empty")
            return

        total_positions = len(self.positions)
        total_moves = sum(len(moves) for moves in self.positions.values())
        avg_moves_per_pos = total_moves / total_positions if total_positions else 0

        print(f"Opening Book Statistics:")
        print(f"  Total positions: {total_positions}")
        print(f"  Total moves: {total_moves}")
        print(f"  Average moves per position: {avg_moves_per_pos:.2f}")

        # Show distribution of move counts
        move_counts = [len(moves) for moves in self.positions.values()]
        for i in range(1, 6):
            count = sum(1 for x in move_counts if x == i)
            percent = 100 * count / total_positions if total_positions else 0
            print(f"  Positions with {i} move{'s' if i != 1 else ''}: {count} ({percent:.1f}%)")

        many_moves = sum(1 for x in move_counts if x > 5)
        percent = 100 * many_moves / total_positions if total_positions else 0
        print(f"  Positions with >5 moves: {many_moves} ({percent:.1f}%)")

    def _meets_elo_requirement(self, game):
        """Check if the game meets the minimum Elo requirement."""
        try:
            white_elo = int(game.headers.get("WhiteElo", 0))
            black_elo = int(game.headers.get("BlackElo", 0))
            avg_elo = (white_elo + black_elo) / 2
            return avg_elo >= self.min_elo
        except (ValueError, TypeError):
            return False