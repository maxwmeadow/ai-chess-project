import time
import random
import json
import copy
import chess
from board import ChessBoard
from bot import ChessBot
import os
import traceback
import signal
import sys

# Keep debug flag
DEBUG = True


def debug_print(message):
    if DEBUG:
        print(message)


def save_bot_to_json(bot, filename):
    piece_name_map = {
        1: "PAWN",
        2: "KNIGHT",
        3: "BISHOP",
        4: "ROOK",
        5: "QUEEN",
        6: "KING"
    }

    data = {
        "piece_values": {piece_name_map[piece_type]: value for piece_type, value in bot.piece_values.items()},
        "position_scores": {piece_name_map[piece_type]: table for piece_type, table in bot.position_scores.items()},
        "pawn_weights": bot.pawn_weights,
        "king_safety_weights": bot.king_safety_weights,
        "mobility_weights": bot.mobility_weights,
        "attack_weights": bot.attack_weights,
        "coordination_weights": bot.coordination_weights,
        "piece_combo_weights": bot.piece_combo_weights,
        "phase_weights": bot.phase_weights,
        "center_control_weight": bot.center_control_weight,
        "threat_weights": bot.threat_weights,
        "eval_weights": bot.eval_weights
    }

    debug_print(f"Writing to {filename}")
    # Use a temporary file for atomic saving to avoid corruption on interruption
    temp_filename = filename + ".tmp"
    with open(temp_filename, 'w') as f:
        json.dump(data, f, indent=4)

    # Rename to the actual filename (atomic operation on most systems)
    os.replace(temp_filename, filename)
    debug_print(f"Write complete")


def load_bot_from_json(bot, filename):
    debug_print(f"Reading from {filename}")
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        debug_print(f"Read complete")
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return False
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {filename}.")
        return False

    # Process data
    reverse_piece_map = {name: getattr(chess, name) for name in data['piece_values']}

    for name, value in data['piece_values'].items():
        bot.piece_values[reverse_piece_map[name]] = value

    for name, table in data['position_scores'].items():
        bot.position_scores[reverse_piece_map[name]] = table

    # Load the new weights if they exist in the file
    if 'pawn_weights' in data:
        bot.pawn_weights = data['pawn_weights']

    if 'king_safety_weights' in data:
        bot.king_safety_weights = data['king_safety_weights']

    if 'mobility_weights' in data:
        bot.mobility_weights = data['mobility_weights']

    if 'attack_weights' in data:
        bot.attack_weights = data['attack_weights']

    if 'coordination_weights' in data:
        bot.coordination_weights = data['coordination_weights']

    if 'piece_combo_weights' in data:
        bot.piece_combo_weights = data['piece_combo_weights']

    if 'phase_weights' in data:
        bot.phase_weights = data['phase_weights']

    if 'center_control_weight' in data:
        bot.center_control_weight = data['center_control_weight']

    if 'threat_weights' in data:
        bot.threat_weights = data['threat_weights']

    if 'eval_weights' in data:
        bot.eval_weights = data['eval_weights']

    return True


class SelfPlayTrainer:
    def __init__(self, bot_constructor, games_per_iteration=10, iterations=3, trainer_id=None):
        self.bot_constructor = bot_constructor
        self.games_per_iteration = games_per_iteration
        self.iterations = iterations
        self.best_bot = bot_constructor()
        self.trainer_id = trainer_id or f"trainer-{random.randint(1000, 9999)}"
        self.performance_history = []
        self.best_bot_filename = "best_bot.json"
        self.iterations_completed = 0
        self.is_running = True

        # Register signal handlers for graceful termination
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

        try:
            success = load_bot_from_json(self.best_bot, self.best_bot_filename)
            if success:
                print(f"Loaded existing {self.best_bot_filename}!")
            else:
                print(f"Failed to load {self.best_bot_filename}, starting fresh.")
                # Save the initial bot if no file exists
                save_bot_to_json(self.best_bot, self.best_bot_filename)
        except Exception as e:
            print(f"Error loading {self.best_bot_filename}: {e}, starting fresh.")
            traceback.print_exc()
            # Save the initial bot as fallback
            save_bot_to_json(self.best_bot, self.best_bot_filename)

    def _handle_interrupt(self, sig, frame):
        """Handle interruption gracefully"""
        print("\nTraining interrupted. Saving current progress...")
        self.is_running = False
        self._save_progress()
        print(f"Progress saved. Best bot stored in {self.best_bot_filename}.")
        print(f"Completed {self.iterations_completed} iterations.")
        sys.exit(0)

    def _save_progress(self):
        """Save the performance history"""
        history_filename = f"history_{self.trainer_id}.json"
        try:
            with open(history_filename, 'w') as f:
                json.dump(self.performance_history, f, indent=4)
            print(f"Performance history saved to {history_filename}")
        except Exception as e:
            print(f"Failed to save performance history: {e}")
            traceback.print_exc()

    def train(self):
        for i in range(self.iterations):
            if not self.is_running:
                break

            self.iterations_completed = i
            print(f"--- Iteration {i + 1}/{self.iterations} ---")

            try:
                # Create challenger from best bot
                print(f"Creating challenger")
                challenger = self.create_challenger()

                print(f"Playing match")
                results = self.play_match(self.best_bot, challenger)

                print(f"Analyzing results")
                analysis = self.analyze_results(results)
                print(f"Results: {analysis}")

                # Save iteration results to history whether we update the best bot or not
                self.performance_history.append({
                    "iteration": i + 1,
                    "timestamp": time.time(),
                    "trainer_id": self.trainer_id,
                    "metrics": analysis
                })

                # Periodically save history during training
                if (i + 1) % 5 == 0 or i == self.iterations - 1:
                    self._save_progress()

                if self.should_update_best_bot(analysis):
                    print(f"Challenger is better! Updating {self.best_bot_filename}.")
                    self.update_best_bot(challenger)
                    save_bot_to_json(self.best_bot, self.best_bot_filename)
                else:
                    print(f"Challenger not better than current best bot. No update needed.")

            except Exception as e:
                print(f"Error during iteration {i + 1}: {e}")
                traceback.print_exc()

                # Still save the error to history
                self.performance_history.append({
                    "iteration": i + 1,
                    "timestamp": time.time(),
                    "trainer_id": self.trainer_id,
                    "metrics": {
                        "error": str(e)
                    }
                })

        # Final save of performance history at the end
        self.iterations_completed = self.iterations
        self._save_progress()
        return self.best_bot

    def create_challenger(self):
        debug_print(f"Creating challenger bot")
        challenger = self.bot_constructor()

        # Copy the best bot's weights to start
        challenger.piece_values = copy.deepcopy(self.best_bot.piece_values)
        challenger.position_scores = copy.deepcopy(self.best_bot.position_scores)
        challenger.pawn_weights = copy.deepcopy(self.best_bot.pawn_weights)
        challenger.king_safety_weights = copy.deepcopy(self.best_bot.king_safety_weights)
        challenger.mobility_weights = copy.deepcopy(self.best_bot.mobility_weights)
        challenger.attack_weights = copy.deepcopy(self.best_bot.attack_weights)
        challenger.coordination_weights = copy.deepcopy(self.best_bot.coordination_weights)
        challenger.piece_combo_weights = copy.deepcopy(self.best_bot.piece_combo_weights)
        challenger.phase_weights = copy.deepcopy(self.best_bot.phase_weights)
        challenger.center_control_weight = self.best_bot.center_control_weight
        challenger.threat_weights = copy.deepcopy(self.best_bot.threat_weights)
        challenger.eval_weights = copy.deepcopy(self.best_bot.eval_weights)

        # Mutate piece values
        for piece in challenger.piece_values:
            challenger.piece_values[piece] *= random.uniform(0.95, 1.05)

        # Mutate piece-square tables
        for piece in challenger.position_scores:
            for i in range(len(challenger.position_scores[piece])):
                challenger.position_scores[piece][i] *= random.uniform(0.95, 1.05)

        # Mutate all the new weights
        for key in challenger.pawn_weights:
            challenger.pawn_weights[key] *= random.uniform(0.95, 1.05)

        for key in challenger.king_safety_weights:
            challenger.king_safety_weights[key] *= random.uniform(0.95, 1.05)

        for key in challenger.mobility_weights:
            challenger.mobility_weights[key] *= random.uniform(0.95, 1.05)

        for key in challenger.attack_weights:
            challenger.attack_weights[key] *= random.uniform(0.95, 1.05)

        for key in challenger.coordination_weights:
            challenger.coordination_weights[key] *= random.uniform(0.95, 1.05)

        for key in challenger.piece_combo_weights:
            challenger.piece_combo_weights[key] *= random.uniform(0.95, 1.05)

        for key in challenger.phase_weights:
            challenger.phase_weights[key] *= random.uniform(0.95, 1.05)

        challenger.center_control_weight *= random.uniform(0.95, 1.05)

        for key in challenger.threat_weights:
            challenger.threat_weights[key] *= random.uniform(0.95, 1.05)

        for key in challenger.eval_weights:
            challenger.eval_weights[key] *= random.uniform(0.95, 1.05)

        debug_print(f"Challenger created successfully")
        return challenger

    def play_match(self, bot1, bot2):
        debug_print(f"Starting match between bots")
        results = []
        for game_num in range(self.games_per_iteration):
            if not self.is_running:
                break

            print(f"Starting game {game_num + 1}/{self.games_per_iteration}")
            board = ChessBoard()
            white, black = (bot1, bot2) if game_num % 2 == 0 else (bot2, bot1)
            color_map = {1: "bot1", -1: "bot2", 0: "draw"} if game_num % 2 == 0 else {-1: "bot1", 1: "bot2", 0: "draw"}

            try:
                result_code = self.play_game(board, white, black, game_num)
                results.append({
                    "game_num": game_num,
                    "winner": color_map[result_code],
                    "result_code": result_code
                })
                print(f"Game {game_num + 1} complete: {color_map[result_code]} wins")
            except Exception as e:
                print(f"Error in game {game_num}: {e}")
                traceback.print_exc()
                # Add a draw in case of error to keep things moving
                results.append({
                    "game_num": game_num,
                    "winner": "draw",
                    "result_code": 0,
                    "error": str(e)
                })

        return results

    def play_game(self, board, white_bot, black_bot, game_num):
        print(f"Game {game_num + 1}: Starting new game")
        move_count = 0
        max_moves = 80  # Limit moves to prevent infinite games

        for ply in range(max_moves):
            if not self.is_running or board.is_game_over():
                if board.is_game_over():
                    print(f"Game {game_num + 1}: Game over at move {ply}")
                break

            bot = white_bot if board.get_board_state().turn == chess.WHITE else black_bot
            bot_name = "white" if board.get_board_state().turn == chess.WHITE else "black"

            print(f"Game {game_num + 1}: Getting move for {bot_name} at ply {ply}")

            try:
                move_start_time = time.time()
                move = bot.get_move(board)
                move_time = time.time() - move_start_time

                if move is None:
                    print(f"Game {game_num + 1}: {bot_name} returned None move")
                    break

                print(f"Game {game_num + 1}: Move found in {move_time:.2f}s - {move}")
                board.make_move(move)
                move_count += 1

                if ply % 5 == 0:  # Reduce frequency of updates
                    print(f"Game {game_num + 1}: Move {ply} made ({move})")

            except Exception as e:
                print(f"Game {game_num + 1}: Error making move at ply {ply}: {e}")
                traceback.print_exc()
                break

        # Check game result
        if board.get_board_state().is_checkmate():
            winner = -1 if board.get_board_state().turn == chess.WHITE else 1
            print(f"Game {game_num + 1}: Checkmate! Winner: {'black' if winner == 1 else 'white'}")
            return winner

        print(f"Game {game_num + 1}: Draw or unfinished ({move_count} moves made)")
        return 0  # Draw or unfinished

    def analyze_results(self, results):
        stats = {"bot1": 0, "bot2": 0, "draw": 0}
        for r in results:
            stats[r["winner"]] += 1
        total = len(results)
        return {
            "bot1_wins": stats["bot1"],
            "bot2_wins": stats["bot2"],
            "draws": stats["draw"],
            "bot1_win_rate": stats["bot1"] / total if total > 0 else 0,
            "bot2_win_rate": stats["bot2"] / total if total > 0 else 0,
            "draw_rate": stats["draw"] / total if total > 0 else 0
        }

    def should_update_best_bot(self, analysis):
        # Consider a bot better if it has a higher win rate
        return analysis["bot2_win_rate"] > analysis["bot1_win_rate"]

    def update_best_bot(self, challenger):
        debug_print(f"Updating best bot with challenger")
        self.best_bot.piece_values = copy.deepcopy(challenger.piece_values)
        self.best_bot.position_scores = copy.deepcopy(challenger.position_scores)
        self.best_bot.pawn_weights = copy.deepcopy(challenger.pawn_weights)
        self.best_bot.king_safety_weights = copy.deepcopy(challenger.king_safety_weights)
        self.best_bot.mobility_weights = copy.deepcopy(challenger.mobility_weights)
        self.best_bot.attack_weights = copy.deepcopy(challenger.attack_weights)
        self.best_bot.coordination_weights = copy.deepcopy(challenger.coordination_weights)
        self.best_bot.piece_combo_weights = copy.deepcopy(challenger.piece_combo_weights)
        self.best_bot.phase_weights = copy.deepcopy(challenger.phase_weights)
        self.best_bot.center_control_weight = challenger.center_control_weight
        self.best_bot.threat_weights = copy.deepcopy(challenger.threat_weights)
        self.best_bot.eval_weights = copy.deepcopy(challenger.eval_weights)


def create_bot():
    return ChessBot()


def run_sequential_training(num_iterations=50, games_per_iteration=5):
    print(f"Starting training with {games_per_iteration} games per iteration")
    print(f"Set to run for {num_iterations} iterations")
    print("You can interrupt training at any time with Ctrl+C")
    print("The best bot will be saved automatically after each improvement")

    try:
        trainer = SelfPlayTrainer(
            create_bot,
            games_per_iteration=games_per_iteration,
            iterations=num_iterations,
            trainer_id="main-trainer"
        )
        trainer.train()
        print("Training completed successfully")
    except Exception as e:
        print(f"Training encountered an error: {e}")
        traceback.print_exc()

    print("Training ended. The best bot has been saved.")


if __name__ == "__main__":
    # Example usage
    run_sequential_training(
        num_iterations=100,  # Set a high number of iterations
        games_per_iteration=5  # Games per iteration
    )