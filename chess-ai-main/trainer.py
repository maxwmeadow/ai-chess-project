from board import ChessBoard
import time, random, json, copy
import chess
from bot import ChessBot
import json


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
        # New weights to save
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

    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def load_bot_from_json(bot, filename):
    with open(filename, 'r') as f:
        data = json.load(f)

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


class SelfPlayTrainer:
    def __init__(self, bot_constructor, games_per_iteration=10, iterations=3):
        self.bot_constructor = bot_constructor
        self.games_per_iteration = games_per_iteration
        self.iterations = iterations
        self.best_bot = bot_constructor()
        self.performance_history = []

        try:
            load_bot_from_json(self.best_bot, "best_bot.json")
            print("Loaded existing best_bot.json!")
        except FileNotFoundError:
            print("No existing best_bot.json found, starting fresh.")

        self.performance_history = []

    def train(self):
        for i in range(self.iterations):
            print(f"--- Iteration {i + 1} ---")
            challenger = self.create_challenger()
            results = self.play_match(self.best_bot, challenger)
            analysis = self.analyze_results(results)
            print(f"Results: {analysis}")
            self.update_best_bot(challenger, analysis)
            self.performance_history.append({
                "iteration": i + 1,
                "timestamp": time.time(),
                "metrics": analysis
            })

        save_bot_to_json(self.best_bot, "best_bot.json")
        print("Best bot saved to best_bot.json!")
        return self.best_bot

    def create_challenger(self):
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

        # Mutate piece values (original code)
        for piece in challenger.piece_values:
            challenger.piece_values[piece] *= random.uniform(0.95, 1.05)

        # Mutate piece-square tables (original code)
        for piece in challenger.position_scores:
            for i in range(len(challenger.position_scores[piece])):
                challenger.position_scores[piece][i] *= random.uniform(0.95, 1.05)

        # Mutate all the new weights

        # Pawn weights
        for key in challenger.pawn_weights:
            challenger.pawn_weights[key] *= random.uniform(0.95, 1.05)

        # King safety weights
        for key in challenger.king_safety_weights:
            challenger.king_safety_weights[key] *= random.uniform(0.95, 1.05)

        # Mobility weights
        for key in challenger.mobility_weights:
            challenger.mobility_weights[key] *= random.uniform(0.95, 1.05)

        # Attack weights
        for key in challenger.attack_weights:
            challenger.attack_weights[key] *= random.uniform(0.95, 1.05)

        # Coordination weights
        for key in challenger.coordination_weights:
            challenger.coordination_weights[key] *= random.uniform(0.95, 1.05)

        # Piece combo weights
        for key in challenger.piece_combo_weights:
            challenger.piece_combo_weights[key] *= random.uniform(0.95, 1.05)

        # Phase weights
        for key in challenger.phase_weights:
            challenger.phase_weights[key] *= random.uniform(0.95, 1.05)

        # Center control weight
        challenger.center_control_weight *= random.uniform(0.95, 1.05)

        # Threat weights
        for key in challenger.threat_weights:
            challenger.threat_weights[key] *= random.uniform(0.95, 1.05)

        # Eval weights
        for key in challenger.eval_weights:
            challenger.eval_weights[key] *= random.uniform(0.95, 1.05)

        return challenger

    def play_match(self, bot1, bot2):
        results = []
        for game_num in range(self.games_per_iteration):
            board = ChessBoard()
            white, black = (bot1, bot2) if game_num % 2 == 0 else (bot2, bot1)
            color_map = {1: "bot1", -1: "bot2", 0: "draw"} if game_num % 2 == 0 else {-1: "bot1", 1: "bot2", 0: "draw"}
            result_code = self.play_game(board, white, black)
            results.append({
                "game_num": game_num,
                "winner": color_map[result_code],
                "result_code": result_code
            })
        return results

    def play_game(self, board, white_bot, black_bot):
        for _ in range(80):
            if board.is_game_over():
                break
            bot = white_bot if board.get_board_state().turn == chess.WHITE else black_bot
            move = bot.get_move(board)
            if move:
                board.make_move(move)
            else:
                break
        if board.get_board_state().is_checkmate():
            return -1 if board.get_board_state().turn == chess.WHITE else 1
        return 0

    def analyze_results(self, results):
        stats = {"bot1": 0, "bot2": 0, "draw": 0}
        for r in results:
            stats[r["winner"]] += 1
        total = len(results)
        return {
            "bot1_wins": stats["bot1"],
            "bot2_wins": stats["bot2"],
            "draws": stats["draw"],
            "bot1_win_rate": stats["bot1"] / total,
            "bot2_win_rate": stats["bot2"] / total,
            "draw_rate": stats["draw"] / total
        }

    def update_best_bot(self, challenger, analysis):
        if analysis["bot2_win_rate"] > analysis["bot1_win_rate"]:
            print("Challenger is better! Updating best_bot.")
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


if __name__ == "__main__":
    trainer = SelfPlayTrainer(create_bot, games_per_iteration=5, iterations=1)
    best_bot = trainer.train()