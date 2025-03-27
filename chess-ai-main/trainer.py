from board import ChessBoard
import time, random, json, copy
import chess
from bot import ChessBot

class SelfPlayTrainer:
    def __init__(self, bot_constructor, games_per_iteration=10, iterations=3):
        self.bot_constructor = bot_constructor
        self.games_per_iteration = games_per_iteration
        self.iterations = iterations
        self.best_bot = bot_constructor()
        self.performance_history = []

    def train(self):
        for i in range(self.iterations):
            print(f"--- Iteration {i+1} ---")
            challenger = self.create_challenger()
            results = self.play_match(self.best_bot, challenger)
            analysis = self.analyze_results(results)
            print(f"Results: {analysis}")
            self.update_best_bot(challenger, analysis)
            self.performance_history.append({
                "iteration": i+1,
                "timestamp": time.time(),
                "metrics": analysis
            })
        return self.best_bot

    def create_challenger(self):
        challenger = self.bot_constructor()
        for piece in challenger.piece_values:
            challenger.piece_values[piece] *= random.uniform(0.95, 1.05)
        for piece in challenger.position_scores:
            for i in range(len(challenger.position_scores[piece])):
                challenger.position_scores[piece][i] *= random.uniform(0.95, 1.05)
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
        for _ in range(200):
            if board.is_game_over():
                break
            bot = white_bot if board.get_board_state().turn == chess.WHITE else black_bot
            move = bot.get_move(board)
            if move:
                board.make_move(move)
            else:
                break
        if board.is_checkmate():
            return -1 if board.turn == chess.WHITE else 1
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


def create_bot():
    return ChessBot()

if __name__ == "__main__":
    trainer = SelfPlayTrainer(create_bot, games_per_iteration=10, iterations=3)
    best_bot = trainer.train()