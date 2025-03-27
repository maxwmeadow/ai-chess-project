import random
import chess

def create_simple_opening_book():
    book = {}
    ob = OpeningBook()

    def add(fen_str, moves):
        book[fen_str] = [(move, weight) for move, weight in moves]

    # Initial position
    board = chess.Board()
    add(ob.simplified_fen(board), [
        ("e2e4", 100), ("d2d4", 90), ("c2c4", 80), ("g1f3", 70)
    ])

    # 1. e4
    board.push_san("e4")
    add(ob.simplified_fen(board), [
        ("e7e5", 100), ("c7c5", 90), ("e7e6", 80), ("c7c6", 70)
    ])

    # 1. e4 e5
    board.push_san("e5")
    add(ob.simplified_fen(board), [
        ("g1f3", 100), ("f1c4", 90), ("f1b5", 90)
    ])

    # Ruy López: 1.e4 e5 2.Nf3 Nc6 3.Bb5
    board.push_san("Nf3")
    add(ob.simplified_fen(board), [
        ("b8c6", 100)
    ])
    board.push_san("Nc6")
    add(ob.simplified_fen(board), [
        ("f1b5", 100)
    ])

    # Italian: 1.e4 e5 2.Nf3 Nc6 3.Bc4
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")
    board.push_san("Nc6")
    add(ob.simplified_fen(board), [
        ("f1c4", 100)
    ])

    # 1. d4
    board = chess.Board()
    board.push_san("d4")
    add(ob.simplified_fen(board), [
        ("d7d5", 100), ("g8f6", 90), ("f7f5", 80)
    ])

    # 1. d4 d5
    board.push_san("d5")
    add(ob.simplified_fen(board), [
        ("c2c4", 100)
    ])

    # Queen's Gambit: 1.d4 d5 2.c4
    board.push_san("c4")
    add(ob.simplified_fen(board), [
        ("e7e6", 90), ("c7c6", 90), ("d5c4", 80)
    ])

    # 1. d4 Nf6
    board = chess.Board()
    board.push_san("d4")
    board.push_san("Nf6")
    add(ob.simplified_fen(board), [
        ("c2c4", 100), ("g1f3", 90), ("g2g3", 80)
    ])

    # 1. Nf3
    board = chess.Board()
    board.push_san("Nf3")
    add(ob.simplified_fen(board), [
        ("d7d5", 100), ("g8f6", 90), ("c7c5", 80)
    ])

    # 1. c4
    board = chess.Board()
    board.push_san("c4")
    add(ob.simplified_fen(board), [
        ("e7e5", 100), ("g8f6", 90), ("c7c5", 80)
    ])

    return book

class OpeningBook:
    def __init__(self, book_data=None, max_book_depth=8):
        self.book = book_data if book_data else {}
        self.max_book_depth = max_book_depth

    def simplified_fen(self, board):
        full_fen = board.fen()
        return ' '.join(full_fen.split(' ')[:2])

    def get_move(self, board, variation=0.2):
        if board.fullmove_number > self.max_book_depth:
            return None

        fen = self.simplified_fen(board)
        if fen not in self.book:
            return None

        moves = self.book[fen]
        moves.sort(key=lambda x: x[1], reverse=True)

        if variation > 0 and len(moves) > 1 and random.random() < variation:
            top_moves = moves[:min(3, len(moves))]
            total_weight = sum(weight for _, weight in top_moves)
            r = random.uniform(0, total_weight)
            cumulative = 0
            for move, weight in top_moves:
                cumulative += weight
                if r <= cumulative:
                    return chess.Move.from_uci(move)

        return chess.Move.from_uci(moves[0][0])

class PolyglotBook:
    def __init__(self, book_path):
        self.book_path = book_path

    def get_move(self, board):
        try:
            with chess.polyglot.open_reader(self.book_path) as reader:
                entries = list(reader.find_all(board))
                if entries:
                    entries.sort(key=lambda entry: entry.weight, reverse=True)
                    return entries[0].move
        except Exception as e:
            print(f"[Polyglot Error] {e}")
        return None