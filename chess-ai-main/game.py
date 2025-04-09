import chess
import chess.svg
from board import ChessBoard
from bot import ChessBot
from human import HumanPlayer
import pygame
import cairosvg
import io
from PIL import Image

IS_BOT = True  # Set to False for human vs bot, True for bot vs bot


class ChessGame:
    def __init__(self):
        self.board = ChessBoard()

        # Initialize players based on IS_BOT flag
        if IS_BOT:
            self.white_player = ChessBot()
            self.black_player = ChessBot()
        else:
            self.white_player = HumanPlayer(chess.WHITE, self)
            self.black_player = ChessBot()

        # Initialize Pygame
        pygame.init()
        self.WINDOW_SIZE = 600
        self.screen = pygame.display.set_mode((self.WINDOW_SIZE, self.WINDOW_SIZE))
        pygame.display.set_caption("Chess Game")

    def svg_to_pygame_surface(self, svg_string):
        """Convert SVG string to Pygame surface"""
        png_data = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'))
        image = Image.open(io.BytesIO(png_data))
        image = image.resize((self.WINDOW_SIZE, self.WINDOW_SIZE))
        mode = image.mode
        size = image.size
        data = image.tobytes()
        return pygame.image.fromstring(data, size, mode)

    def display_board(self, last_move=None, selected_square=None):
        """Display the current board state"""
        # Build highlight dictionary for the selected square
        highlight_squares = None
        if selected_square is not None:
            highlight_squares = {
                selected_square: {"fill": "#FFFF00", "stroke": "none"}
            }

        # Create SVG with highlighted last move and selected square
        svg = chess.svg.board(
            board=self.board.get_board_state(),
            lastmove=last_move,
            squares=highlight_squares,  # colored square highlight
            size=self.WINDOW_SIZE
        )

        # Convert SVG to Pygame surface and display
        py_image = self.svg_to_pygame_surface(svg)
        self.screen.blit(py_image, (0, 0))
        pygame.display.flip()

    def process_events(self):
        """Process Pygame events to keep window responsive"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        return True

    def play_game(self):
        """Main game loop"""
        last_move = None
        running = True

        while running and not self.board.is_game_over():
            # Process events to keep window responsive
            running = self.process_events()
            if not running:
                break

            # Get current player for selected square highlighting
            current_player = self.white_player if self.board.get_board_state().turn else self.black_player
            selected_square = getattr(current_player, 'selected_square', None)

            # Display current board with highlights
            self.display_board(last_move, selected_square)

            # Determine current player
            current_player = self.white_player if self.board.get_board_state().turn else self.black_player

            # Get player's move with periodic UI updates
            move = None
            if isinstance(current_player, ChessBot):
                # For bots, get the move in chunks to keep UI responsive
                for _ in range(10):  # Check for move 10 times before giving up
                    move = current_player.get_move(self.board)
                    if move is not None:
                        break
                    pygame.time.wait(100)  # Short delay between checks
                    running = self.process_events()
                    if not running:
                        break
                    self.display_board(last_move, selected_square)
            else:
                move = current_player.get_move(self.board)

            if move is None:
                print("Game ended by player")
                break

            # Make the move
            if not self.board.make_move(move):
                print(f"Illegal move attempted: {move}")
                break

            print(f"Move played: {move}")
            last_move = move

            # Add delay only for bot moves and update display
            if isinstance(current_player, ChessBot):
                for _ in range(10):  # Split delay into smaller chunks
                    pygame.time.wait(100)
                    running = self.process_events()
                    if not running:
                        break
                    self.display_board(last_move)

        # Display final position if game ended normally
        if running:
            self.display_board(last_move)
            result = self.board.get_result()
            print(f"Game Over! Result: {result}")

            # Keep window open until closed
            while running:
                running = self.process_events()
                pygame.time.wait(100)

        pygame.quit()


if __name__ == "__main__":
    game = ChessGame()
    game.play_game()