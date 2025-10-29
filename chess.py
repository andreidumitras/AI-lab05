import pygame
import os
from typing import List, Tuple, Optional, Dict, Any

# --- Pygame Setup and Constants ---
pygame.init()

# Define screen dimensions
SIZE: int = 800
ROWS: int = 8
COLS: int = 8
SQUARE_SIZE: int = SIZE // ROWS
FPS: int = 60
WIN = pygame.display.set_mode((SIZE, SIZE))
pygame.display.set_caption("Simplified Minimax Chess")

# Colors
WHITE: Tuple[int, int, int] = (255, 255, 255)
BLACK: Tuple[int, int, int] = (0, 0, 0)
BOARD_LIGHT: Tuple[int, int, int] = (238, 238, 210)
BOARD_DARK: Tuple[int, int, int] = (118, 150, 86)
HIGHLIGHT_COLOR: Tuple[int, int, int] = (255, 0, 0) # Red for selection

# Piece values (Used in the evaluation function)
PIECE_VALUES: Dict[str, int] = {
    'pawn': 10, 'knight': 30, 'bishop': 30, 'rook': 50,
    'queen': 90, 'king': 900
}

# --- Image Loading (Placeholder) ---
# NOTE: The game relies on a folder named 'images' containing the specified PNG files.
IMAGES: Dict[str, pygame.Surface] = {}
IMAGE_PATH: str = 'images'
PIECES_NAMES: List[str] = [
    'black-bishop', 'black-knight', 'black-queen', 'white-bishop', 'white-knight', 'white-queen',
    'black-king', 'black-pawn', 'black-rook', 'white-king', 'white-pawn', 'white-rook'
]

def load_images() -> None:
    """Loads and scales all piece images."""
    try:
        for name in PIECES_NAMES:
            path: str = os.path.join(IMAGE_PATH, f'{name}.png')
            image: pygame.Surface = pygame.image.load(path).convert_alpha()
            scaled_image: pygame.Surface = pygame.transform.scale(image, (SQUARE_SIZE, SQUARE_SIZE))
            # Store in the global IMAGES dictionary
            IMAGES[name] = scaled_image
    except pygame.error as e:
        # Fallback if images directory is missing or images are not found
        print(f"Error loading image: {e}. Using a simple rectangle placeholder.")
        # We will use simple drawing logic if images fail to load
        pass

# Call image loading at startup
load_images()

# --- Piece Class and Board State ---

class Piece:
    """Represents a single chess piece."""
    def __init__(self, color: str, piece_type: str, row: int, col: int):
        self.color: str = color  # 'white' or 'black'
        self.type: str = piece_type  # 'pawn', 'king', 'queen', etc.
        self.row: int = row
        self.col: int = col
        self.image_name: str = f'{color}-{piece_type}'

    def draw(self, surface: pygame.Surface) -> None:
        """Draws the piece on the given surface."""
        x: int = self.col * SQUARE_SIZE
        y: int = self.row * SQUARE_SIZE

        if self.image_name in IMAGES:
            surface.blit(IMAGES[self.image_name], (x, y))
        else:
            # Simple placeholder drawing if images are not available
            color_rgb = WHITE if self.color == 'white' else BLACK
            pygame.draw.circle(surface, color_rgb, (x + SQUARE_SIZE // 2, y + SQUARE_SIZE // 2), SQUARE_SIZE // 3)
            font = pygame.font.Font(None, 24)
            text = font.render(self.type[0].upper(), True, BLACK if self.color == 'white' else WHITE)
            surface.blit(text, (x + SQUARE_SIZE // 2 - text.get_width() // 2, y + SQUARE_SIZE // 2 - text.get_height() // 2))

class Board:
    """Manages the chess board state."""
    def __init__(self):
        # A list of lists representing the 8x8 grid
        self.board: List[List[Optional[Piece]]] = [[None for _ in range(COLS)] for _ in range(ROWS)]
        self.initialize_pieces()

    def initialize_pieces(self) -> None:
        """Sets up the initial board configuration (simplified chess)."""
        # Pawns
        for c in range(COLS):
            self.board[6][c] = Piece('white', 'pawn', 6, c)
            self.board[1][c] = Piece('black', 'pawn', 1, c)

        # Rooks, Knights, Bishops, Queens, Kings (Simplified placement)
        initial_order: List[str] = ['rook', 'knight', 'bishop', 'queen', 'king', 'bishop', 'knight', 'rook']
        for c, piece_type in enumerate(initial_order):
            self.board[7][c] = Piece('white', piece_type, 7, c)
            self.board[0][c] = Piece('black', piece_type, 0, c)

    def get_piece(self, row: int, col: int) -> Optional[Piece]:
        """Returns the piece at a given position."""
        if 0 <= row < ROWS and 0 <= col < COLS:
            return self.board[row][col]
        return None

    def make_move(self, start_pos: Tuple[int, int], end_pos: Tuple[int, int]) -> None:
        """Executes a move by updating the board state."""
        start_row, start_col = start_pos
        end_row, end_col = end_pos

        piece: Optional[Piece] = self.board[start_row][start_col]
        if piece is None:
            return # Should not happen if move is generated correctly

        # Check for pawn promotion (simplified: promote to Queen if it reaches the end)
        if piece.type == 'pawn':
            if piece.color == 'white' and end_row == 0:
                piece.type = 'queen'
                piece.image_name = 'white-queen'
            elif piece.color == 'black' and end_row == 7:
                piece.type = 'queen'
                piece.image_name = 'black-queen'

        # Move the piece
        self.board[end_row][end_col] = piece
        self.board[start_row][start_col] = None

        # Update piece's internal position
        piece.row = end_row
        piece.col = end_col

    def is_on_board(self, r: int, c: int) -> bool:
        """Checks if coordinates are within the board boundaries."""
        return 0 <= r < ROWS and 0 <= c < COLS

    def get_all_pieces(self, color: str) -> List[Piece]:
        """Returns a list of all pieces of a specific color."""
        pieces: List[Piece] = []
        for r in range(ROWS):
            for c in range(COLS):
                piece: Optional[Piece] = self.board[r][c]
                if piece and piece.color == color:
                    pieces.append(piece)
        return pieces

    def get_valid_moves(self, piece: Piece) -> List[Tuple[int, int]]:
        """
        Generates valid moves for a single piece.
        NOTE: This is a simplified move generation, ignoring check/checkmate logic
        to focus on the Minimax implementation.
        """
        moves: List[Tuple[int, int]] = []
        r, c = piece.row, piece.col
        opponent_color: str = 'black' if piece.color == 'white' else 'white'

        def check_direction(dr: int, dc: int, limit: int = 8) -> None:
            """Helper for sliding pieces (Rook, Bishop, Queen)."""
            for i in range(1, limit):
                nr, nc = r + dr * i, c + dc * i
                if not self.is_on_board(nr, nc):
                    break
                target_piece: Optional[Piece] = self.board[nr][nc]
                if target_piece is None:
                    moves.append((nr, nc)) # Empty square
                elif target_piece.color == opponent_color:
                    moves.append((nr, nc)) # Capture
                    break # Stop after capturing
                else:
                    break # Blocked by own piece

        if piece.type == 'pawn':
            direction: int = -1 if piece.color == 'white' else 1
            # Single forward move
            nr, nc = r + direction, c
            if self.is_on_board(nr, nc) and self.board[nr][nc] is None:
                moves.append((nr, nc))
                # Double forward move from start position
                if (piece.color == 'white' and r == 6) or (piece.color == 'black' and r == 1):
                    nr2, nc2 = r + 2 * direction, c
                    if self.board[nr2][nc2] is None:
                        moves.append((nr2, nc2))

            # Diagonal captures
            for dc in [-1, 1]:
                nr, nc = r + direction, c + dc
                if self.is_on_board(nr, nc):
                    target_piece: Optional[Piece] = self.board[nr][nc]
                    if target_piece and target_piece.color == opponent_color:
                        moves.append((nr, nc))

        elif piece.type == 'rook':
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                check_direction(dr, dc)

        elif piece.type == 'bishop':
            for dr, dc in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                check_direction(dr, dc)

        elif piece.type == 'queen':
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                check_direction(dr, dc)

        elif piece.type == 'king':
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    check_direction(dr, dc, limit=2) # King moves only 1 step

        elif piece.type == 'knight':
            knight_moves: List[Tuple[int, int]] = [
                (r - 2, c - 1), (r - 2, c + 1), (r - 1, c - 2), (r - 1, c + 2),
                (r + 1, c - 2), (r + 1, c + 2), (r + 2, c - 1), (r + 2, c + 1)
            ]
            for nr, nc in knight_moves:
                if self.is_on_board(nr, nc):
                    target_piece: Optional[Piece] = self.board[nr][nc]
                    if target_piece is None or target_piece.color == opponent_color:
                        moves.append((nr, nc))

        return moves

    def generate_all_valid_moves(self, color: str) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Generates all possible moves for the given color (start_pos, end_pos)."""
        all_moves: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        for piece in self.get_all_pieces(color):
            start_pos: Tuple[int, int] = (piece.row, piece.col)
            valid_ends: List[Tuple[int, int]] = self.get_valid_moves(piece)
            for end_pos in valid_ends:
                all_moves.append((start_pos, end_pos))
        return all_moves

    def get_board_copy(self) -> 'Board':
        """Returns a deep copy of the current board instance."""
        new_board = Board()
        # Clear the default setup
        new_board.board = [[None for _ in range(COLS)] for _ in range(ROWS)]

        for r in range(ROWS):
            for c in range(COLS):
                piece: Optional[Piece] = self.board[r][c]
                if piece:
                    # Create a new Piece instance for the new board
                    new_piece: Piece = Piece(piece.color, piece.type, piece.row, piece.col)
                    new_board.board[r][c] = new_piece
        return new_board

# --- Minimax AI Logic ---

def evaluate_board(board: Board) -> int:
    """
    Simple material evaluation function.
    Positive score favors White (User), Negative favors Black (AI).
    """
    score: int = 0
    for r in range(ROWS):
        for c in range(COLS):
            piece: Optional[Piece] = board.get_piece(r, c)
            if piece:
                value: int = PIECE_VALUES.get(piece.type, 0)
                # Assign scores based on color
                if piece.color == 'white':
                    score += value
                elif piece.color == 'black':
                    score -= value
    return score

def minimax(board: Board, depth: int, is_maximizing_player: bool, alpha: float, beta: float) -> Tuple[int, Optional[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    """
    Minimax algorithm with Alpha-Beta Pruning.
    Returns: (score, best_move)
    """


    """
The Pseudocode:
// Determine which color to move
if is_maximizing_player then
    current_color = 'white'
else
    current_color = 'black'

// Generate all legal moves for the current player
possible_moves = board.generate_all_valid_moves(current_color)

// If no legal moves, treat as terminal (simplified: evaluate current board)
if possible_moves is empty then
    return (EVALUATE_BOARD(board), None)

best_move = None

if is_maximizing_player then
    max_eval = -INFINITY
    for each move in possible_moves do
        // Apply move on a copy of the board
        new_board = board.get_board_copy()
        new_board.make_move(move.start, move.end)

        // Recurse: next is minimizing player
        (evaluation, _) = MINIMAX(new_board, depth - 1, False, alpha, beta)

        // Track best value and move
        if evaluation > max_eval then
            max_eval = evaluation
            best_move = move
        end if

        // Alpha-beta update
        alpha = MAX(alpha, max_eval)
        if beta <= alpha then
            // Beta cut-off: no need to examine further moves
            break
        end if
    end for

    // Return integer score and chosen move
    return (INT(max_eval), best_move)

else
    // Minimizing player (Black / AI)
    min_eval = +INFINITY
    for each move in possible_moves do
        new_board = board.get_board_copy()
        new_board.make_move(move.start, move.end)

        // Recurse: next is maximizing player
        (evaluation, _) = MINIMAX(new_board, depth - 1, True, alpha, beta)

        if evaluation < min_eval then
            min_eval = evaluation
            best_move = move
        end if

        // Alpha-beta update
        beta = MIN(beta, min_eval)
        if beta <= alpha then
            // Alpha cut-off
            break
        end if
    end for

    return (INT(min_eval), best_move)
end if
    """

# --- Drawing Functions ---

def draw_board(surface: pygame.Surface) -> None:
    """Draws the checkerboard pattern."""
    surface.fill(BOARD_LIGHT)
    for r in range(ROWS):
        for c in range(COLS):
            if (r + c) % 2 == 1:
                pygame.draw.rect(surface, BOARD_DARK, (c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

def draw_pieces(surface: pygame.Surface, board: Board) -> None:
    """Draws all pieces on the board."""
    for r in range(ROWS):
        for c in range(COLS):
            piece: Optional[Piece] = board.get_piece(r, c)
            if piece:
                piece.draw(surface)

def draw_highlight(surface: pygame.Surface, pos: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    """Highlights a single square at (row, col)."""
    r, c = pos
    x: int = c * SQUARE_SIZE
    y: int = r * SQUARE_SIZE
    pygame.draw.rect(surface, color, (x, y, SQUARE_SIZE, SQUARE_SIZE), 5) # Draw a thick border

def draw_valid_moves(surface: pygame.Surface, moves: List[Tuple[int, int]]) -> None:
    """Draws small circles on all possible destination squares."""
    for r, c in moves:
        x: int = c * SQUARE_SIZE + SQUARE_SIZE // 2
        y: int = r * SQUARE_SIZE + SQUARE_SIZE // 2
        pygame.draw.circle(surface, HIGHLIGHT_COLOR, (x, y), 10)

def display_message(surface: pygame.Surface, text: str) -> None:
    """Draws a centered message overlay."""
    font = pygame.font.Font(None, 74)
    text_surface = font.render(text, True, (255, 255, 255))
    text_rect = text_surface.get_rect(center=(SIZE // 2, SIZE // 2))
    
    # Draw a background rectangle for better visibility
    bg_rect = text_rect.inflate(40, 20)
    pygame.draw.rect(surface, (0, 0, 0, 180), bg_rect, border_radius=10)
    
    surface.blit(text_surface, text_rect)
    pygame.display.flip()
    pygame.time.wait(3000) # Wait 3 seconds

# --- Main Game Loop ---

def get_row_col_from_mouse(pos: Tuple[int, int]) -> Tuple[int, int]:
    """Converts mouse (x, y) coordinates to (row, col)."""
    x, y = pos
    row: int = y // SQUARE_SIZE
    col: int = x // SQUARE_SIZE
    return row, col

if __name__ == '__main__':
    """Main function to run the Pygame game."""
    run: bool = True
    clock: pygame.time.Clock = pygame.time.Clock()
    board: Board = Board()
    current_turn: str = 'white' # User starts as white

    # Selection state
    selected_piece_pos: Optional[Tuple[int, int]] = None
    valid_moves: List[Tuple[int, int]] = []
    
    # Game state flags
    game_over: bool = False
    ai_thinking: bool = False
    
    # AI depth (tune for performance vs difficulty)
    AI_DEPTH: int = 3

    while run:
        clock.tick(FPS)
        
        # 1. AI Turn Logic
        if current_turn == 'black' and not game_over and not ai_thinking:
            ai_thinking = True
            
            # Simple check for immediate game over (no moves left)
            if not board.generate_all_valid_moves('black'):
                 display_message(WIN, "White Wins by Stalemate/No Moves!")
                 game_over = True
                 ai_thinking = False
                 continue
                 
            # Minimax call
            # Note: We maximize for White (User) score, so the AI (Black)
            # wants the minimum score, hence is_maximizing_player=False.
            print(f"AI is thinking at depth {AI_DEPTH}...")
            score, ai_move = minimax(board, AI_DEPTH, False, -float('inf'), float('inf'))
            print(f"AI Score: {score}, Best Move: {ai_move}")
            
            if ai_move:
                start_pos, end_pos = ai_move
                board.make_move(start_pos, end_pos)
                current_turn = 'white'
            else:
                # If AI can't find a move, it's a win for the user
                display_message(WIN, "White Wins!")
                game_over = True
            
            ai_thinking = False

        # 2. Event Handling (User Turn)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            
            if current_turn == 'white' and not game_over and not ai_thinking:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos: Tuple[int, int] = pygame.mouse.get_pos()
                    r, c = get_row_col_from_mouse(pos)
                    
                    if selected_piece_pos:
                        # Attempt a move
                        if (r, c) in valid_moves:
                            start_r, start_c = selected_piece_pos
                            board.make_move((start_r, start_c), (r, c))
                            current_turn = 'black' # Switch turn to AI
                            
                        # Deselect or select new piece
                        selected_piece_pos = None
                        valid_moves = []

                    # Select a new piece
                    piece: Optional[Piece] = board.get_piece(r, c)
                    if piece and piece.color == 'white':
                        selected_piece_pos = (r, c)
                        valid_moves = board.get_valid_moves(piece)
                        
                        # If no valid moves, reset selection
                        if not valid_moves:
                            selected_piece_pos = None
                        
                # Simple check for immediate game over (no moves left)
                if current_turn == 'white' and not board.generate_all_valid_moves('white'):
                    display_message(WIN, "Black Wins by Stalemate/No Moves!")
                    game_over = True


        # 3. Drawing
        draw_board(WIN)
        
        # Highlight selected piece and its valid moves
        if selected_piece_pos:
            draw_highlight(WIN, selected_piece_pos, HIGHLIGHT_COLOR)
            draw_valid_moves(WIN, valid_moves)
            
        draw_pieces(WIN, board)
        
        # Draw game state indicator
        font = pygame.font.Font(None, 36)
        status_text: str = f"Turn: {current_turn.upper()}"
        if ai_thinking:
             status_text = "AI Thinking..."
        
        text_surface = font.render(status_text, True, BLACK)
        WIN.blit(text_surface, (10, 10))
        
        if game_over:
             # Redisplay final message
             display_message(WIN, "GAME OVER!")


        pygame.display.flip()

    pygame.quit()

