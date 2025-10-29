import pygame
import time
from typing import List, Optional

# Config
FPS = 30
CELL = 100
MARGIN = 20
FONT_SIZE = 20

# Utility
def ttt_winner(board: List[int]) -> Optional[int]:
    wins = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
    for a,b,c in wins:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    if all(x != 0 for x in board):
        return 0  # draw
    return None

def minimax(board: List[int], player: int) -> int:
    """
    Minimax from perspective: 1 = human (maximizer), -1 = AI (minimizer).
    Returns score.
    """
    # TODO:
# get the ttt_winner
#     if winner == 1 then
#         best = 1                       // use -2 in original code since scores are in [-1,1]
#     else if winner -1
#         best = -1
#     else return 0                     // use  2 in original code
#   best = -2 if player == 1 else best = 2 if player != 1

# for i from 0 to 8 do
#     if board[i] == 0 then          // empty cell
#         board[i] = player          // make move
#         value = MINIMAX(board, -player)   // evaluate resulting position
#         board[i] = 0               // undo move

#         if player == 1 then
#             best = max(best, value)    // maximizer chooses highest score
#         else
#             best = min(best, value)    // minimizer chooses lowest score
#         end if
#     end if
# end for
# return best
    pass

def run_minimax_tictactoe():
    """
    Minimax Tic-Tac-Toe demo.
    Human = 1 (X), AI = -1 (O). Human starts by default.
    Controls: click to place (human), A toggle auto (AI plays automatically), R reset.
    """
    pygame.init()
    W = 3 * CELL + 2 * MARGIN
    H = 3 * CELL + 120
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption('Minimax Tic-Tac-Toe')
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, FONT_SIZE)

    board = [0] * 9
    human = 1
    ai = -1
    turn = 1  # human starts
    auto = False

    def draw_board():
        screen.fill((255, 255, 255))
        for r in range(3):
            for c in range(3):
                rect = pygame.Rect(MARGIN + c * CELL, MARGIN + r * CELL, CELL - 2, CELL - 2)
                pygame.draw.rect(screen, (230, 230, 230), rect)
                v = board[r * 3 + c]
                if v == 1:
                    pygame.draw.line(screen, (0,0,0), rect.topleft + pygame.Vector2(10,10), rect.bottomright - pygame.Vector2(10,10), 4)
                    pygame.draw.line(screen, (0,0,0), rect.topright + pygame.Vector2(-10,10), rect.bottomleft + pygame.Vector2(10,-10), 4)
                elif v == -1:
                    pygame.draw.circle(screen, (0,0,0), rect.center, 28, 4)
        # evaluations for AI choices (display minimax scores)
        if turn == -1:
            for i in range(9):
                if board[i] == 0:
                    board[i] = -1
                    val = minimax(board, 1)
                    board[i] = 0
                    r = i // 3; c = i % 3
                    txt = font.render(str(val), True, (0, 100, 0))
                    screen.blit(txt, (MARGIN + c * CELL + CELL - 20, MARGIN + r * CELL + CELL - 24))
        # footer/status
        scr = ttt_winner(board)
        status = 'In progress' if scr is None else ('Draw' if scr == 0 else ('Human (X) wins' if scr == 1 else 'AI (O) wins'))
        screen.blit(font.render(f'Mode: Minimax - Turn: {"Human" if turn == 1 else "AI"}', True, (0,0,0)), (10, 3 * CELL + 30))
        screen.blit(font.render(f'Status: {status}', True, (0,0,0)), (10, 3 * CELL + 55))
        pygame.display.flip()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    auto = not auto
                elif event.key == pygame.K_r:
                    board[:] = [0] * 9
                    turn = 1
            elif event.type == pygame.MOUSEBUTTONDOWN and turn == 1:
                mx, my = event.pos
                if MARGIN < mx < MARGIN + 3 * CELL and MARGIN < my < MARGIN + 3 * CELL:
                    c = (mx - MARGIN) // CELL
                    r = (my - MARGIN) // CELL
                    idx = r * 3 + c
                    if board[idx] == 0:
                        board[idx] = 1
                        turn = -1
        if turn == -1 and (auto or True):
            # AI picks best move by minimax
            best_val = 2
            best_move = None
            for i in range(9):
                if board[i] == 0:
                    board[i] = -1
                    val = minimax(board, 1)
                    board[i] = 0
                    if val < best_val:
                        best_val = val
                        best_move = i
            if best_move is not None:
                board[best_move] = -1
            turn = 1
            time.sleep(0.2)

        draw_board()
        clock.tick(FPS)
    pygame.quit()

if __name__ == "__main__":
    run_minimax_tictactoe()