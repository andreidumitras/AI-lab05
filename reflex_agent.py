import pygame
import random
import sys

# --- Constants ---
WINDOW_SIZE = 400
GRID_SIZE = 20
FPS = 30

# Colors
WHITE = (255, 255, 255)
RED   = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)


# --- Environment Setup ---
pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Reflex Agent Example")
clock = pygame.time.Clock()

# --- Entities ---
def create_agent() -> dict:
    """Return a dictionary representing the agent."""
    return {"x": 5, "y": 5}

def create_food() -> dict:
    """Return random food coordinates on the grid."""
    return {
        "x": random.randint(0, (WINDOW_SIZE // GRID_SIZE) - 1),
        "y": random.randint(0, (WINDOW_SIZE // GRID_SIZE) - 1)
    }

# ------ Reflex Agent Behavior ---------
def reflex_action(agent: dict, food: dict) -> None:
    """
    Decide the next move based on the current percept.
    Simple condition-action rules:
    Move one step toward the food.
    """
    # TODO:
    # Compare agent position vs. food position.
    # Move one step toward it (left, right, up, or down).
    # It ignores history, walls, or any other obstacles - only current percepts.
    # If same position, stay put.
    pass

def manual_action(agent: dict, keys: int) -> None:
    """Move based on user input (arrow keys)."""
    if keys[pygame.K_UP]:
        agent["y"] -= 1
    elif keys[pygame.K_DOWN]:
        agent["y"] += 1
    elif keys[pygame.K_LEFT]:
        agent["x"] -= 1
    elif keys[pygame.K_RIGHT]:
        agent["x"] += 1

# --- Game Loop Logic ---
def update(agent: dict, food: dict, mode: str, keys) -> None:
    """Update the environment state each frame."""
    # TODO:
    # The agent decides and moves (via reflex_action).
    # If the agent has reached the food’s position, we replace it with a new random food.
    if mode == 'auto':
        pass
    elif mode == 'manual':
        manual_action(agent, keys)
        agent['x'] = max(0, min(agent['x'], (WINDOW_SIZE // GRID_SIZE) - 1))
        agent['y'] = max(0, min(agent['y'], (WINDOW_SIZE // GRID_SIZE) - 1))
        if agent['x'] == food['x'] and agent['y'] == food['y']:
            food.update(create_food())


def draw(screen: "Surface", agent: dict, food: dict, mode: str) -> None:
    """Draw all elements on screen."""
    screen.fill(BLACK)

    # Draw food
    pygame.draw.circle(
        screen,
        GREEN,
        (food["x"] * GRID_SIZE + GRID_SIZE // 2, food["y"] * GRID_SIZE + GRID_SIZE // 2),
        GRID_SIZE // 2
    )

    # Draw agent
    pygame.draw.rect(
        screen,
        RED,
        (agent["x"] * GRID_SIZE, agent["y"] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
    )
    
    font = pygame.font.SysFont(None, 24)
    text = font.render(f"Mode: {mode.upper()}", True, WHITE)
    screen.blit(text, (10, 10))

    pygame.display.flip()


def handle_events(mode: str) -> str:
    """Handle quit events."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_a:
            mode = 'manual' if mode == 'auto' else 'auto'
            print(mode)
    return mode


# --- Run Game ---
if __name__ == "__main__":
    agent = create_agent()
    food = create_food()
    mode = 'manual'
    while True:
        mode = handle_events(mode)
        keys = pygame.key.get_pressed()

        update(agent, food, mode, keys)
        draw(screen, agent, food, mode)
        clock.tick(FPS)
