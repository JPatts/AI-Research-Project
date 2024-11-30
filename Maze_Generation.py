import pygame
import sys
import random
import heapq
from collections import deque

# Initialize pygame
pygame.init()

# Set colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200,200,200) 

# Set grid size
GRID_SIZE = 60  # Size of each cell in pixels
LINE_WIDTH = 3  # Width of walls

# Function to set up screen size dynamically
def setup_screen(num_rows, num_cols):
    width = num_cols * GRID_SIZE
    height = num_rows * GRID_SIZE
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("HUMAN vs ZOMBIE")
    return screen

# Draw the maze
def draw_maze(screen, grid, num_rows, num_cols):
    font = pygame.font.Font(None, 24) 
    for row in range(num_rows):
        for col in range(num_cols):
            x = col * GRID_SIZE
            y = row * GRID_SIZE

            # Draw walls
            if grid[row][col][0]:  # Top wall
                pygame.draw.line(screen, BLACK, (x, y), (x + GRID_SIZE, y), LINE_WIDTH)
            if grid[row][col][1]:  # Right wall
                pygame.draw.line(screen, BLACK, (x + GRID_SIZE, y), (x + GRID_SIZE, y + GRID_SIZE), LINE_WIDTH)
            if grid[row][col][2]:  # Bottom wall
                pygame.draw.line(screen, BLACK, (x, y + GRID_SIZE), (x + GRID_SIZE, y + GRID_SIZE), LINE_WIDTH)
            if grid[row][col][3]:  # Left wall
                pygame.draw.line(screen, BLACK, (x, y), (x, y + GRID_SIZE), LINE_WIDTH)

            # draw coordinates on sqaures for testing purposes
            text_surface = font.render(f"({row},{col})", True, GRAY)
            text_rect = text_surface.get_rect(center=(x + GRID_SIZE // 2, y + GRID_SIZE // 2))
            screen.blit(text_surface, text_rect)

def draw_character(screen, row, col, image):
    x = col * GRID_SIZE
    y = row * GRID_SIZE
    screen.blit(image, (x,y))

def random_move(position, grid, num_rows, num_cols):
    row, col = position
    moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # (row_delta, col_delta) for left, up, right, down

    random.shuffle(moves)
    for move in moves:
        new_row = row + move[0]
        new_col = col + move[1]
        wall_idx = moves.index(move)
        if 0 <= new_row < num_rows and 0 <= new_col < num_cols:
            if not grid[row][col][wall_idx]:
                return new_row, new_col
    return position

def compute_distances_from_zombie(zombie_pos, grid, num_rows, num_cols):
    distances = [[-1 for _ in range(num_cols)] for _ in range(num_rows)]
    queue = deque()
    queue.append(zombie_pos)
    distances[zombie_pos[0]][zombie_pos[1]] = 0

    while queue:
        row, col = queue.popleft()
        current_distance = distances[row][col]
        neighbors = get_neighbors((row,col), grid, num_rows, num_cols)
        for neighbor in neighbors:
            n_row, n_col = neighbor
            if distances[n_row][n_col] == -1: # not visited 
                distances[n_row][n_col] = current_distance + 1
                queue.append((n_row, n_col))
    return distances

def get_neighbors(position, grid, num_rows, num_cols):
    row, col = position
    neighbors = []
    moves = [(-1,0), (0,1), (1,0), (0,-1)]
    for idx, move in enumerate(moves):
        new_row = row + move[0]
        new_col = col + move[1]
        if 0 <= new_row < num_rows and 0 <= new_col < num_cols:
            if not grid[row][col][idx]:
                neighbors.append((new_row, new_col))
    return neighbors

def find_furthest_cells(distances):
    max_distance = max(max(row) for row in distances)
    furthest_cells = []
    for row in range(len(distances)):
        for col in range(len(distances[0])):
            if distances[row][col] == max_distance:
                furthest_cells.append((row, col))
    return furthest_cells

def a_star(start, goal, grid, num_rows, num_cols):
    open_set = []
    heapq.heappush(open_set, (0 + manhattan_distance(start,goal), 0, start))
    came_from = {}
    g_score = {start: 0}
    closed_set = set()

    while open_set:
        _, current_g, current = heapq.heappop(open_set)
        if current == goal:
            # reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path # returns path from start to goal

        if current in closed_set:
            continue
        closed_set.add(current)

        neighbors = get_neighbors(current, grid, num_rows, num_cols)
        for neighbor in neighbors:
            tentative_g = g_score[current] + 1
            if neighbor in closed_set:
                continue
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + manhattan_distance(neighbor, goal)
                heapq.heappush(open_set, (f_score, tentative_g, neighbor))

    return None # no path found

def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

# Define the five game boards
def board_1():
    num_rows, num_cols = 5, 5
    # Initialize grid with no walls
    grid = [[[False, False, False, False] for _ in range(num_cols)] for _ in range(num_rows)]
    return grid, num_rows, num_cols

def board_2():
    num_rows, num_cols = 8, 8
    # Initialize grid with no walls
    grid = [[[False, False, False, False] for _ in range(num_cols)] for _ in range(num_rows)]

    # Make line on bottom of sqaure from top left to one before top right
    for col in range(0,4):
        grid[0][col][2] = True

    for row in range(1,4):
        grid[row][4][3] = True

    return grid, num_rows, num_cols

def board_3():
    num_rows, num_cols = 10, 10
    # Initialize grid with no walls
    grid = [[[False, False, False, False] for _ in range(num_cols)] for _ in range(num_rows)]

    return grid, num_rows, num_cols

def board_4():
    num_rows, num_cols = 15, 15
    # Initialize grid with no walls
    grid = [[[False, False, False, False] for _ in range(num_cols)] for _ in range(num_rows)]

    # Example: Creating a spiral maze

    for layer in range(7):
        # Top edge
        for col in range(layer, num_cols - layer):
            grid[layer][col][0] = True  # Add top wall
        # Right edge
        for row in range(layer, num_rows - layer):
            grid[row][num_cols - layer - 1][1] = True  # Add right wall
        # Bottom edge
        for col in range(layer, num_cols - layer):
            grid[num_rows - layer - 1][col][2] = True  # Add bottom wall
        # Left edge
        for row in range(layer, num_rows - layer):
            grid[row][layer][3] = True  # Add left wall

    # Create entry and exit points
    grid[0][0][3] = False  # Entry at left wall of cell (0, 0)
    grid[num_rows - 1][num_cols - 1][1] = False  # Exit at right wall of cell (14, 14)

    return grid, num_rows, num_cols

def board_5():
    num_rows, num_cols = 20, 20
    # Initialize grid with no walls
    grid = [[[False, False, False, False] for _ in range(num_cols)] for _ in range(num_rows)]

    return grid, num_rows, num_cols

human_image = pygame.image.load('old_man.png')
zombie_image = pygame.image.load('zombie_2.png')

#scale images
human_image = pygame.transform.scale(human_image, (GRID_SIZE,GRID_SIZE))
zombie_image = pygame.transform.scale(zombie_image, (GRID_SIZE,GRID_SIZE))

# Main function
def main():
    if len(sys.argv) != 2:
        print("Usage: python maze_game.py <board_number>")
        sys.exit(1)

    try:
        board_number = int(sys.argv[1])
    except ValueError:
        print("Please provide a valid board number (1-5).")
        sys.exit(1)

    if board_number == 1:
        grid, num_rows, num_cols = board_1()
    elif board_number == 2:
        grid, num_rows, num_cols = board_2()
    elif board_number == 3:
        grid, num_rows, num_cols = board_3()
    elif board_number == 4:
        grid, num_rows, num_cols = board_4()
    elif board_number == 5:
        grid, num_rows, num_cols = board_5()
    else:
        print("Board number must be between 1 and 5.")
        sys.exit(1)

    screen = setup_screen(num_rows, num_cols)
    clock = pygame.time.Clock()

    human_pos = (0,0)
    zombie_pos = (num_rows - 1,num_cols - 1)

    running = True
    human_path = []
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
        # compute human distance from zombie
        distances = compute_distances_from_zombie(zombie_pos, grid, num_rows, num_cols)

        # find furthest reachable cell from zombie 
        furthest_cells = find_furthest_cells(distances)

        # Choose one of the furthest cells randomly
        goal_cell = random.choice(furthest_cells)

        # Find path for human using A* + manhattan distance 
        human_path = a_star(human_pos, goal_cell, grid, num_rows, num_cols)
        if not human_path:
            human_path = [human_pos] # stays in place when no move

        # move human along its calculated path
        if len(human_path) > 1:
            human_pos = human_path[1]
        
        # move zombie randomly 
        zombie_pos = random_move(zombie_pos, grid, num_rows, num_cols)

        # draw and animate game
        screen.fill(WHITE)
        draw_maze(screen, grid, num_rows, num_cols)
        draw_character(screen,human_pos[0],human_pos[1], human_image)
        draw_character(screen,zombie_pos[0],zombie_pos[1], zombie_image)
        pygame.display.flip()
        clock.tick(10)

        # check for collision
        if human_pos == zombie_pos:
            print("Zombie caught the Human! GAME OVER")
            running = False

    pygame.quit()

if __name__ == "__main__":
    main()