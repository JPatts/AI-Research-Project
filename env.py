import gymnasium as gym 
from gymnasium import spaces
import numpy as np
import random
import pygame
import heapq
from collections import deque
import sys
import os

class MazeEnv(gym.Env):
    metadata = {'render.modes': ['human'], "render_fps": 4}

    # Init will load a board passed into the constructor, otherwise it will be the 
    # simplest board loaded. This creates the environment and loads the images
    # used for the zombie and human. 
    def __init__(self, board_number=1):
        super(MazeEnv, self).__init__()


        # Helpful constants
        self.GRID_SIZE = 60
        self.LINE_WIDTH = 3
        self.WHITE = (255, 255, 255)
        self.GRAY = (200, 200, 200)
        self.BLACK = (0, 0, 0)

        # This will load the board configuration
        self.board_number = board_number
        self.grid, self.num_rows, self.num_cols = self._load_board(board_number)

        # Define the action and observation space
        self.action_space = spaces.Discrete(4) # up, down, left, right
        obs_size = 4 + (self.num_rows * self.num_cols * 4)
        self.observation_space = spaces.Box(
            low=0, 
            high=max(self.num_rows, self.num_cols), 
            shape=(obs_size,), 
            dtype=np.float32
        )

        # Initializing pygame for rendering the environment and pointing to the assets for the zombie
        # and the human
        pygame.init()
        self.screen = None
        self.human_image = pygame.transform.scale(
            pygame.image.load(os.path.join('assets', 'old_man.png')), 
            (self.GRID_SIZE, self.GRID_SIZE)
        )
        self.zombie_image = pygame.transform.scale(
            pygame.image.load(os.path.join('assets', 'zombie.png')), 
            (self.GRID_SIZE, self.GRID_SIZE)
        )

        # Reset initailizes the environment and puts the zombie and human at opposite corners
        self.reset()


    # Used at the begining of every episode to set the humand and zombie at the 
    # opposite corners of the env
    def reset(self):
        self.human_pos = (0, 0) # top left
        self.zombie_pos = (self.num_rows - 1, self.num_cols - 1) # bottom right
        return self._get_obs()
    
    # Step is the function that will move both the human and the zombie through
    # the environment. Step refers to every step the humand and zombie makes. This 
    # function is also where the reward system is applied to the zombie 
    def step(self, action):
        # Asserting a valid action can be made
        assert self.action_space.contains(action), f"Invalid action {action}"
        # Getting the previous distance for use in the reward system
        prev_distance = self._manhattan_distance(self.zombie_pos, self.human_pos)
        steps_taken = getattr(self, 'steps_taken', 0) + 1 # Also used in reward system
        setattr(self, 'steps_taken', steps_taken)

        # This will update the zombie position
        self.zombie_pos = self._get_new_position(self.zombie_pos, action)
        # This will update the human position
        self._move_human()
        # Using the manhattan distance to find the distance between the zombie and human
        new_distance = self._manhattan_distance(self.zombie_pos, self.human_pos)

        # Reward system
        max_distance = self.num_rows + self.num_cols - 2 # Maximum distance of the env
        distance_ratio = new_distance / max_distance # Ratio of the distance to the max distance
        time_penalty = -0.1 * (steps_taken / 100) # Time penalty for wandering 

        if self.zombie_pos == self.human_pos: # Very large reward for capture
            reward = 100 + (50 * (1 - steps_taken / 200))
        elif new_distance == 1: # Big reward if zombie is close
            reward = 25 * (1 - distance_ratio) 
        elif new_distance == 2: # Reward for getting closer
            reward = 10 * (prev_distance - new_distance) / max_distance
        elif new_distance == max_distance - 1: # Discouraging zombie from being the max distance away
            reward = -15 * (new_distance - prev_distance) / max_distance 
        else:
            reward = -5 # This should penalize wandering

        reward += time_penalty # Apply time penalty
        reward = max(-50, min(100, reward)) # This will clip rewards

        done = self.zombie_pos == self.human_pos 
        # Info will be used for data calculation within run_env.py
        info = {
            'distance_to_human' : self._manhattan_distance(self.zombie_pos, self.human_pos),
            'human_pos' : self.human_pos,
            'zombie_pos' : self.zombie_pos
        }
        return self._get_obs(), reward, done, info
    
    # Not currently used
    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        return [seed]

    # Get observation is used to get the observation of the environment for use by the zombie
    def _get_obs(self): 
        grid_flat = np.array(self.grid).flatten()
        return np.concatenate([
            np.array(self.human_pos),
            np.array(self.zombie_pos),
            grid_flat
        ]).astype(np.float32)
    
    # Rendering the environment itself
    def render(self, mode='human'):
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.num_cols * self.GRID_SIZE, self.num_rows * self.GRID_SIZE))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill(self.WHITE)
        self._draw_maze()
        human_row, human_col = self.human_pos
        zombie_row, zombie_col = self.zombie_pos
        self._draw_character(human_row, human_col, self.human_image)  
        self._draw_character(zombie_row, zombie_col, self.zombie_image)
        pygame.display.flip()

    # Used if the user clicks the x on the pygame window
    def close(self):
        if self.screen is not None: 
            pygame.quit()
    
    # Draws the zombie and the human within the grid of the environment
    def _draw_character(self, row, col, image):
        try:
            cell_size = self.GRID_SIZE

            scaled_size = int(cell_size * 0.8)
            scaled_image = pygame.transform.scale(image, (scaled_size, scaled_size))

            x = col * cell_size + (cell_size - scaled_size) // 2
            y = row * cell_size + (cell_size - scaled_size) // 2

            self.screen.blit(scaled_image, (x, y))
        except Exception as e:
            print(f"Error drawing character: {e}")
            
    # Draws the maze itself
    def _draw_maze(self):
        font = pygame.font.Font(None, 24) 
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                x = col * self.GRID_SIZE
                y = row * self.GRID_SIZE

                # Draw walls
                if self.grid[row][col][0]:  # Top wall
                    pygame.draw.line(self.screen, self.BLACK, (x, y), (x + self.GRID_SIZE, y), self.LINE_WIDTH)
                if self.grid[row][col][1]:  # Right wall
                    pygame.draw.line(self.screen, self.BLACK, (x + self.GRID_SIZE, y), (x + self.GRID_SIZE, y + self.GRID_SIZE), self.LINE_WIDTH)
                if self.grid[row][col][2]:  # Bottom wall
                    pygame.draw.line(self.screen, self.BLACK, (x, y + self.GRID_SIZE), (x + self.GRID_SIZE, y + self.GRID_SIZE), self.LINE_WIDTH)
                if self.grid[row][col][3]:  # Left wall
                    pygame.draw.line(self.screen, self.BLACK, (x, y), (x, y + self.GRID_SIZE), self.LINE_WIDTH)

                # draw coordinates on sqaures for testing purposes
                text_surface = font.render(f"({row},{col})", True, self.GRAY)
                text_rect = text_surface.get_rect(center=(x + self.GRID_SIZE // 2, y + self.GRID_SIZE // 2))
                self.screen.blit(text_surface, text_rect)

    # Loads the board configuration. There are 5 different choices here
    def _load_board(self, board_number):
        # Plain 5x5 board
        if board_number == 1:
            num_rows, num_cols = 5, 5
            grid = [[[False, False, False, False] for _ in range(num_cols)] for _ in range(num_rows)]
            
        # A 8x8 board with a wall
        elif board_number == 2:
            num_rows, num_cols = 8, 8
            grid = [[[False, False, False, False] for _ in range(num_cols)] for _ in range(num_rows)]
            # Add walls for board 2
            # 0: top wall, 1: Right wall, 2; Bottom wall, 3: Left wall

            # adds bottom wall to grid point (0,0) to (0,6)
            for col in range(6):
                grid[0][col][2] = True 
            
            # adds top wall to grid point (0,0) to (0,6)
            for col in range(6):
                grid[1][col][0] = True

        # Plain 10x10 board
        elif board_number == 3: 
            num_rows, num_cols = 10, 10
            grid = [[[False, False, False, False] for _ in range(num_cols)] for _ in range(num_rows)]

        # A 15x15 board with more complex walls 
        elif board_number == 4:
            num_rows, num_cols = 15, 15
            grid = [[[False, False, False, False] for _ in range(num_cols)] for _ in range(num_rows)]

        # Plain 20x20 board 
        elif board_number == 5:
            num_rows, num_cols = 20, 20
            grid = [[[False, False, False, False] for _ in range(num_cols)] for _ in range(num_rows)]
            
        else:
            raise ValueError("Board number must be between 1 and 5")
            
        return grid, num_rows, num_cols    

    # What else do I need to say about these two functions?
    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _a_star(self, start, goal):
        open_set = []
        heapq.heappush(open_set, (0 + self._manhattan_distance(start,goal), 0, start))
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

            current_row, current_col = current
            neighbors = self._get_neighbors(current_row,current_col)
            for neighbor in neighbors:
                tentative_g = g_score[current] + 1
                if neighbor in closed_set:
                    continue
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._manhattan_distance(neighbor, goal)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))

        return None

    def _get_neighbors(self, row, col):
        neighbors = []          
        # Directions:
        # 0: top wall, 1: Right wall, 2; Bottom wall, 3: Left wall

        # Move up if no top wall
        if row > 0 and not self.grid[row][col][0]:
            neighbors.append((row - 1, col))

        # Move right if no right wall
        if col < self.num_cols - 1 and not self.grid[row][col][1]:
            neighbors.append((row, col + 1))

        # Move down if no bottom wall
        if row < self.num_rows - 1 and not self.grid[row][col][2]:
            neighbors.append((row + 1, col))
    
        # Move left if no left wall
        if col > 0 and not self.grid[row][col][3]:
            neighbors.append((row, col - 1))

        return neighbors
    
    def _find_furthest_cells(self, distances):
        max_distance = max(max(row) for row in distances)
        furthest_cells = []
        for row in range(len(distances)):
            for col in range(len(distances[0])):
                if distances[row][col] == max_distance:
                    furthest_cells.append((row, col))
        return furthest_cells
    
    def _get_new_position(self, position, action):
        row, col = position
        
        # 0: top wall, 1: Right wall, 2; Bottom wall, 3: Left wall

        if action == 0 and row > 0 and not self.grid[row][col][0]:
                row -= 1
        elif action == 1 and col < self.num_cols - 1 and not self.grid[row][col][1]:
                col += 1
        elif action == 2 and row < self.num_rows - 1 and not self.grid[row][col][2]:
                row += 1
        elif action == 3 and col > 0 and not self.grid[row][col][3]:
                col -= 1
        return (row, col)

    def _compute_distance_from_zombie(self):
        distances = [[-1 for _ in range(self.num_cols)] for _ in range(self.num_rows)]
        queue = deque()
        queue.append(self.zombie_pos)
        distances[self.zombie_pos[0]][self.zombie_pos[1]] = 0

        while queue:
            row, col = queue.popleft()
            current_distance = distances[row][col]
            neighbors = self._get_neighbors(row, col)
            for neighbor in neighbors:
                n_row, n_col = neighbor
                if distances[n_row][n_col] == -1: # not visited 
                    distances[n_row][n_col] = current_distance + 1
                    queue.append((n_row, n_col))
        return distances

    def _move_human(self):
        # comput distance from zombie to all reachable cells 
        zombie_distances = self._compute_distance_from_zombie()

        # Find all reachable cells fo the human using BFS 
        reachable_distances = self._compute_reachable_cells(self.human_pos)

        # combine reachable cells with zombie distances
        furthest_cells = []
        max_distance = - 1

        for row in range(self.num_rows):
            for col in range(self.num_cols):
                if reachable_distances[row][col] != -1: # cell is reachable by human agent 
                    distance_from_zombie = zombie_distances[row][col]
                    if distance_from_zombie > max_distance:
                        max_distance = distance_from_zombie
                        furthest_cells = [(row, col)]
                    elif distance_from_zombie == max_distance:
                        furthest_cells.append((row,col))
        
        if not furthest_cells:
            # if no furthest cells found, do nothing
            return
    
        # randomly choose one of furthest cells found
        goal = random.choice(furthest_cells)

        # use A* to plan path to goal
        path = self._a_star(self.human_pos, goal)

        # Move to next step in path if it exists
        if path and len(path) > 1:
            self.human_pos = path[1]
    
    def _compute_reachable_cells(self, start_pos):
        # BFS from start_pos using _get_neighbors
        distances = [[-1]*self.num_cols for _ in range(self.num_rows)]
        queue = deque([start_pos])
        distances[start_pos[0]][start_pos[1]] = 0

        while queue:
            r, c = queue.popleft()
            for nr, nc in self._get_neighbors(r,c):
                if distances[nr][nc] == -1: # not visited 
                    distances[nr][nc] = distances[r][c] + 1
                    queue.append((nr,nc))
        
        return distances