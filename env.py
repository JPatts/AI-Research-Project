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

        # load all background images 
        self.background_images = {}
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                bg_path = self.grid[row][col]['background']
                if bg_path not in self.background_images:
                    self.background_images[bg_path] = pygame.transform.scale(
                        pygame.image.load(bg_path), (self.GRID_SIZE, self.GRID_SIZE)
                    )

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

        self.human_images = []
        for i in range(1,7):
            img = pygame.image.load(os.path.join('assets/human_images',f'human_{i}.png'))
            img = pygame.transform.scale(img, (self.GRID_SIZE, self.GRID_SIZE))
            self.human_images.append(img)
        
        self.zombie_images = []
        for i in range(1,7):
            img = pygame.image.load(os.path.join('assets/zombie_images',f'zombie_{i}.png'))
            img = pygame.transform.scale(img, (self.GRID_SIZE, self.GRID_SIZE))
            self.zombie_images.append(img)

        self.human_dead_image = pygame.transform.scale(
            pygame.image.load(os.path.join('assets/human_images', 'human_dead.png')), (self.GRID_SIZE, self.GRID_SIZE))

        self.frame_count = 0
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
        # this function is updated since wall data and background images are now a dictionary which cannot be converterd numerically
        # this function now extracts only the wall data out of this dictionary so it can be used like before 
        # old function:

        # Extract wall information into a numeric array
        wall_data = []
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                walls = self.grid[row][col]['walls']
                # Convert booleans to 0/1
                wall_data.extend([1 if w else 0 for w in walls])

        # Convert positions and wall data to float32
        human_pos_array = np.array(self.human_pos, dtype=np.float32)
        zombie_pos_array = np.array(self.zombie_pos, dtype=np.float32)
        walls_array = np.array(wall_data, dtype=np.float32)

        # Concatenate them
        return np.concatenate([human_pos_array, zombie_pos_array, walls_array])
 
    # Rendering the environment itself
    def render(self, mode='human'):
        self.frame_count += 1

        if self.screen is None:
            self.screen = pygame.display.set_mode((self.num_cols * self.GRID_SIZE, self.num_rows * self.GRID_SIZE))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill(self.WHITE)
        self._draw_maze()

        # This function will highlight the A* path and goal if they exist
        self._highlight_human_path()

        # determine frame count for human and zombie
        human_frame = self.frame_count % len(self.human_images)
        zombie_frame = self.frame_count % len(self.zombie_images)

        # Draw characters
        human_row, human_col = self.human_pos
        zombie_row, zombie_col = self.zombie_pos

        self._draw_character(human_row, human_col, self.human_images[human_frame])  
        self._draw_character(zombie_row, zombie_col, self.zombie_images[zombie_frame])

        pygame.display.flip()

    def _game_over_screen(self):
        # Draw the final state with the human lying down
        human_row, human_col = self.human_pos
        zombie_row, zombie_col = self.zombie_pos

        self.screen.fill(self.WHITE)
        self._draw_maze()

        # Draw the human lying down at their last position
        self._draw_character(human_row, human_col, self.human_dead_image)

        # Move the zombie away from the human position
        escape_path = self._find_zombie_escape_path(human_row, human_col)

        for (z_row, z_col) in escape_path:
            # Draw updated maze and positions
            self.screen.fill(self.WHITE)
            self._draw_maze()

            # Draw the human lying down
            self._draw_character(human_row, human_col, self.human_dead_image)

            # Draw the zombie in its new position
            self._draw_character(z_row, z_col, self.zombie_images[self.frame_count % len(self.zombie_images)])

            pygame.display.flip()
            #pygame.time.delay(200)  # Delay to simulate animation
            pygame.time.Clock().tick(5)

        # Optionally fade out or display "Game Over" text
        font = pygame.font.Font(None, 72)
        text_surface = font.render("Game Over", True, (255, 0, 0))
        text_rect = text_surface.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() // 2))
        self.screen.blit(text_surface, text_rect)
        pygame.display.flip()

    def _highlight_human_path(self):
        if hasattr(self, 'human_path') and self.human_path and hasattr(self, 'human_goal') and self.human_goal:
            # create transparent overlay 
            overlay = pygame.Surface((self.num_cols * self.GRID_SIZE, self.num_rows * self.GRID_SIZE), pygame.SRCALPHA)

            # RGBA format (R, G, B, A) where A is alpha transparency (0-255)
            path_color = (255, 0, 0, 255)

            # make highlighted sqaures smaller
            square_scale = 0.2
            sqaure_size = int(self.GRID_SIZE * square_scale)
            square_offset = (self.GRID_SIZE - sqaure_size) // 2

            # draw path cells
            for(r,c) in self.human_path:
                x = c * self.GRID_SIZE + square_offset
                y = r * self.GRID_SIZE + square_offset
                pygame.draw.rect(overlay, path_color, (x,y,sqaure_size,sqaure_size))

            # Make goal cell blink 
            current_time = pygame.time.get_ticks()
            blink_on = (current_time // 50) % 2 == 0  # Blink every 50 ms
            
            if blink_on:
                goal_color = (255, 248, 20, 255)
            else:
                goal_color = (255, 255, 255, 255)

            goal_r, goal_c = self.human_goal
            gx = goal_c * self.GRID_SIZE + square_offset
            gy = goal_r * self.GRID_SIZE + square_offset
            pygame.draw.rect(overlay, goal_color, (gx, gy, sqaure_size, sqaure_size))

            # blit overlay onto screen
            self.screen.blit(overlay, (0,0))

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
            
    def _draw_maze(self):
        font = pygame.font.Font(None, 24) 
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                x = col * self.GRID_SIZE
                y = row * self.GRID_SIZE
                
                # draw background background images before walls
                bg_path = self.grid[row][col]['background']
                bg_image = self.background_images[bg_path]
                self.screen.blit(bg_image, (x,y))

                # Draw walls
                walls = self.grid[row][col]['walls']
                if walls[0]:  # Top wall
                    pygame.draw.line(self.screen, self.BLACK, (x, y), (x + self.GRID_SIZE, y), self.LINE_WIDTH)
                if walls[1]:  # Right wall
                    pygame.draw.line(self.screen, self.BLACK, (x + self.GRID_SIZE, y), (x + self.GRID_SIZE, y + self.GRID_SIZE), self.LINE_WIDTH)
                if walls[2]:  # Bottom wall
                    pygame.draw.line(self.screen, self.BLACK, (x, y + self.GRID_SIZE), (x + self.GRID_SIZE, y + self.GRID_SIZE), self.LINE_WIDTH)
                if walls[3]:  # Left wall
                    pygame.draw.line(self.screen, self.BLACK, (x, y), (x, y + self.GRID_SIZE), self.LINE_WIDTH)

                # DO NOT DELETE COMEMENTED CODE BELOW
                # draws coordinates on sqaures for testing purposes -- gets commented out for end visuals 
                """
                text_surface = font.render(f"({row},{col})", True, self.GRAY)
                text_rect = text_surface.get_rect(center=(x + self.GRID_SIZE // 2, y + self.GRID_SIZE // 2))
                self.screen.blit(text_surface, text_rect) """

    # Loads the board configuration. There are 5 different choices here
    def _load_board(self, board_number): 
        # Plain 5x5 board
        if board_number == 1:
            num_rows, num_cols = 5, 5
            grid = [[{'walls': [False, False, False, False], 'background': 'assets/background_images/grass_patch_1.png'} for _ in range(num_cols)] for _ in range(num_rows)]
            
        # A 8x8 board with two walls
        elif board_number == 2:
            num_rows, num_cols = 8, 8
            grid = [[{'walls': [False, False, False, False], 'background': 'assets/background_images/grass_patch_1.png'} for _ in range(num_cols)] for _ in range(num_rows)]
            # Add walls for board 2
            # 0: top wall, 1: Right wall, 2; Bottom wall, 3: Left wall

            for col in range(0,6):
                grid[1][col]['walls'][2] = True # add bottom wall to grid point (0,0)
                grid[2][col]['walls'][0] = True # add top wall to grid point (0,1)
            
            for col in range(2,8):
                grid[5][col]['walls'][2] = True # add bottom wall to grid point (0,0)
                grid[6][col]['walls'][0] = True # add top wall to grid point (0,1)

        # sporadic 10x10 board
        elif board_number == 3: 
            num_rows, num_cols = 8, 8
            grid = [[{'walls': [True, True, True, True], 'background': 'assets/background_images/grass_patch_1.png'} for _ in range(num_cols)] for _ in range(num_rows)]

            self.grid = grid
            self.num_rows = num_rows
            self.num_cols = num_cols

            self._carve_maze(0,0)
            self._carve_maze(0,num_cols - 1)

        # A 5x5 board with more complex walls 
        elif board_number == 4:
            num_rows, num_cols = 5, 5
            grid = [[{'walls': [True, True, True, True], 'background': 'assets/background_images/grass_patch_1.png'} for _ in range(num_cols)] for _ in range(num_rows)]
            
            self.grid = grid
            self.num_rows = num_rows
            self.num_cols = num_cols

            self._carve_maze(num_rows - 1,num_cols - 1)
        
        # intricate 10x10 board 
        elif board_number == 5:
            num_rows, num_cols = 9, 9
            grid = [[{'walls': [True, True, True, True], 'background': 'assets/background_images/grass_patch_1.png'} for _ in range(num_cols)] for _ in range(num_rows)]
            self.grid = grid
            self.num_rows = num_rows
            self.num_cols = num_cols

            self._carve_maze(0,0)

           
            # removes all out walls 
            for r in range(num_rows):
                grid[r][0]['walls'][2] = False
                grid[r][0]['walls'][0] = False
                grid[r][num_cols - 1]['walls'][2] = False
                grid[r][num_cols - 1]['walls'][0] = False
            
            for c in range(num_cols):
                grid[0][c]['walls'][3] = False
                grid[0][c]['walls'][1] = False
                grid[num_rows - 1][c]['walls'][3] = False
                grid[num_rows - 1][c]['walls'][1] = False

            # remove middle row and middle col walls to make + path
            for r in range(num_rows):
                grid[r][4]['walls'][2] = False
                grid[r][4]['walls'][0] = False
            
            for c in range(num_cols):
                grid[4][c]['walls'][3] = False
                grid[4][c]['walls'][1] = False

        else:
            raise ValueError("Board number must be between 1 and 5")

        return grid, num_rows, num_cols    

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

        walls = self.grid[row][col]['walls']
        # Move up if no top wall
        if row > 0 and not walls[0]:
            neighbors.append((row - 1, col))

        # Move right if no right wall
        if col < self.num_cols - 1 and not walls[1]:
            neighbors.append((row, col + 1))

        # Move down if no bottom wall
        if row < self.num_rows - 1 and not walls[2]:
            neighbors.append((row + 1, col))
    
        # Move left if no left wall
        if col > 0 and not walls[3]:
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

        walls = self.grid[row][col]['walls']

        if action == 0 and row > 0 and not walls[0]:
                row -= 1
        elif action == 1 and col < self.num_cols - 1 and not walls[1]:
                col += 1
        elif action == 2 and row < self.num_rows - 1 and not walls[2]:
                row += 1
        elif action == 3 and col > 0 and not walls[3]:
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
            # store the path and goal for path and goal overlay
            self.human_goal= goal
            self.human_path = path
            self.human_pos = path[1]
        else:
            # if no path found
            self.human_goal = None
            self.human_path = None 
    
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
    
    def _find_zombie_escape_path(self, human_row, human_col):
        # move zombie away from human
        corners = [(0,0),(0, self.num_cols -1),(self.num_rows - 1, 0),(self.num_rows - 1, self.num_cols -1)]
        escape_corner = max(corners, key=lambda corner: self._manhattan_distance(corner, (human_row,human_col)))

        # use A* to get path
        path = self._a_star(self.zombie_pos, escape_corner)
        return path if path else [self.zombie_pos]
    
    def _carve_maze(self, start_r, start_c):
        directions = [
            (-1, 0, 0, 2),  # Up
            (0, 1, 1, 3),   # Right
            (1, 0, 2, 0),   # Down
            (0, -1, 3, 1)   # Left
            ]

        random.seed(42)  # Fixed seed allows for recreation

        # Depth First Search based maze-carving
        stack = [(start_r, start_c)]
        visited = set()
        visited.add((start_r, start_c))

        while stack:
            r, c = stack[-1]
            # Try directions in random order
            random.shuffle(directions)
            carved = False
            for dr, dc, wall_curr, wall_next in directions:
                nr, nc = r + dr, c + dc
                if self._in_bounds(nr, nc) and (nr, nc) not in visited:
                    # Remove walls between (r,c) and (nr,nc)
                    self.grid[r][c]['walls'][wall_curr] = False
                    self.grid[nr][nc]['walls'][wall_next] = False
                    visited.add((nr, nc))
                    stack.append((nr, nc))
                    carved = True
                    break
            if not carved:
                stack.pop()

    def _in_bounds(self, r, c):
        return 0 <= r < self.num_rows and 0 <= c < self.num_cols