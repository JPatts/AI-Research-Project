import gymnasium as gym 
from gymnasium import spaces
import numpy as np
import random
import pygame
import heapq
from collections import deque
import sys

class MazeEnv(gym.Env):
    metadata = {'render.modes': ['human'], "render_fps": 4}

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
        self.observation_space = spaces.Box(low=0, high=max(self.num_rows, self.num_cols), shape=(obs_size,), dtype=np.float32)

        # Initializing pygame for rendering the environment
        pygame.init()
        self.screen = None
        self.human_image = pygame.transform.scale(pygame.image.load('old_man.png'), (self.GRID_SIZE, self.GRID_SIZE))
        self.zombie_image = pygame.transform.scale(pygame.image.load('zombie.png'), (self.GRID_SIZE, self.GRID_SIZE))

        # Reset initailizes the environment
        self.reset()


    def reset(self):
        self.human_pos = (0, 0)
        self.zombie_pos = (self.num_rows - 1, self.num_cols - 1)
        return self._get_obs()
    
    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action {action}"
        # This will update the zombie position
        prev_distance = self._manhattan_distance(self.zombie_pos, self.human_pos)
        self.zombie_pos = self._get_new_position(self.zombie_pos, action)

        # This will update the human position
        self._move_human()

        # Moving the zombie and maybe giving a reward
        new_distance = self._manhattan_distance(self.zombie_pos, self.human_pos)
        #reward = prev_distance - new_distance # Giving a reward if the distance decreases
        max_distance = self.num_rows + self.num_cols - 2

        if new_distance == 1:
            reward = 50 # If the agent gets very close, they will get a big reward
        elif new_distance == 2:
            reward = 20
        elif new_distance == max_distance - 1:
            reward = -10 # Discouraging the agent from being at the max distance away
        else:
            if new_distance < prev_distance:
                reward = 1
            else:
                reward = -1
        
        done = self.zombie_pos == self.human_pos
        info = {
            'distance_to_human' : self._manhattan_distance(self.zombie_pos, self.human_pos),
            'human_pos' : self.human_pos,
            'zombie_pos' : self.zombie_pos
        }
        return self._get_obs(), reward, done, info
    
    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        return [seed]

    def _get_obs(self): 
        grid_flat = np.array(self.grid).flatten()
        return np.concatenate([
            np.array(self.human_pos),
            np.array(self.zombie_pos),
            grid_flat
        ]).astype(np.float32)
    
    def render(self, mode='human'):
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.num_cols * self.GRID_SIZE, self.num_rows * self.GRID_SIZE))
        
        self.screen.fill(self.WHITE)
        self._draw_maze()
        human_row, human_col = self.human_pos
        zombie_row, zombie_col = self.zombie_pos
        self._draw_character(human_row, human_col, self.human_image)  
        self._draw_character(zombie_row, zombie_col, self.zombie_image)
        pygame.display.flip()

    def close(self):
        if self.screen is not None: 
            pygame.quit()
    
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

    def _load_board(self, board_number):
        if board_number == 1:
            num_rows, num_cols = 5, 5
            grid = [[[False, False, False, False] for _ in range(num_cols)] for _ in range(num_rows)]
            
        elif board_number == 2:
            num_rows, num_cols = 8, 8
            grid = [[[False, False, False, False] for _ in range(num_cols)] for _ in range(num_rows)]
            # Add walls for board 2
            for col in range(0,4):
                grid[0][col][2] = True
            for row in range(1,4):
                grid[row][4][3] = True
                
        elif board_number == 3:
            num_rows, num_cols = 10, 10
            grid = [[[False, False, False, False] for _ in range(num_cols)] for _ in range(num_rows)]
            
        elif board_number == 4:
            num_rows, num_cols = 15, 15
            grid = [[[False, False, False, False] for _ in range(num_cols)] for _ in range(num_rows)]
            # Create spiral maze
            for layer in range(7):
                # Top edge
                for col in range(layer, num_cols - layer):
                    grid[layer][col][0] = True
                # Right edge
                for row in range(layer, num_rows - layer):
                    grid[row][num_cols - layer - 1][1] = True
                # Bottom edge
                for col in range(layer, num_cols - layer):
                    grid[num_rows - layer - 1][col][2] = True
                # Left edge
                for row in range(layer, num_rows - layer):
                    grid[row][layer][3] = True
            # Create entry/exit points
            grid[0][0][3] = False
            grid[num_rows - 1][num_cols - 1][1] = False
            
        elif board_number == 5:
            num_rows, num_cols = 20, 20
            grid = [[[False, False, False, False] for _ in range(num_cols)] for _ in range(num_rows)]
            
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

            neighbors = self._get_neighbors(current)
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

    def _get_neighbors(self, position):
        row, col = position
        neighbors = []
        moves = [(-1,0), (0,1), (1,0), (0,-1)]
        for idx, move in enumerate(moves):
            new_row = row + move[0]
            new_col = col + move[1]
            if 0 <= new_row < self.num_rows and 0 <= new_col < self.num_cols:
                if not self.grid[row][col][idx]:
                    neighbors.append((new_row, new_col))
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
        if action == 0: # up
            row -= 1
        elif action == 1: # right
            col += 1
        elif action == 2: # down
            row += 1
        elif action == 3: # left
            col -= 1
        return (row, col) if 0 <= row < self.num_rows and 0 <= col < self.num_cols else position

    def _compute_distance_from_zombie(self):
        distances = [[-1 for _ in range(self.num_cols)] for _ in range(self.num_rows)]
        queue = deque()
        queue.append(self.zombie_pos)
        distances[self.zombie_pos[0]][self.zombie_pos[1]] = 0

        while queue:
            row, col = queue.popleft()
            current_distance = distances[row][col]
            neighbors = self._get_neighbors((row,col))
            for neighbor in neighbors:
                n_row, n_col = neighbor
                if distances[n_row][n_col] == -1: # not visited 
                    distances[n_row][n_col] = current_distance + 1
                    queue.append((n_row, n_col))
        return distances

    def _move_human(self):
        distance = self._compute_distance_from_zombie()

        max_distance = max(max(row) for row in distance)
        furthest_cells = [
            (row, col)
            for row in range(self.num_rows)
            for col in range(self.num_cols)
            if distance[row][col] == max_distance
        ]
        goal = random.choice(furthest_cells)
        path = self._a_star(self.human_pos, goal)

        if path and len(path) > 1:
            self.human_pos = path[1]