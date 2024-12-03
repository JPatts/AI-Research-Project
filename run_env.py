import gymnasium as gym
from main_env import MazeEnv
import time
import numpy as np
import random
import register_env

class QLeaningAgent:
    def __init__(self, state_size, action_size):
        self.q_table = {}
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.alpha = 0.1
        self.gamma = 0.95
        self.action_size = action_size

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        return max(range(self.action_size), key=lambda a: self.q_table.get((str(state), a), 0.0))
    
    def update(self, state, action, reward, next_state, done):
        current_q = self.q_table.get((str(state), action), 0.0)
        next_max_q = max([self.q_table.get((str(next_state), a), 0.0) for a in range(self.action_size)])
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[(str(state), action)] = new_q
        if done and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        

def main():
    env = MazeEnv(board_number=1)
    agent = QLeaningAgent(env.observation_space.shape[0], env.action_space.n)


    for episode in range(3):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            env.render()

            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            total_reward += reward

            print(f"Action: {action}, Reward: {reward}")
            print(f"Distance to human: {info['distance_to_human']} ")
            print(f"Current award amount: {total_reward}")

            time.sleep(0.5)
            steps += 1
        

        print(f"Episode {episode} finished with total reward {total_reward}")
        print(f"Number of steps for zombie to win: {steps}")

    env.close()

def run_gym():
    env = gym.make('MazeEnv-v0')
    env = gym.wrappers.Monitor(env, "recordings", force=True)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    episodes = 1000
    for episode in range(episodes):
        state = env.reset()
        dont = False
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            if episode % 100 == 0:
                env.render()
    
    env.close()

if __name__ == "__main__":
    
    main()
    #run_gym()