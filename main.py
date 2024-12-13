#This file might not be used for gym, but it is used for qlearning
from env import MazeEnv
from qlearning import QLearningAgent
import time
import numpy as np
import matplotlib.pyplot as plt
import os

#this will plot the agent training parameters
#reward: # of rewards per episode
#steps: # of steps per episode
#epsilon: exploration rate
#episodes: # of episodes total

def plot_training(reward, steps, epsilon, episodes):
    plt.figure(figsize=(18, 6)) #feel free to change this if it is too small or too big. i just put random numbers

    #plotting reward
    plt.subplot(1, 3, 1)
    plt.plot(reward, label='Total Reward')
    window_size = 50 #this is the window size for the moving average

    if len(reward) > window_size:
        moving_avg = np.convolve(reward, np.ones(window_size), mode='valid')
        plt.plot(range(window_size - 1, len(reward)), moving_avg, label = 'Moving Average', color='red')

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode Rewards Over Time')
    plt.legend()

    #plotting steps per episode
    plt.subplot(1, 3, 2)
    plt.plot(steps, label='Steps per Episode', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Steps Taken per Episode')
    plt.legend()

    #plotting epsilon decay over episodes
    plt.subplot(1, 3, 3)
    plt.plot(epsilon, label='Epsilon (Exploration Rate)', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay Over Episodes')
    plt.legend()

    plt.tight_layout() #this will make sure the plots don't overlap
    plt.show()

#running three boards and returning the metrics. returns a dictionary of rewards, steps, and epsilon per episode
def test_agent(board_num, episodes, render_frequency, agent, pretrained_model=None):
    print(f"Running board {board_num}...")
    env = MazeEnv(board_number=board_num)

    if pretrained_model:
        agent.load(pretrained_model) # this is going to load the model from episode 1000 of 5x5 and throw it at a 20x20

    rewards, steps, epsilon = [], [], []

    try:
        for episode in range(episodes):
            state = env.reset() # reset env at the start of each epi
            done = False
            total_reward = 0
            total_steps = 0

            while not done:
                if episode % render_frequency == 0:
                    env.render() #render env
                    time.sleep(0.1)

                action = agent.get_action(state)
                next_state, reward, done, info = env.step(action)
                agent.update(state, action, reward, next_state, done)
                state = next_state

                total_reward += reward #accumulate rewards
                total_steps += 1 #accumulate steps

                if done and episode % render_frequency == 0:
                    env._game_over_screen()
                    break
                if done and episode % render_frequency != 0:
                    break

            #record the rewards, steps, and epsilon
            rewards.append(total_reward)
            steps.append(total_steps)
            epsilon.append(agent.get_stats()['epsilon'])

            if episode % 10 == 0: #print metrics every 10 episodes
                avg_reward = np.mean(rewards[-10:])
                avg_steps = np.mean(steps[-10:])
                avg_epsilon = np.mean(epsilon[-10:])

                print(f"\nBoard (board_num), Episode {episode}/{episodes}")
                print(f"Average Reward (last 10): {avg_reward:.2f}")
                print(f"Average Steps (last 10): {avg_steps:.2f}")
                print(f"Average Epsilon (last 10): {avg_epsilon:.2f}")
                print(f"Learning Rate: {agent.get_stats()['alpha']:.2f}")
                print(f"Q-table size: {agent.get_stats()['q_table_size']}")

            #save model every 100 episodes
            if episode % 100 == 0:
                agent.save(os.path.join(os.path.dirname(__file__), 'models', f'zombie_agent_ep{episode}.pkl'))

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
        agent.save('zombie_agent_interrupted.pkl')

    finally:
        env.close()

    return {'rewards': rewards, 'steps': steps, 'epsilon': epsilon}

#this loads in a model to make sure the plotting works without me having to wait forever
def test_plot(board_num:int, episodes=100, pretrained_model=900):
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    pretrained_path = os.path.join(model_dir, f'zombie_agent_ep{pretrained_model}.pkl')

    if not os.path.exists(pretrained_path):
        print(f"Model {pretrained_path} does not exist")
        return
    
    env = MazeEnv(board_number=board_num)
    agent = QLearningAgent(env.observation_space.shape[0], env.action_space.n)

    agent.load(pretrained_path)

    rewards, steps, epsilon = [], [], []

    try:
        for episode in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0
            total_steps = 0

            while not done:
                action = agent.get_action(state)
                next_state, reward, done, info = env.step(action)
                state = next_state

                total_reward += reward
                total_steps += 1

            rewards.append(total_reward)
            steps.append(total_steps)
            epsilon.append(agent.get_stats()['epsilon'])
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
        
    finally:
        env.close()

    plot_training(rewards, steps, epsilon, episodes)

# All of this can be changed, just use the same constructs that are here for further testing
def main():

    #call to the test_plot function to test the plotting
    quick_test = False

    if quick_test:
        test_plot(1)
        return

    print("Select a board to test the agent on (enter the corresponding number):")
    print("The boards grow in complexity in higher numbers.")
    print("1. 5x5 board  -- A small empty playing resulting in quicker trials")
    print("2. 8x8 board  -- A board with two simple hallways near each starting point")
    print("3. 8x8 board -- A board with sporadic walls and hallways")
    print("4. 5x5 board  -- Small board with intricate maze system discourages zombie from learning fast")
    print("5. 10x10 board  -- The most complex board that has empty passages along outer ring")

    try:
        board_num = int(input())
    except ValueError:
        print("Invalid input. Please enter a number between 1 and 5.")
        return
    
    #map input to board number
    if board_num not in [1, 2, 3, 4, 5]:
        print("Invalid input. Please enter a number between 1 and 5.")
        return
        
    episodes = 1000

    render_frequency = 50

    env = MazeEnv(board_number=board_num)
    agent = QLearningAgent(env.observation_space.shape[0], env.action_space.n)

    print("Training agent...")
    metrics = test_agent(board_num, episodes, render_frequency, agent)

    plot_training(metrics['rewards'], metrics['steps'], metrics['epsilon'], episodes)

# Main, obv
if __name__ == "__main__":
    main()