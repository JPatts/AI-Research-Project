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

def plot_training(board, reward, steps, epsilon, episodes):
    plt.figure(figsize=(18, 6)) 

    if not os.path.exists('plots'):
        os.makedirs('plots')

    # plotting reward
    plt.subplot(1, 3, 1)
    plt.plot(reward, label='Total Reward')
    window_size = 50 # this is the window size for the moving average

    if len(reward) > window_size:
        moving_avg = np.convolve(reward, np.ones(window_size), mode='valid')
        plt.plot(range(window_size - 1, len(reward)), moving_avg, label = 'Moving Average', color='red')

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode Rewards Over Time')
    plt.legend()

    # plotting steps per episode
    plt.subplot(1, 3, 2)
    plt.plot(steps, label='Steps per Episode', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Steps Taken per Episode')
    plt.legend()

    # plotting epsilon decay over episodes
    plt.subplot(1, 3, 3)
    plt.plot(epsilon, label='Epsilon (Exploration Rate)', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay Over Episodes')
    plt.legend()

    plt.tight_layout() # this will make sure the plots don't overlap

    filename = f'Board_{board}_Training_Data.png'
    filepath = os.path.join(os.path.dirname(__file__), 'plots', filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()



# Returns a dictionary of rewards, steps, and epsilon per episode
def test_agent(board_num, episodes, render_frequency, agent, pretrained_model=None):
    print(f"Running board {board_num}...")
    env = MazeEnv(board_number=board_num)

    if pretrained_model:
        agent.load(pretrained_model) # this is going to load a saved model if one exists

    rewards, steps, epsilon = [], [], []

    try:
        for episode in range(episodes):
            state = env.reset() # reset env at the start of each episode
            done = False
            total_reward = 0
            total_steps = 0

            while not done:
                if episode % render_frequency == 0: # Renders every 50 episodes
                    env.render() #rendering the environment
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

                print(f"\nBoard {board_num}, Episode {episode}/{episodes}")
                print(f"Average Reward (last 10): {avg_reward:.2f}")
                print(f"Average Steps (last 10): {avg_steps:.2f}")
                print(f"Average Epsilon (last 10): {avg_epsilon:.2f}")
                print(f"Learning Rate: {agent.get_stats()['alpha']:.2f}")
                print(f"Q-table size: {agent.get_stats()['q_table_size']}")

            #Save the final model
            if done:
                agent.save(os.path.join(os.path.dirname(__file__), 'models', f'zombie_agent_ep{board_num}.pkl'))

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
        agent.save('zombie_agent_interrupted.pkl')

    finally:
        env.close()

    return {'rewards': rewards, 'steps': steps, 'epsilon': epsilon}

def display_training_stats(reward, steps, epsilon, episodes):
    #Calculating and printing training statistics
    final_stats = {
        'reward': {
            'mean': np.mean(reward),
            'min': np.min(reward),
            'max': np.max(reward),
            'std': np.std(reward)
        },
        'steps': {
            'mean': np.mean(steps),
            'min': np.min(steps),
            'max': np.max(steps),
            'std': np.std(steps)
        },
        'final_epsilon': epsilon[-1]
    }

    print("\nFinal Training Statistics:")
    print("=" * 50)
    print(f"Total Episodes: {episodes}")
    print("\nReward Statistics:")
    print(f"  Mean Reward: {final_stats['reward']['mean']:.2f}")
    print(f"  Min Reward:  {final_stats['reward']['min']:.2f}")
    print(f"  Max Reward:  {final_stats['reward']['max']:.2f}")
    print(f"  Std Reward:  {final_stats['reward']['std']:.2f}")
    print("\nSteps Statistics:")
    print(f"  Mean Steps: {final_stats['steps']['mean']:.2f}")
    print(f"  Min Steps:  {final_stats['steps']['min']:.2f}")
    print(f"  Max Steps:  {final_stats['steps']['max']:.2f}")
    print(f"  Std Steps:  {final_stats['steps']['std']:.2f}")
    print(f"\nFinal Epsilon: {final_stats['final_epsilon']:.4f}")
    print("=" * 50)



def main():

    while True:
        print("Select a board to test the agent on (enter the corresponding number):")
        print("The boards grow in complexity in higher numbers.")
        print("1. 5x5 board  -- A small empty environment resulting in quicker trials")
        print("2. 8x8 board  -- A board with two simple hallways near each starting point")
        print("3. 8x8 board -- A board with sporadic walls and hallways")
        print("4. 5x5 board  -- Small board with intricate maze system")
        print("5. 10x10 board  -- The most complex board that has empty passages along outer ring (Will take a long time to finish)")
        print("6. Exit")

        try:
            board_num = int(input())
            if board_num == 6:
                print("Exiting...")
                return 
            
            # map input to board number
            if board_num not in [1, 2, 3, 4, 5, 6]:
                print("Invalid input. Please enter a number between 1 and 6.")
                continue 

            episodes = 1000
            render_frequency = 50

            env = MazeEnv(board_number=board_num)
            agent = QLearningAgent(env.observation_space.shape[0], env.action_space.n)

            print("Training agent...")
            metrics = test_agent(board_num, episodes, render_frequency, agent)
            plot_training(board_num, metrics['rewards'], metrics['steps'], metrics['epsilon'], episodes)

            print(f"Training complete on board {board_num}.\n")
            display_training_stats(metrics['rewards'], metrics['steps'], metrics['epsilon'], episodes)
            print("")
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 6.")
            return

# Main, obv
if __name__ == "__main__":
    main()