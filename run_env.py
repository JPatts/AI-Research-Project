#This file might not be used for gym, but it is used for qlearning
from main_env import MazeEnv
from qlearning import QLearningAgent
import time
import numpy as np
import matplotlib.pyplot as plt

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





def main():
    env = MazeEnv(board_number=1)
    agent = QLearningAgent(env.observation_space.shape[0], env.action_space.n)
    
    episodes = 1000
    save_frequency = 100
    render_frequency = 50
    
    # Track metrics
    episode_rewards = []
    episode_steps = []

    try:
        for episode in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0
            steps = 0

            while not done:
                # Render less frequently
                if episode % render_frequency == 0:
                    env.render()
                    time.sleep(0.1)

                action = agent.get_action(state)
                next_state, reward, done, info = env.step(action)
                agent.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1

            episode_rewards.append(total_reward)
            episode_steps.append(steps)

            # Prints progress every 10 episodes
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_steps = np.mean(episode_steps[-10:])
                stats = agent.get_stats()
                
                print(f"\nEpisode {episode}/{episodes}")
                print(f"Average Reward (last 10): {avg_reward:.2f}")
                print(f"Average Steps (last 10): {avg_steps:.2f}")
                print(f"Epsilon: {stats['epsilon']:.2f}")
                print(f"Learning Rate: {stats['alpha']:.2f}")
                print(f"Q-table size: {stats['q_table_size']}")

            # Saving the model periodically
            if episode % save_frequency == 0:
                agent.save(f'zombie_agent_ep{episode}.pkl')

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
        agent.save('zombie_agent_interrupted.pkl')

    finally:
        env.close()

        # Plot final learning curves
        if len(episode_rewards) > 0:
            print("\nFinal Statistics:")
            print(f"Best Episode Reward: {max(episode_rewards):.2f}")
            print(f"Average Episode Reward: {np.mean(episode_rewards):.2f}")
            print(f"Final Q-table Size: {len(agent.q_table)}")

if __name__ == "__main__":
    main()