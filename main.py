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

# All of this can be changed, just use the same constructs that are here for further testing
def main():
    env = MazeEnv(board_number=1)  # this chooses the board number that will be used at runtime
    # Only a single agent is made right now, more agents can be made and tested
    agent = QLearningAgent(env.observation_space.shape[0], env.action_space.n)
    
    # Helpful definitions
    episodes = 1000
    save_frequency = 100
    render_frequency = 50
    
    # Used to track metrics
    episode_rewards = []
    episode_steps = []

    # This is where all the episodes will play out
    try:
        for episode in range(episodes):
            state = env.reset() # Begin with a state where zombie and human are opposite
            done = False
            total_reward = 0
            steps = 0

            while not done:
                # This will make the program render less frequently
                if episode % render_frequency == 0:
                    env.render()
                    time.sleep(0.1)

                # This is what happens every step the zombie and human take
                # all values are saved into step which moves the humand and 
                # zombie and rewards the zombie. The state is then updated
                # the total reward is tracked as well as the number of steps
                action = agent.get_action(state)
                next_state, reward, done, info = env.step(action)
                agent.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1

            # Appending the number of steps and rewards amount to the local lists
            episode_rewards.append(total_reward)
            episode_steps.append(steps)

            # Prints progress every 10 episodes, we dont need to know everything
            # All of this data can be used for plots
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

            # Saving the model periodically in the models folder. A learned model could 
            # then be used on a new env to see how it does
            if episode % save_frequency == 0:
                agent.save(os.path.join(os.path.dirname(__file__), 'models', f'zombie_agent_ep{episode}.pkl'))

    # Used for is the user ctrl + c the program 
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
        agent.save('zombie_agent_interrupted.pkl')

    # Closing the env as its not needed anymore
    finally:
        env.close()

        # Display final learning stats
        if len(episode_rewards) > 0:
            print("\nFinal Statistics:")
            print(f"Best Episode Reward: {max(episode_rewards):.2f}")
            print(f"Average Episode Reward: {np.mean(episode_rewards):.2f}")
            print(f"Final Q-table Size: {len(agent.q_table)}")

# Main, obv
if __name__ == "__main__":
    main()