import matplotlib.pyplot as plt

def plot_rewards(step_rewards, episode_rewards, epsilon, learning_rate, discount_factor):
    plt.figure(figsize=(14, 7))
    
    # Plot for steps
    plt.subplot(1, 2, 1)
    steps, s_rewards = zip(*step_rewards)
    plt.plot(steps, s_rewards, label='Rewards per Step')
    max_step_reward_idx = s_rewards.index(max(s_rewards))
    plt.scatter(steps[max_step_reward_idx], s_rewards[max_step_reward_idx], color='red')
    plt.annotate(f"Win Model at Step: {steps[max_step_reward_idx]} with Reward: {s_rewards[max_step_reward_idx]}", 
                 xy=(steps[max_step_reward_idx], s_rewards[max_step_reward_idx]),
                 xytext=(steps[max_step_reward_idx]-len(steps)*0.2, s_rewards[max_step_reward_idx]-max(s_rewards)*0.1),
                 arrowprops=dict(arrowstyle="->"))
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title(f"Points per Step")
    plt.legend()
    plt.grid(True)
    
    # Plot for episodes
    plt.subplot(1, 2, 2)
    episodes, e_rewards = zip(*episode_rewards)
    plt.plot(episodes, e_rewards, label='Rewards per Episode')
    max_episode_reward_idx = e_rewards.index(max(e_rewards))
    plt.scatter(episodes[max_episode_reward_idx], e_rewards[max_episode_reward_idx], color='red')
    plt.annotate(f"Win Model at Episode: {episodes[max_episode_reward_idx]} with Reward: {e_rewards[max_episode_reward_idx]}", 
                 xy=(episodes[max_episode_reward_idx], e_rewards[max_episode_reward_idx]),
                 xytext=(episodes[max_episode_reward_idx]-len(episodes)*0.2, e_rewards[max_episode_reward_idx]-max(e_rewards)*0.1),
                 arrowprops=dict(arrowstyle="->"))
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f"Starting Epsilon={epsilon}, Learning Rate={learning_rate}, Discount Factor={discount_factor}")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def running_average(data, window_size):
    """Compute running average of a list."""
    cumsum = [0]
    for i, x in enumerate(data, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=window_size:
            avg = (cumsum[i] - cumsum[i-window_size]) / window_size
            yield avg

def plot_learning_curve(rewards, window_size=100):
    episodes, e_rewards = zip(*rewards)
    smoothed_rewards = list(running_average(e_rewards, window_size))
    
    plt.figure(figsize=(10, 5))
    plt.plot(episodes[len(episodes) - len(smoothed_rewards):], smoothed_rewards, label=f'Running Average (window size = {window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Reward')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def read_data_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        data = []
        for line in lines:
            parts = line.split()
            step_or_episode = int(parts[1][:-1])  # Remove the colon at the end and convert to integer
            reward = float(parts[5])
            data.append((step_or_episode, reward))
        return data

if __name__ == "__main__":
    # Read data from files
    episode_rewards = read_data_from_file("dec_run_episodes_264.txt")
    step_rewards = read_data_from_file("dec_un_steps_1000000.txt")
    
    # Set these values according to your requirement or fetch from your RL model
    epsilon = 1
    learning_rate = 0.3
    discount_factor = 0.9
    plot_learning_curve(episode_rewards, window_size=10)  # Feel free to adjust the window_size as needed.

    # plot_rewards(step_rewards, episode_rewards, epsilon, learning_rate, discount_factor)

