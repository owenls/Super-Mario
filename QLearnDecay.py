import time
import random
import numpy as np
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import gym
import pickle
import matplotlib.pyplot as plt
import cv2
from collections import deque

def save_to_file(filename, step_or_episode, reward):
    with open(filename, 'a') as file:
        file.write(f"Step/Episode {step_or_episode + 1}: Total Reward = {reward}\n")

def plot_rewards(step_rewards, episode_rewards, epsilon, learning_rate, discount_factor):
    plt.figure(figsize=(14, 7))
    
    # Plot for steps
    plt.subplot(1, 2, 1)
    steps, s_rewards = zip(*step_rewards)
    plt.plot(steps, s_rewards, label='Rewards per Step')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    
    # Plot for episodes
    plt.subplot(1, 2, 2)
    episodes, e_rewards = zip(*episode_rewards)
    plt.plot(episodes, e_rewards, label='Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f"Epsilon={epsilon}, Learning Rate={learning_rate}, Discount Factor={discount_factor}")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()



class QLearningAgent:
    def __init__(self, env, max_life=3, num_stacked_frames=4):
        self.env = JoypadSpace(env, SIMPLE_MOVEMENT)
        self.state_space_size = self.calculate_state_space_size()
        self.action_space_size = 6  # Number of actions available
        self.Q = {}
        self.episode_count = 0  # Initialize episode count
        self.all_rewards = []
        
        self.all_wins = []
        self.learning_rate = 0.1
        self.discount_factor = 0.85
        self.epsilon = 1
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.1
        self.frame_delay = 1
        self.current_state = None
        self.done = True
        self.max_life = max_life
        self.life_count = max_life
        self.consecutive_stuck_frames = 0
        self.previous_x_pos = 0

        # Initialize frame stack parameters
        self.num_stacked_frames = num_stacked_frames
        self.frame_stack = deque(maxlen=num_stacked_frames)

    def downsample_image(self, image, scale_percent=45):
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    def to_grayscale(self, image):
        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def preprocess_state(self, obs):
        
        grayscale_obs = self.to_grayscale(obs[0])
        cropped_obs = grayscale_obs[50:-12, :]
        downsampled_obs = self.downsample_image(cropped_obs, scale_percent=40)
        _, binary_obs = cv2.threshold(downsampled_obs, 128, 255, cv2.THRESH_BINARY)
        return tuple(binary_obs.flatten())
    
    def update_epsilon(self):
        """Decay epsilon with a minimum limit"""
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def select_action(self, obs, step):
        self.update_epsilon()  # Update epsilon using exponential decay
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        else:
            if obs not in self.Q:
                self.Q[obs] = np.zeros(self.action_space_size)
            return np.argmax(self.Q[obs])

    def calculate_state_space_size(self):
        return np.prod(self.env.observation_space.shape)

    def update_q_table(self, state, action, reward, obs, done, info):
        # Reward for reaching the end of the level
        end_level_reward = 1000
        if done and 'flag_get' in info and info['flag_get']:
            reward += end_level_reward

        # Penalty for death
        death_penalty = -500
        if done and 'life' in info and info['life'] < self.life_count:
            reward += death_penalty

        # Reward for progress
        progress_reward = 10
        if 'x_pos' in info and info['x_pos'] > self.previous_x_pos:
            reward += progress_reward

        # Stuck Penalty
        stuck_penalty = -20
        if 'x_pos' in info and info['x_pos'] == self.previous_x_pos:
            self.consecutive_stuck_frames += 1
        else:
            self.consecutive_stuck_frames = 0

        if self.consecutive_stuck_frames > 20:
            reward -= stuck_penalty
            self.consecutive_stuck_frames = 0

        # Update Q-values
        if state not in self.Q:
            self.Q[state] = np.zeros(self.action_space_size)
        if obs not in self.Q:
            self.Q[obs] = np.zeros(self.action_space_size)
        
        max_next_action_value = np.max(self.Q[obs])
        self.Q[state][action] = (1 - self.learning_rate) * self.Q[state][action] + \
            self.learning_rate * (reward + self.discount_factor * max_next_action_value)

        self.previous_x_pos = info.get('x_pos', None)


    def run(self, steps):
        step_count = 0
        episode_rewards = []
        step_rewards = []
        highest_reward = 0  # Initialize variable to store the highest reward achieved
        
        while step_count < steps:
            self.current_state = self.preprocess_state(self.env.reset())
            total_reward = 0
            self.done = False
            self.previous_x_pos = 0  # Reset starting x-position for new episode
            
            while not self.done and step_count < steps:
                action = self.select_action(self.current_state, step_count)
                obs, reward, terminated, truncated, info = self.env.step(action)
                obs = self.preprocess_state(obs)
                
                self.update_q_table(self.current_state, action, reward, obs, self.done, info)
                self.current_state = obs
                total_reward += reward

                step_count += 1
                step_rewards.append((step_count, total_reward))

                self.done = terminated or truncated

            self.episode_count += 1
            episode_rewards.append((self.episode_count, total_reward))
            print(f"Episode {self.episode_count}, Steps: {step_count}, Total Reward = {total_reward}")

            # Check if total_reward is a new highest and save model if it is
            if step_count % 100000:
                highest_reward = total_reward
                model_filename = f'decay/mario_model_{step_count}_reward_{highest_reward}.pkl'
                self.save_model(model_filename)
            if self.done and 'flag_get' in info and info['flag_get']:
                win = f'decay/mario_win_model_{step_count}_reward_{highest_reward}.pkl'
                self.save_model(win)

        self.env.close()
        return step_rewards, episode_rewards


    def load_model(self, filename):
        try:
            with open(filename, 'rb') as file:
                self.Q = pickle.load(file)
                print(f"Loaded model from {filename}")
        except FileNotFoundError:
            print("No saved model found.")

    def save_model(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.Q, file)
            print(f"Model saved as {filename}")

if __name__ == "__main__":
    env = gym.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human")
    mario_agent = QLearningAgent(env)

    model_filename = 'decay1m/mario_model_521672_reward_1448.0.pkl'
    mario_agent.load_model(model_filename)

    num_steps = 10000000 
    step_rewards, episode_rewards = mario_agent.run(num_steps)

    # Saving all steps and episode rewards to file
    for step, reward in step_rewards:
        save_to_file(f'dec10m_un_steps_{num_steps}.txt', step, reward)
    for episode, reward in episode_rewards:
        save_to_file(f'dec10m_run_episodes_{len(episode_rewards)}.txt', episode, reward)

    # Plot the rewards
    plot_rewards(step_rewards, episode_rewards, mario_agent.epsilon, mario_agent.learning_rate, mario_agent.discount_factor)
 