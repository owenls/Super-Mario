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

class QLearningAgent:
    def __init__(self, env, max_life=3, num_stacked_frames=4):
        self.env = JoypadSpace(env, SIMPLE_MOVEMENT)
        self.state_space_size = self.calculate_state_space_size()
        self.action_space_size = 6  # Number of actions available
        self.Q = {}
        self.episode_count = 0  # Initialize episode count
        self.all_rewards = []
        self.all_wins = []
        self.learning_rate = 0.3
        self.discount_factor = 0.9
        self.epsilon = 1
        self.epsilon_decay = 0.9995
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
        if state not in self.Q:
            self.Q[state] = np.zeros(self.action_space_size)
        if obs not in self.Q:
            self.Q[obs] = np.zeros(self.action_space_size)
        # death_penalty = -500
        # progress_reward = 10
        # stuck_penalty = -100
        end_level_reward = 1000
        # jump_success_reward = 50  # New reward
        # jump_fail_penalty = -50   # New penalty

        # if state not in self.Q:
        #     self.Q[state] = np.zeros(self.action_space_size)
        # if obs not in self.Q:
        #     self.Q[obs] = np.zeros(self.action_space_size)

        # Reward for reaching the end of the level
        if done and 'flag_get' in info and info['flag_get']:
            self.save_model("win_model.pkl")
            reward += end_level_reward

        # # Penalty for death
        # if done and 'life' in info and info['life'] < self.life_count:
        #     reward -= death_penalty

        # # Reward for progress
        # if 'x_pos' in info and info['x_pos'] > self.previous_x_pos:
        #     reward += progress_reward

        # # Check for failed jumps (you might need to refine this based on specific game mechanics)
        # if 'x_pos' in info and info['x_pos'] == self.previous_x_pos:
        #     reward -= jump_fail_penalty

        # if done and 'life' in info and info['life'] < self.life_count and 'x_pos' in info and abs(info['x_pos'] - self.previous_x_pos) < 5:
        #     reward -= death_penalty

        # if 'x_pos' in info and info['x_pos'] == self.previous_x_pos:
        #     self.consecutive_stuck_frames += 1
        # else:
        #     self.consecutive_stuck_frames = 0
        #     self.previous_x_pos = info.get('x_pos', None)

        # if self.consecutive_stuck_frames > 20:
        #     reward -= stuck_penalty
        #     self.consecutive_stuck_frames = 0

        max_next_action_value = np.max(self.Q[obs])
        self.Q[state][action] = (1 - self.learning_rate) * self.Q[state][action] + \
            self.learning_rate * (reward + self.discount_factor * max_next_action_value)

        self.previous_x_pos = info.get('x_pos', None)

    def run(self, steps):
        highest_reward = 0  # Variable to store the highest reward achieved
        for step in range(steps):
            if self.done:
                self.current_state = self.preprocess_state(self.env.reset())
                self.done = False
                self.episode_count += 1  # Increment episode count when a new episode starts

            action = self.select_action(self.current_state, step)  # Pass 'step' to select_action
            obs, reward, terminated, truncated, info = self.env.step(action)
            obs = self.preprocess_state(obs)

            self.update_q_table(self.current_state, action, reward, obs, self.done, info)
            self.current_state = obs
            self.done = terminated or truncated

            if self.done:
                self.life_count -= 1
                if self.life_count <= 0:
                    self.done = True
                    self.life_count = self.max_life

                total_reward = np.sum(list(self.Q.values()))
                self.all_rewards.append(np.sum(list(self.Q.values())))
                if 'flag_get' in info and info['flag_get']:
                    self.all_wins.append(1)
                else:
                    self.all_wins.append(0)
                # Save model if the agent achieves a new highest reward
                if total_reward > highest_reward:
                    highest_reward = total_reward
                    model_filename = f'models/10M/mario_model_{step}_reward_{total_reward}.pkl'
                    self.save_model(model_filename)

            if step % 100 == 0:
                level_completed = 'Yes' if 'flag_get' in info and info['flag_get'] else 'No'
                print(f"Episode: {self.episode_count}, Step: {step}, Total Reward: {np.sum(list(self.Q.values()))}, Completed Level: {level_completed}")

        # plt.plot(list(self.Q.values()))
        # plt.xlabel('Step')
        # plt.ylabel('Total Reward')
        # plt.title('Rewards over Steps')
        # plt.show()

        self.env.close()

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
    mario_agent = QLearningAgent(env, num_stacked_frames=4)

    try:
        # mario_agent.load_model('models/100k/mario_model_69594_reward_4581.176886087973.pkl')
        print("Loaded saved model.")
    except FileNotFoundError:
        print("No saved model found.")

    num_steps = 10000000 # Set the number of steps for the run
    mario_agent.run(num_steps)

    # Compute average rewards per 1000 episodes
    avg_rewards = [np.mean(mario_agent.all_rewards[i:i+100]) for i in range(0, len(mario_agent.all_rewards), 100)]
    episodes_range = range(100, len(mario_agent.all_rewards)+1, 100)

    if len(avg_rewards) == len(episodes_range):
        plt.plot(episodes_range, avg_rewards, '-o')
        plt.xlabel('Episodes')
        plt.ylabel('Average Reward')
        plt.title('Average Reward per 100 Episodes')
        plt.show()
    else:
        print("Mismatch in dimensions: episodes_range:", len(episodes_range), "avg_rewards:", len(avg_rewards))


    # # Compute win rate per 1000 episodes
    # win_rates = [np.mean(mario_agent.all_wins[i:i+100]) for i in range(0, len(mario_agent.all_wins), 100)]

    # # Plot win rate
    # plt.plot(episodes_range, win_rates, '-o')
    # plt.xlabel('Episodes')
    # plt.ylabel('Win Rate')
    # plt.title('Win Rate per 100 Episodes')
    # plt.show()