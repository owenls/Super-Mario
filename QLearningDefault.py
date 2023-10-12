# Author 1: Owen Smith (22957291)
# Author 2: John Lumagbas (23419439)

import matplotlib.pyplot as plt
import time
import random
import numpy as np
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import gym
import pickle

# Class for the Q-Learning Implementation of the Mario Agent.

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

def save_to_file(filename, episode, reward):
    with open(filename, 'a') as file:
        file.write(f"Episode {episode + 1}: Total Reward = {reward}\n")
class QLearningAgent:
    def __init__(self, env):
        self.env = JoypadSpace(env, SIMPLE_MOVEMENT)
        self.state_space_size = self.calculate_state_space_size()
        self.action_space_size = 6  # Number of actions availible
        self.Q = {}  # This will store the Q table data.

        """ Q-Learning Control Explained (notes for self really)"""
        # Learning rate (alpha) is 0-1. Approaching 0 is slower but smoother,
        # towards 1 means the agent might shift between different Q-value estimates.
        self.learning_rate = 0.3

        # Discount Factor (gamma) is 0-1. Approaching 0 means the agent focuses mostly
        # short term success (sorta greedy). Approaching 1 means the agent values
        # future rewards almost as much as immediate rewards.
        self.discount_factor = 0.9

        # Epsilon is 0-1. It controls the exploration-exploitation trade-off.
        # Close to zero means the agent mostly exploits its current knowledge.
        # Close to 1 means he chooses more random actions to try discover better strategies.
        # High epsilon values good when the agent doesn't know much about the environment.
        self.epsilon = 0.1

        self.frame_delay = 0.02  # This is just the delay used to slow down the frames
        self.current_state = None
        self.done = True

    # Preprocesses the raw 'obs' data from the environment to create a hashable representation.
    def preprocess_state(self, obs):
        return tuple(obs[0].flatten())

    # Chooses the next action
    # It implements the epsilon-greedy strategy - balances exploration (random actions)
    # and exploitation (best-known actions) based on the value of epsilon.
    def select_action(self, obs):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        else:
            if obs not in self.Q:
                self.Q[obs] = np.zeros(self.action_space_size)
            return np.argmax(self.Q[obs])

    # Calculate the size of the state space based on the observation space shape
    # Without this i didn't know how large the space was. Was getting KeyError
    def calculate_state_space_size(self):
        return np.prod(self.env.observation_space.shape)

    # Updates the Q-value for the (state, action) pair in the Q-table using the
    # Q-learning update rule. It also incorporates the observed reward and the best
    # estimate of future rewards.
    def update_q_table(self, state, action, reward, obs):
        if state not in self.Q:
            self.Q[state] = np.zeros(self.action_space_size)
        if obs not in self.Q:
            self.Q[obs] = np.zeros(self.action_space_size)

        max_next_action_value = np.max(self.Q[obs])
        self.Q[state][action] = (1 - self.learning_rate) * self.Q[state][action] + \
            self.learning_rate * \
            (reward + self.discount_factor * max_next_action_value)

    # Called to save a model - pretty much just when we reach the flag
    def save_model(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.Q, file)

    # Called to load an existing model. The existence check happens at the point it was called.
    def load_model(self, filename):
        with open(filename, 'rb') as file:
            self.Q = pickle.load(file)

    # Called to run the environment.
    def run(self, steps):
        step_count = 0
        episode_count = 0
        episode_rewards = []
        step_rewards = []
        highest_reward = 0  # Initialize variable to store the highest reward achieved
        while step_count < steps:
            self.current_state = self.preprocess_state(self.env.reset())
            total_reward = 0
            self.done = False
            
            while not self.done and step_count < steps:
                action = self.select_action(self.current_state)
                obs, reward, terminated, truncated, info = self.env.step(action)
                obs = self.preprocess_state(obs)

                self.update_q_table(self.current_state, action, reward, obs)
                self.current_state = obs
                total_reward += reward

                step_count += 1
                step_rewards.append((step_count, total_reward))

                # If Mario reaches the flag then save the Q table data
                self.done = terminated or truncated
                if self.done and 'flag_get' in info and info['flag_get']:
                    self.save_model('mario_model.pkl')

            episode_count += 1
            episode_rewards.append((episode_count, total_reward))
            print(f"Episode {episode_count}, Steps: {step_count}, Total Reward = {total_reward}")

            # Check if total_reward is a new highest and save model if it is
            if total_reward > highest_reward:
                highest_reward = total_reward
                model_filename = f'default/mario_model_{step_count}_reward_{highest_reward}.pkl'
                self.save_model(model_filename)

        self.env.close()
        return step_rewards, episode_rewards
    
 
if __name__ == "__main__":
    env = gym.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human")
    mario_agent = QLearningAgent(env)

    try:
        mario_agent.load_model('mario_model.pkl')
        print("Loaded saved model.")
    except FileNotFoundError:
        print("No saved model found.")

    steps = 1000
    000
    step_rewards, episode_rewards = mario_agent.run(steps=steps)

    # Saving all steps and episode rewards to file
    for step, reward in step_rewards:
        save_to_file(f'def_run_steps_{steps}.txt', step, reward)
    for episode, reward in episode_rewards:
        save_to_file(f'def_run_episodes_{len(episode_rewards)}.txt', episode, reward)

    plot_rewards(step_rewards, episode_rewards, mario_agent.epsilon, mario_agent.learning_rate, mario_agent.discount_factor)