# Author 1: Owen Smith (22957291)
# Author 2: John Lumagbas (23419439)
import sys
import time
import random
import numpy as np
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import gym
import pickle

# Class for the Q-Learning Implementation of the Mario Agent.


class QLearningAgent:
    def __init__(self, env):
        self.env = JoypadSpace(env, SIMPLE_MOVEMENT)
        self.state_space_size = self.calculate_state_space_size()
        self.action_space_size = 6  # Number of actions availible
        self.Q = {}  # This will store the Q table data.

        """ Q-Learning Control Explained (notes for self really)"""
        # Learning rate (alpha) is 0-1. Approaching 0 is slower but smoother,
        # towards 1 means the agent might shift between different Q-value estimates.
        self.learning_rate = 0.1

        # Discount Factor (gamma) is 0-1. Approaching 0 means the agent focuses mostly
        # short term success (sorta greedy). Approaching 1 means the agent values
        # future rewards almost as much as immediate rewards.
        self.discount_factor = 0.4

        # Epsilon is 0-1. It controls the exploration-exploitation trade-off.
        # Close to zero means the agent mostly exploits its current knowledge.
        # Close to 1 means he chooses more random actions to try discover better strategies.
        # High epsilon values good when the agent doesn't know much about the environment.
        self.epsilon = 0.2

        self.frame_delay = 0.02  # This is just the delay used to slow down the frames
        self.current_state = None
        self.done = True
        self.prev_x_pos = 0

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
    def update_q_table(self, state, action, reward, obs, info):
        if state not in self.Q:
            self.Q[state] = np.zeros(self.action_space_size)
        if obs not in self.Q:
            self.Q[obs] = np.zeros(self.action_space_size)

        max_next_action_value = np.max(self.Q[obs])

        # Calculate the change in x_pos
        current_x_pos = info['x_pos']
        x_pos_change = current_x_pos - self.prev_x_pos

        # Provide a reward based on the change in x_pos
        reward += x_pos_change * 0.3

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

    def run(self, episodes):
        # Initialize episode number
        episode = 1

        # Check if the log file exists and has data
        try:
            with open('episode_log.txt', 'r') as log_file:
                lines = log_file.readlines()
                if lines:
                    # Get the last line and extract the episode number
                    last_line = lines[-1].strip()
                    last_episode = int(last_line.split(':')[0].split(' ')[
                                       1])  # Extract the episode number
                    episode = last_episode + 1
        except FileNotFoundError:
            pass  # The log file doesn't exist, start from episode 1

        with open('episode_log.txt', 'a') as log_file:
            for _ in range(episode, episode + episodes):
                self.current_state = self.preprocess_state(self.env.reset())
                total_reward = 0
                life_reward = 0

                self.done = False
                episode_done = False  # Variable to track the end of the current episode
                while not self.done:
                    action = self.select_action(self.current_state)
                    obs, reward, terminated, truncated, info = self.env.step(
                        action)
                    obs = self.preprocess_state(obs)

                    self.update_q_table(self.current_state,
                                        action, reward, obs, info)

                    self.current_state = obs
                    total_reward += reward
                    life_reward += reward

                    # Check if Mario reaches the flag
                    if info['flag_get']:
                        episode_done = True

                    self.done = terminated or truncated

                    # End of the episode
                    if episode_done:
                        self.save_model(
                            'mario_model_episode_{}.pkl'.format(episode))

                # Write to File
                log_file.write(
                    f"Episode {episode}: Total Reward = {total_reward} | x_pos = {info['x_pos']} | completed: {episode_done}\n")

                # Write Exact same thing to terminal so we can watch it - file often doesnt update live
                print(
                    f"Episode {episode}: Total Reward = {total_reward} | x_pos = {info['x_pos']} | completed: {episode_done}")

                self.save_model('latest_model.pkl')
                episode += 1

        # Close the log file after all episodes are completed
        log_file.close()

        self.save_model('mario_model_final.pkl')
        self.env.close()


if __name__ == "__main__":
    env = gym.make("SuperMarioBros-v0",
                   apply_api_compatibility=True, render_mode="human")
    mario_agent = QLearningAgent(env)

    try:
        model_to_load = 'latest_model.pkl'
        mario_agent.load_model(model_to_load)
        print(f"\n\nLOADED -- {model_to_load} --\n\n")
    except FileNotFoundError:
        print("No saved model found.")

    mario_agent.run(episodes=1000)
