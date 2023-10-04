# Author 1: Owen Smith (22957291)
# Author 2: John Lumagbas (23419439)
import time
import random
import numpy as np
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import gym
import pickle
import matplotlib.pyplot as plt


# Class for the Q-Learning Implementation of the Mario Agent.


class QLearningAgent:
    def __init__(self, env, max_life=3):
        self.env = JoypadSpace(env, COMPLEX_MOVEMENT)
        self.state_space_size = self.calculate_state_space_size()
        self.action_space_size = 6  # Number of actions availible
        self.Q = {}  # This will store the Q table data.

        """ Q-Learning Control Explained (notes for self really)"""
        # Learning rate (alpha) is 0-1. Approaching 0 is slower but smoother,
        # towards 1 means the agent might shift between different Q-value estimates.
        self.learning_rate = 1

        # Discount Factor (gamma) is 0-1. Approaching 0 means the agent focuses mostly
        # short term success (sorta greedy). Approaching 1 means the agent values
        # future rewards almost as much as immediate rewards.
        self.discount_factor = 1

        # Epsilon is 0-1. It controls the exploration-exploitation trade-off.
        # Close to zero means the agent mostly exploits its current knowledge.
        # Close to 1 means he chooses more random actions to try discover better strategies.
        # High epsilon values good when the agent doesn't know much about the environment.
        self.epsilon = 1

        self.frame_delay = 1  # This is just the delay used to slow down the frames
        self.current_state = None
        self.done = True
        self.current_episode = 0
        self.max_life = max_life  # Maximum number of lives Mario can have
        self.life_count = max_life  # Initialize the life count
        
        self.fall_penalty = 100  # Penalty for falling
        self.stuck_penalty = 50  # Penalty for getting stuck
        self.consecutive_stuck_frames = 0
        self.previous_x_pos = None
        self.fall_penalty = 100  # Penalty for falling
        self.stuck_penalty = 500  # Penalty for getting stuck
    # Preprocesses the raw 'obs' data from the environment to create a hashable representation.
    def preprocess_state(self, obs):
        return tuple(obs[0].flatten())

    # Chooses the next action
    # It implements the epsilon-greedy strategy - balances exploration (random actions)
    # and exploitation (best-known actions) based on the value of epsilon.
    
    def select_action(self, obs, episode):
        # Increase exploration at the beginning, gradually decrease it over episodes
        epsilon = max(0.01, 0.2 - 0.002 * episode)  # Starts with epsilon=0.2, decreases slowly
        
        if random.uniform(0, 1) < epsilon:
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
    def update_q_table(self, state, action, reward, obs, done, info):
        if state not in self.Q:
            self.Q[state] = np.zeros(self.action_space_size)
        if obs not in self.Q:
            self.Q[obs] = np.zeros(self.action_space_size)

        # Define your rewards and penalties
        jump_obstacle_reward = 10  # Positive reward for jumping over obstacles
        jump_enemy_reward = 20  # Positive reward for jumping on enemies (defeating them)
        collect_coin_reward = 5  # Positive reward for collecting coins
        level_complete_reward = 1000  # Substantial positive reward for completing the level

        time_penalty = 0.1  # Penalty for taking too long to complete
        stuck_penalty = 1  # Penalty for getting stuck
        death_penalty = 100  # Substantial penalty for dying
        jump_fail_penalty = 30
        # Apply rewards and penalties based on the game events
        if reward > 0:
            if reward == jump_obstacle_reward:
                reward += jump_obstacle_reward
            elif reward == jump_enemy_reward:
                reward += jump_enemy_reward
            elif reward == collect_coin_reward:
                reward += collect_coin_reward
            elif reward == level_complete_reward:
                reward += level_complete_reward
            

        elif done:
            if 'time' in info and info['time'] == 0:
                reward -= time_penalty * self.frame_delay
            elif 'x_pos' in info and info['x_pos'] == self.previous_x_pos:
                reward -= stuck_penalty  # Apply stuck penalty if Mario gets stuck
            elif 'life' in info and info['life'] < self.life_count:
                reward -= self.fall_penalty
            elif 'x_pos' in info and info['x_pos'] == self.previous_x_pos:
                reward -= jump_fail_penalty

        elif reward == 0 and not done:
            reward -= time_penalty * self.frame_delay

        # Update consecutive stuck frames and check for stuck penalty
        if 'x_pos' in info and info['x_pos'] == self.previous_x_pos:
            self.consecutive_stuck_frames += 1
        else:
            self.consecutive_stuck_frames = 0
        stuck_threshold = 5
        if self.consecutive_stuck_frames >= stuck_threshold:
            reward -= stuck_penalty  # Apply additional stuck penalty if Mario remains stuck

        # Q-learning update rule
        max_next_action_value = np.max(self.Q[obs])
        self.Q[state][action] = (1 - self.learning_rate) * self.Q[state][action] + \
            self.learning_rate * \
            (reward + self.discount_factor * max_next_action_value)

        # Update previous_x_pos for the next iteration
        self.previous_x_pos = info.get('x_pos', None)
    # Called to save a model - pretty much just when we reach the flag
    def save_model(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.Q, file)

    # Called to load an existing model. The existence check happens at the point it was called.
    def load_model(self, filename):
        with open(filename, 'rb') as file:
            self.Q = pickle.load(file)

    def run(self, episodes):
        for episode in range(episodes):
            self.current_state = self.preprocess_state(self.env.reset())
            total_reward = 0
            self.life_count = 1  # Set life count to 1 for each episode
            completed = False  # Flag to track if the level was completed in this episode

            # Run the episode until Mario loses a life or completes it
            while self.life_count > 0:
                if self.done:  # Check if the episode is done; if so, reset the environment
                    self.current_state = self.preprocess_state(self.env.reset())
                    self.done = False

                # Mario starts a new life
                while not self.done:
                    action = self.select_action(self.current_state, episode)
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    obs = self.preprocess_state(obs)

                    self.update_q_table(self.current_state, action, reward, obs, self.done, info)

                    self.current_state = obs
                    total_reward += reward

                    # If Mario loses a life, end the episode
                    if 'life' in info and info['life'] < self.life_count:
                        self.done = True
                        self.life_count = 0  # Decrease the life count
                        break

                    # If Mario completes the level, end the episode
                    if 'flag_get' in info and info['flag_get']:
                        self.done = True
                        self.save_model('mario_model.pkl')
                        completed = True
                        break

                    # Uncomment the line below to slow down frames (optional)
                    # time.sleep(self.frame_delay)

            print(f"Episode {episode + 1}: Total Reward = {total_reward}, Position = {info.get('x_pos')}, Completed: {completed}")

        plt.plot(total_reward)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Rewards over Episodes')
        plt.show()

        self.env.close()



if __name__ == "__main__":
    env = gym.make("SuperMarioBros-v0",
                   apply_api_compatibility=True, render_mode="human")
    mario_agent = QLearningAgent(env)

    try:
        mario_agent.load_model('mario_model.pkl')
        print("Loaded saved model.")
    except FileNotFoundError:
        print("No saved model found.")

    mario_agent.run(episodes=10000)
