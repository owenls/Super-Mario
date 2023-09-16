import time
import random
import numpy as np
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import gym
import pickle


class QLearningAgent:
    def __init__(self, env):
        self.env = JoypadSpace(env, SIMPLE_MOVEMENT)
        self.state_space_size = self.calculate_state_space_size()
        self.action_space_size = 6  # Number of possible actions (0 to 5)
        self.Q = {}  # Use a dictionary to store Q-values
        self.learning_rate = 0.1  # Adjust as needed
        self.discount_factor = 0.9  # Adjust as needed
        self.epsilon = 0.1  # Exploration vs. exploitation trade-off
        self.frame_delay = 0.02
        self.current_state = None

    def preprocess_state(self, state):
        # Convert state to a tuple to use as a dictionary key
        return tuple(state[0].flatten())

    def select_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        else:
            if state not in self.Q:
                self.Q[state] = np.zeros(self.action_space_size)
            return np.argmax(self.Q[state])

    def calculate_state_space_size(self):
        # Calculate the size of the state space based on the observation space shape
        return np.prod(self.env.observation_space.shape)

    def update_q_table(self, state, action, reward, next_state):
        if state not in self.Q:
            self.Q[state] = np.zeros(self.action_space_size)
        if next_state not in self.Q:
            self.Q[next_state] = np.zeros(self.action_space_size)

        max_next_action_value = np.max(self.Q[next_state])
        self.Q[state][action] = (1 - self.learning_rate) * self.Q[state][action] + \
            self.learning_rate * \
            (reward + self.discount_factor * max_next_action_value)

    def save_model(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.Q, file)

    def load_model(self, filename):
        with open(filename, 'rb') as file:
            self.Q = pickle.load(file)

    def run(self, episodes):
        for episode in range(episodes):
            self.current_state = self.preprocess_state(self.env.reset())
            total_reward = 0

            done = False
            while not done:
                action = self.select_action(self.current_state)
                next_state, reward, done, _, info = self.env.step(action)
                next_state = self.preprocess_state(next_state)

                self.update_q_table(
                    self.current_state, action, reward, next_state)

                self.current_state = next_state
                total_reward += reward

                if done and 'flag_get' in info and info['flag_get']:
                    # Save the model when Mario reaches the flag
                    self.save_model('mario_model.pkl')

                # time.sleep(self.frame_delay)

            print(f"Episode {episode + 1}: Total Reward = {total_reward}")

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

    mario_agent.run(episodes=100)
