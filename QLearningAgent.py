import time
import random
import numpy as np
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import gym


class QLearningAgent:
    def __init__(self, env):
        self.env = JoypadSpace(env, SIMPLE_MOVEMENT)
        self.state_space_size = self.calculate_state_space_size()
        self.action_space_size = 6  # Number of possible actions (0 to 5)
        self.Q = np.zeros((self.state_space_size, self.action_space_size))
        self.learning_rate = 0.01
        self.discount_factor = 0.8
        self.epsilon = 0.1  # Exploration vs. exploitation trade-off
        self.frame_delay = 0.02
        self.current_state = None

    def preprocess_state(self, state):
        return np.array(state[0])

    def select_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        else:
            return np.argmax(self.Q[state, :])

    def calculate_state_space_size(self):
        # Calculate the size of the state space based on observation space or other factors
        observation_space = self.env.observation_space.shape
        # Example: If you flatten the observation space, the state space size would be its length
        state_space_size = np.prod(observation_space)
        return state_space_size

    def update_q_table(self, state, action, reward, next_state):
        max_next_action_value = np.max(self.Q[next_state, :])
        self.Q[state, action] = (1 - self.learning_rate) * self.Q[state, action] + \
            self.learning_rate * \
            (reward + self.discount_factor * max_next_action_value)

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

                # time.sleep(self.frame_delay)

            print(f"Episode {episode + 1}: Total Reward = {total_reward}")

        self.env.close()


if __name__ == "__main__":
    env = gym.make("SuperMarioBros-v0",
                   apply_api_compatibility=True, render_mode="human")
    mario_agent = QLearningAgent(env)
    mario_agent.run(episodes=2000)
