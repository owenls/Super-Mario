import random
import numpy as np
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import gym
import pickle
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(self, env, max_life=3):
        self.env = JoypadSpace(env, COMPLEX_MOVEMENT)
        self.state_space_size = self.calculate_state_space_size()
        self.action_space_size = 6  # Number of actions available
        self.Q = {}  # Q table

        self.learning_rate = 0.78
        self.discount_factor = 0.89
        self.epsilon = 1.0  # Initial exploration rate
        self.min_epsilon = 0.1  # Minimum exploration rate
        self.decay_rate = 0.0001  # Rate at which exploration rate decays

        self.frame_delay = 1
        self.current_state = None
        self.done = True
        self.max_life = max_life
        self.life_count = max_life
        self.consecutive_stuck_frames = 0
        self.previous_x_pos = None

    def preprocess_state(self, obs):
        return tuple(obs[0].flatten())

    def select_action(self, obs):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        else:
            if obs not in self.Q:
                self.Q[obs] = np.zeros(self.action_space_size)
            return np.argmax(self.Q[obs])

    def calculate_state_space_size(self):
        return np.prod(self.env.observation_space.shape)

    def update_q_table(self, state, action, reward, obs, done, info):
        death_penalty = -500  # Penalty for dying
        fall_penalty = -800  # Penalty for falling into a gap
        goomba_kill_reward = 200
        turtle_kill_reward = 300
        if state not in self.Q:
            self.Q[state] = np.zeros(self.action_space_size)
        if obs not in self.Q:
            self.Q[obs] = np.zeros(self.action_space_size)

        if done and 'x_pos' in info:
            reward += info['x_pos'] * 10
            if 'flag_get' in info and info['flag_get']:
                reward += 100000
        if done and 'life' in info and info['life'] < self.life_count:
            reward -= death_penalty
        if 'enemy' in info and info['enemy'] == 'Goomba':
            reward += goomba_kill_reward
        if 'enemy' in info and info['enemy'] == 'Turtle':
            reward += turtle_kill_reward

       
        if done and 'x_pos' in info and info['x_pos'] == self.previous_x_pos:
            reward -= fall_penalty
            reward -= death_penalty
        max_next_action_value = np.max(self.Q[obs])
        self.Q[state][action] = (1 - self.learning_rate) * self.Q[state][action] + \
            self.learning_rate * \
            (reward + self.discount_factor * max_next_action_value)

        self.previous_x_pos = info.get('x_pos', None)

    def run(self, steps):
        highest_reward = 0  # Variable to store the highest reward achieved
        for step in range(steps):
            if self.done:
                self.current_state = self.preprocess_state(self.env.reset())
                self.done = False

            action = self.select_action(self.current_state)
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

                # Save model if the agent achieves a new highest reward
                if total_reward > highest_reward:
                    highest_reward = total_reward
                    if 'flag_get' in info and info['flag_get']:
                        model_filename = f'models/run_2/mario_model_{step}_finish_reward_{total_reward}.pkl'
                    else:
                        model_filename = f'models/run_2/mario_model_{step}_reward_{total_reward}.pkl'
                    self.save_model(model_filename)

            if step % 100 == 0:
                print(f"Step: {step}, Total Reward: {np.sum(list(self.Q.values()))}, Epsilon: {self.epsilon}")

            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon - self.decay_rate)

        plt.plot(list(self.Q.values()))
        plt.xlabel('Step')
        plt.ylabel('Total Reward')
        plt.title('Rewards over Steps')
        plt.show()

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
    mario_agent = QLearningAgent(env)

    try:
        mario_agent.load_model('models/mario_model_96786_reward_862.3364495865121.pkl')
        print("Loaded saved model.")
    except FileNotFoundError:
        print("No saved model found.")

    num_steps = 10000000  # Set the number of steps for the run
    mario_agent.run(num_steps)

    steps_range = range(0, num_steps + 1, 100)  # Adjust the step interval for the x-axis
    plt.plot(steps_range, [np.sum(list(mario_agent.Q.values())[:step]) for step in steps_range])
    plt.xlabel('Steps')
    plt.ylabel('Total Reward')
    plt.title(f'Graph of Run with {num_steps} Steps')
    plt.show()
