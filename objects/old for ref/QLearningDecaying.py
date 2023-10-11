# EPSILON DECAYING VERSION
import time
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
        self.Q = {}

        self.learning_rate = 0.9
        self.discount_factor = 0.3
        self.epsilon = 1
        self.frame_delay = 1
        self.current_state = None
        self.done = True
        self.current_episode = 0
        self.max_life = max_life
        self.life_count = max_life
        self.fall_penalty = 100
        self.consecutive_stuck_frames = 0
        self.previous_x_pos = None
        self.stuck_penalty = 1000

    def preprocess_state(self, obs):
        return tuple(obs[0].flatten())

    def select_action(self, obs, episode):
        epsilon = max(0.01, 0.0 * episode)
        if random.uniform(0, 1) < epsilon:
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

        # jump_obstacle_reward = 1000
        # jump_enemy_reward = 900
        # collect_coin_reward = 5
        # level_complete_reward = 10000
        # unnecessary_jump_penalty = 600
        # time_penalty = 0.1
        # stuck_penalty = 400
        # death_penalty = 5000
        # jump_fail_penalty = 800

        # if done:
        #     if 'time' in info and info['time'] == 0:
        #         reward -= time_penalty * self.frame_delay
        #     elif 'x_pos' in info and info['x_pos'] == self.previous_x_pos:
        #         reward -= stuck_penalty
        #     elif 'life' in info and info['life'] < self.life_count:
        #         reward -= self.fall_penalty
        #     elif 'x_pos' in info and info['x_pos'] == self.previous_x_pos:
        #         reward -= jump_fail_penalty

        # if 'life' in info and info['life'] < self.life_count:
        #     reward -= death_penalty

        # if reward == 0 and not done:
        #     reward -= time_penalty * self.frame_delay

        # if 'x_pos' in info and info['x_pos'] == self.previous_x_pos:
        #     self.consecutive_stuck_frames += 1
        # else:
        #     self.consecutive_stuck_frames = 0
        # stuck_threshold = 5
        # if self.consecutive_stuck_frames >= stuck_threshold:
        #     reward -= stuck_penalty

        max_next_action_value = np.max(self.Q[obs])
        self.Q[state][action] = (1 - self.learning_rate) * self.Q[state][action] + \
            self.learning_rate * (reward + self.discount_factor * max_next_action_value)

        self.previous_x_pos = info.get('x_pos', None)

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
            self.life_count = 1
            completed = False

            while self.life_count > 0:
                if self.done:
                    self.current_state = self.preprocess_state(self.env.reset())
                    self.done = False
                
                while not self.done:
                    action = self.select_action(self.current_state, episode)
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    obs = self.preprocess_state(obs)

                    self.update_q_table(self.current_state, action, reward, obs, self.done, info)

                    self.current_state = obs
                    total_reward += reward

                    if 'life' in info and info['life'] < self.life_count:
                        self.done = True
                        self.life_count = 0
                        break

                    if 'flag_get' in info and info['flag_get']:
                        self.done = True
                        model_filename = f'mario_model_episode_{episode + 1}.pkl'
                        self.save_model(model_filename)
                        completed = True
                        break
    

            position = info.get('x_pos', -1)  # Default to -1 if 'x_pos' is not available
            print(f"Episode {episode + 1}: Total Reward = {total_reward}, Position = {position}, Completed: {completed}")

        plt.plot(total_reward)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Rewards over Episodes')
        plt.show()

        self.env.close()

if __name__ == "__main__":
    env = gym.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human")
    mario_agent = QLearningAgent(env)

    try:
        mario_agent.load_model('mario_1model_ep18.pkl')
        print("Loaded saved model.")
    except FileNotFoundError:
        print("No saved model found.")

    mario_agent.run(episodes=100)
