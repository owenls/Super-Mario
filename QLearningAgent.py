# Q-Learning Agent

import time
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np



env = gym.make("SuperMarioBros-v0",
                            apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)
done = True
env.reset()
num_states = env.observation_space.shape[0]
num_actions = len(SIMPLE_MOVEMENT)

q_table = np.zeros((num_states, num_actions))
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.1  # Exploration rate
num_episodes = 1000 # FOR TRAINING, HOW MUCH IS NEEADED TO RUN IT
for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        # WTF?
        # if np.random.rand() < epsilon:
        #     action = env.action_space.sample()
        # else:
        #     action = np.argmax(q_table[state])


        next_state, reward, done, _ = env.step(action)

        # Q-learning update
        q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])

        state = next_state

env.close()
