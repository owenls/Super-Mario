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

frame_delay = 0.03
right_action = 1
jump_action = 5
jump_and_go_right = 2

jump_duration = 20


def shouldJump(obs, info):
    pos = info['x_pos']  # Mario's current 'x' position
    gap = pos+2  # The gap in front that we chek to see if should jump
    target_color = [228, 92, 16]
    # Select the second half of the array at y-coordinate 206 (index 205)
    half_obs = obs[202][pos:gap]
    # Check if any pixel in the selected region matches the target color
    is_goomba_present = np.any(np.all(half_obs == target_color, axis=-1))
    if is_goomba_present:
        return True

    target_color = [184, 248, 24]
    # Select the second half of the array at y-coordinate 206 (index 205)
    half_obs = obs[186][pos:gap]
    # Check if any pixel in the selected region matches the target color
    is_pipe_present = np.any(np.all(half_obs == target_color, axis=-1))
    return is_pipe_present


for step in range(7000):
    if done:
        state = env.reset()

    obs, reward, terminated, truncated, info = env.step(right_action)
    done = terminated or truncated

    if shouldJump(obs, info):
        for _ in range(jump_duration):
            obs, reward, terminated, truncated, info = env.step(
                jump_and_go_right)
            done = terminated or truncated

    time.sleep(frame_delay)

env.close()
