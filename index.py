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

goomba = [228, 92, 16]
pipe = [184, 248, 24]
gold_box = [252, 160, 68]
sky = [104, 136, 252]
frame_delay = 0.03
right_action = 1
jump_action = 5
jump_and_go_right = 2
jump_duration = 20

# Boolean function saying if mario should jump or not.


def shouldJump(obs, info):
    # CHECK IF GOOMBA NEARBY
    target_color = goomba
    half_obs = obs[202][125:165]
    # Check if any pixel in the selected region matches the target color
    is_goomba_present = np.any(np.all(half_obs == target_color, axis=-1))
    if is_goomba_present:
        return True

    # CHECK IF GOOMBA NEARBY
    target_color = pipe
    half_obs = obs[186][125:150]
    # Check if any pixel in the selected region matches the target color
    is_pipe_present = np.any(np.all(half_obs == target_color, axis=-1))
    if is_pipe_present:
        return True

    # CHECK IF FLOOR IS MISSING ( if floor is colour of sky)
    target_color = sky
    half_obs = obs[215][125:150]
    # Check if any pixel in the selected region matches the target color
    is_floor_gone = np.any(np.all(half_obs == target_color, axis=-1))
    if is_floor_gone:
        return True

    return False


for step in range(7000):
    if done:
        state = env.reset()

    obs, reward, terminated, truncated, info = env.step(right_action)
    done = terminated or truncated

    if done:
        state = env.reset()

    if shouldJump(obs, info):
        for _ in range(jump_duration):
            obs, reward, terminated, truncated, info = env.step(
                jump_and_go_right)
            done = terminated or truncated

    time.sleep(frame_delay)

    # plt.imshow(obs)
    # plt.grid(True)
    # plt.show()
    # # Convert obs to an Image object
    # obs_image = Image.fromarray(obs)

    # # Save the image to a file (optional)
    # obs_image.save("obs.png")

    # # Display the image (optional)
    # obs_image.show()

env.close()
