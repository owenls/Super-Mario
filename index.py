import time
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
from PIL import Image
import numpy as np

env = gym.make("SuperMarioBros-v0",
               apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)
done = True
env.reset()

frame_delay = 0.02
right_action = 1
left_action = 6
jump_action = 5

step_counter = 0
x_position = 0  # Initialize x_position

x_position_threshold = 1.0  # Adjust the threshold for X position change
stuck_threshold = 50  # Minimum consecutive steps to be considered stuck
stuck_counter = 0


def goombaCheck(obs):
    target_color = [228, 92, 16]
    # is_color_present = np.any(np.all(obs == target_color, axis=-1))
    is_color_present = np.all(obs[207] == target_color)

    if is_color_present:
        return True
    else:
        return False


for step in range(7000):
    if done:
        state = env.reset()

    obs, reward, terminated, truncated, info = env.step(right_action)
    done = terminated or truncated
    step_counter += 1
    current_x_position = info['x_pos']
    dx = abs(current_x_position - x_position)

    # Check if the character's X position has changed significantly
    if dx < x_position_threshold:
        stuck_counter += 1
    else:
        stuck_counter = 0

    # Check if the character is stuck for a consecutive number of steps
    if stuck_counter >= stuck_threshold and step_counter > 10:
        print("The character is stuck!")

        # Add logic to move left for a longer duration
        for move in range(10):  # Adjust the number of steps to move left
            obs, reward, terminated, truncated, info = env.step(left_action)
            done = terminated or truncated
            if done:
                state = env.reset()

        # Now jump to the right
        obs, reward, terminated, truncated, info = env.step(right_action)
        done = terminated or truncated

        # Wait for a moment
        time.sleep(1.0)  # Adjust the duration as needed

        # Then jump
        obs, reward, terminated, truncated, info = env.step(jump_action)
        done = terminated or truncated

        # Check for Goomba and jump if detected
    if goombaCheck(obs):
        obs, reward, terminated, truncated, info = env.step(jump_action)
        done = terminated or truncated

    x_position = current_x_position

    time.sleep(frame_delay)

env.close()
