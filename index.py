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

frame_delay = 0.02
right_action = 1
jump_action = 5

jump_interval = 30

step_counter = 0
jump_counter = 0
max_jump_duration = 20
distanceToJump = 30

def goombaCheck(obs):
    target_color = [228, 92, 16]
    # Select the second half of the array at y-coordinate 206 (index 205)
    half_obs = obs[202][128:200]
    # Check if any pixel in the selected region matches the target color
    is_goomba_present = np.any(np.all(half_obs == target_color, axis=-1))
    if is_goomba_present:
        return True

   
    return is_goomba_present


# Chec if pipe is detected
def pipeDetection(obs):
    target_color = [184, 248, 24]
    # Select the second half of the array at y-coordinate 206 (index 205)
    half_obs = obs[186][100:150]
    # Check if any pixel in the selected region matches the target color
    is_pipe_present = np.any(np.all(half_obs == target_color, axis=-1))

    if is_pipe_present:
        return True
    
    return is_pipe_present



for step in range(7000):
    if done:
        state = env.reset()

    obs, reward, terminated, truncated, info = env.step(right_action)
    done = terminated or truncated

    if goombaCheck(obs):
        obs, reward, terminated, truncated, info = env.step(jump_action)
        done = terminated or truncated
    

    if pipeDetection(obs):
        if not jumping:  
            for _ in range(4):  
                obs, reward, terminated, truncated, info = env.step(jump_action)
                done = terminated or truncated
            jumping = True
    else:
        jumping = False 

    step_counter += 1
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
