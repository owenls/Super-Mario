import time
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
env = gym.make("SuperMarioBros-v0",
               apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)
done = True
env.reset()

frame_delay = 0.01
right_action = 1
jump_action = 5

jump_interval = 30

step_counter = 0
jump_counter = 0


for step in range(7000):
    if done:
        state = env.reset()

    obs, reward, terminated, truncated, info = env.step(right_action)
    done = terminated or truncated

    if jump_counter >= jump_interval:
        obs, reward, terminated, truncated, info = env.step(jump_action)
        done = terminated or truncated
        jump_counter = 0

    step_counter += 1
    jump_counter += 1

    time.sleep(frame_delay)


env.close()
