# Author 1: Owen Smith (22957291)
# Author 2: John Lumagbas (23419439)
import time
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


class RuleBasedMarioAgent:
    def __init__(self):
        self.env = gym.make("SuperMarioBros-v0",
                            apply_api_compatibility=True, render_mode="human")
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
        self.done = True
        self.env.reset()

        self.goomba = [228, 92, 16]
        self.level2Goomba = [0, 136, 136]
        self.pipe = [184, 248, 24]
        self.gold_box = [252, 160, 68]
        self.sky = [104, 136, 252]
        self.turtle = [252, 252, 252]
        self.frame_delay = 0.01
        self.right_action = 1
        self.jump_action = 5
        self.jump_and_go_right = 2
        self.jump_duration = 19
        self.stuck = False
        self.previous_x_pos = 10
        self.consecutive_stuck_frames = 0

    def shouldJump(self, obs, info):
        print(info['life'])
        # CHECK IF GOOMBA NEARBY
        target_color = self.goomba
        half_obs = obs[202][125:155]
        is_goomba_present = np.any(np.all(half_obs == target_color, axis=-1))
        if is_goomba_present:
            return True

        # CHECK FOR GOOMBA ON LEVEL 2 (Different Colour)
        target_color = self.level2Goomba
        half_obs = obs[202][125:165]
        is_goomba_present = np.any(np.all(half_obs == target_color, axis=-1))
        if is_goomba_present:
            return True

        # CHECK IF PIPE NEARBY
        target_color = self.pipe
        half_obs = obs[192][125:180]
        is_pipe_present = np.any(np.all(half_obs == target_color, axis=-1))
        if is_pipe_present:
            return True

        # CHECK IF FLOOR IS MISSING (if floor is color of sky)
        target_color = self.sky
        half_obs = obs[215][125:150]
        is_floor_gone = np.any(np.all(half_obs == target_color, axis=-1))
        if is_floor_gone:
            return True

        # CHECK IF TURTLE IS NEARBY
        target_color = self.turtle
        half_obs = obs[202][125:150]
        turtle_near = np.any(np.all(half_obs == target_color, axis=-1))
        if turtle_near:
            return True

        # Checks if Mario is STUCK on something that went undetected.
        if info['x_pos'] == self.previous_x_pos:
            self.consecutive_stuck_frames += 1
        else:
            self.consecutive_stuck_frames = 0
        stuck_threshold = 20
        if self.consecutive_stuck_frames >= stuck_threshold:
            return True

        self.previous_x_pos = info['x_pos']

        return False

    def run(self):
        for step in range(10000):
            if self.done:
                state = self.env.reset()

            obs, reward, terminated, truncated, info = self.env.step(
                self.right_action)
            self.done = terminated or truncated

            if self.done:
                state = self.env.reset()

            if self.shouldJump(obs, info):
                for _ in range(self.jump_duration):
                    obs, reward, terminated, truncated, info = self.env.step(
                        self.jump_and_go_right)
                    self.done = terminated or truncated
                    if self.done:
                        state = self.env.reset()
            time.sleep(self.frame_delay)

        self.env.close()


# Usage of the MarioAgent class
if __name__ == "__main__":
    rule_based_mario = RuleBasedMarioAgent()
    rule_based_mario.run()
