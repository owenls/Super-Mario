# Author 1: Owen Smith (22957291)
# Author 2: John Lumagbas (23419439)
import time
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym


class QLearningAgent:
    def __init__(self):
        self.env = gym.make("SuperMarioBros-v0",
                            apply_api_compatibility=True, render_mode="human")
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
        self.done = True
        self.env.reset()
        self.frame_delay = 0.02

    def run(self):
        for step in range(10000):
            if self.done:
                state = self.env.reset()

            obs, reward, terminated, truncated, info = self.env.step(1)
            self.done = terminated or truncated

            time.sleep(self.frame_delay)

        self.env.close()


# Usage of the QLearningAgent class
if __name__ == "__main__":
    QLearning_mario = QLearningAgent()
    QLearning_mario.run()
