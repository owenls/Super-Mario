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
def plot_rewards(step_rewards, episode_rewards):
    plt.figure(figsize=(14, 7))
    
    # Plot for steps
    plt.subplot(1, 2, 1)
    plt.plot(step_rewards, label='Rewards per Step')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    
    # Plot for episodes
    plt.subplot(1, 2, 2)
    plt.plot(episode_rewards, label='Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def save_to_file(filename, episode, reward):
    with open(filename, 'a') as file:
        file.write(f"Episode {episode + 1}: Total Reward = {reward}\n")
class RuleBasedMarioAgent:

    # Initializes the RuleBasedMarioAgent.
    #   - Sets up the Super Mario environment.
    #   - Defines colors for game elements.
    #   - Specifies agent actions, frame delay, and other parameters.
    def __init__(self):
        self.env = gym.make("SuperMarioBros-1-1-v0",
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

    # Determines whether Mario should perform a jump action based on the observed game state.
    #   - Checks for nearby goombas, pipes, missing floor, and turtles.
    #   - Monitors Mario for getting stuck.

    # Args:
    #    obs (numpy.ndarray): The observed game screen.
    #    info (dict): Additional game information.

    # Returns:
    #    bool: True if Mario should jump, False otherwise.
    def shouldJump(self, obs, info):
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

    # Runs the Rule-Based Mario Agent in the Super Mario Bros. environment.
    #   - Executes actions to move Mario right.
    #   - Decides to jump based on the 'shouldJump' function.

    def run(self, total_steps):
        steps = 0
        episode_count = 0
        episode_rewards = []
        step_rewards = []
        total_reward = 0
        while steps < total_steps:
            if self.done:
                self.env.reset()
                episode_count += 1
                episode_rewards.append(total_reward)
                print(f"Episode {episode_count}, Total Reward = {total_reward}")
                total_reward = 0

            obs, reward, terminated, truncated, info = self.env.step(self.right_action)
            
            if self.shouldJump(obs, info):
                for _ in range(self.jump_duration):
                    if terminated or truncated:
                        break
                    obs, reward, jump_terminated, jump_truncated, info = self.env.step(self.jump_and_go_right)
                    terminated = terminated or jump_terminated
                    truncated = truncated or jump_truncated
                    total_reward += reward
                    steps += 1
                    step_rewards.append(total_reward)
                    time.sleep(self.frame_delay)

            if "flag_get" in info and info["flag_get"]:
                self.done = True
                continue

            total_reward += reward
            steps += 1
            step_rewards.append(total_reward)
            
            # Mark as done if Mario terminated or truncated
            self.done = terminated or truncated
            
            time.sleep(self.frame_delay)

        self.env.close()
        return step_rewards, episode_rewards


# Usage of the MarioAgent class
if __name__ == "__main__":
    rule_based_mario = RuleBasedMarioAgent()

    total_steps = 1000000
    step_rewards, episode_rewards = rule_based_mario.run(total_steps=total_steps)

    # Plotting and saving the rewards
    plot_rewards(step_rewards, episode_rewards)
    for episode, reward in enumerate(episode_rewards):
        save_to_file(f'rb_run_episodes_{len(episode_rewards)}.txt', episode, reward)