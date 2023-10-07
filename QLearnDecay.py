import numpy as np
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import cv2
import pickle
# Actions for complex movement
ACTIONS = COMPLEX_MOVEMENT

# Preprocess frame function
def preprocess_frame(frame):
    # Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return gray

class QLearnDecay:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=1.0, decay=0.995):
        self.n_actions = n_actions
        self.q_table = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        predict = self.q_table[state, action]
        target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.alpha * (target - predict)

    def update_epsilon(self):
        self.epsilon *= self.decay

if __name__ == "__main__":
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, ACTIONS)

    n_states = 256  # This is a placeholder. Define the number of states you have after preprocessing.
    n_actions = len(ACTIONS)

    agent = QLearnDecay(n_states, n_actions)
    num_episodes = 1000

    for episode in range(num_episodes):
        state = preprocess_frame(env.reset())
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = preprocess_frame(next_state)
            agent.learn(state, action, reward, next_state)
            state = next_state

            # Update epsilon
            agent.update_epsilon()

            # Save the model
            if episode % 10000 == 0 or (done and info['flag_get']):
                filename = f"model_{episode}.pkl" if not info['flag_get'] else f"model_completed_{episode}.pkl"
                with open(filename, 'wb') as file:
                    pickle.dump(agent.q_table, file)

        print(f"Episode: {episode}, Epsilon: {agent.epsilon}")

    env.close()
