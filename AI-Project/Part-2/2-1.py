#For a 4x4 GridWorld with only a win terminal state and two blocking states positioned
#randomly, implement a simple Q-Learning agent.

import numpy as np
import random

class GridWorld:
    def __init__(self):
        self.size = 4
        self.win_state = (3, 3)  # goal cell
        self.block_states = [(1, 1), (2, 2)]  # blocked cells
        self.reset()

    def reset(self):
        self.agent_pos = (0, 0)  # start position
        return self.agent_pos

    def is_valid(self, pos):
        x, y = pos
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return False
        if pos in self.block_states:
            return False
        return True

    def step(self, action):
        x, y = self.agent_pos
        if action == 0:   # up
            new_pos = (x - 1, y)
        elif action == 1: # down
            new_pos = (x + 1, y)
        elif action == 2: # left
            new_pos = (x, y - 1)
        elif action == 3: # right
            new_pos = (x, y + 1)
        else:
            new_pos = (x, y)

        if self.is_valid(new_pos):
            self.agent_pos = new_pos

        reward = 0
        done = False
        if self.agent_pos == self.win_state:
            reward = 1
            done = True

        return self.agent_pos, reward, done

    def get_state_index(self, pos):
        # convert position to single index for Q-table
        return pos[0] * self.size + pos[1]

# Q-Learning parameters
alpha = 0.1     # learning rate
gamma = 0.9     # discount factor
epsilon = 0.2   # exploration rate
episodes = 1000

env = GridWorld()
q_table = np.zeros((env.size * env.size, 4))  # states x actions

for episode in range(episodes):
    state = env.reset()
    done = False

    while not done:
        state_idx = env.get_state_index(state)

        # Epsilon-greedy action selection
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, 3)
        else:
            action = np.argmax(q_table[state_idx])

        next_state, reward, done = env.step(action)
        next_state_idx = env.get_state_index(next_state)

        # Update Q-table
        q_table[state_idx, action] = q_table[state_idx, action] + alpha * (
            reward + gamma * np.max(q_table[next_state_idx]) - q_table[state_idx, action]
        )

        state = next_state

print("Training completed.")
print("Q-table:")
print(q_table)
