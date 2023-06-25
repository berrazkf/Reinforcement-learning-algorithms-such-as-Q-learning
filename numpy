# Example of Q-learning using a simple grid world environment
import numpy as np

# Define the grid world environment
grid = np.array([
    [-1, -1, -1, -1],
    [-1, -1, -1, -1],
    [-1, -1, -1, -1],
    [-1, -1, -1, 100]
])

# Define the Q-table
q_table = np.zeros_like(grid)

# Set hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1
episodes = 1000

# Q-learning algorithm
for episode in range(episodes):
    state = np.random.randint(0, 4)
    while state != 3:
        if np.random.uniform() < epsilon:
            action = np.random.randint(0, 4)
        else:
            action = np.argmax(q_table[state])
        next_state = action
        reward = grid[state, action]
        q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])
        state = next_state

# Print the learned Q-table
print("Q-table:")
print(q_table)
