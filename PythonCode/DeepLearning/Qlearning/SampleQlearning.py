import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

###############################################################################
# 1. Environment
###############################################################################
class SimpleGame:
    """
    A simple 1D board environment.
    
    State representation:
      - Discrete positions from 0 to 4 (inclusive).
      - We start at position 2.
      - Goal position is 4.

    Actions: 
      - [-1, 1] corresponding to move left or move right.
    """
    def __init__(self):
        self.state_size = 5
        self.action_space = [-1, 1]
        self.reset()  # set initial state

    def reset(self):
        self.state = 2  # start from center
        self.goal = 4
        return self.state

    def step(self, action):
        next_state = self.state + action

        # Determine reward and whether game is done
        if next_state == self.goal:
            reward = 10
            done = True
        elif next_state < 0 or next_state >= self.state_size:
            reward = -5
            done = True
        else:
            reward = -1
            done = False

        # Only update state if still in valid game
        if not done:
            self.state = next_state

        return self.state, reward, done

###############################################################################
# 2. Q-Network
###############################################################################
class QNetwork(nn.Module):
    """
    A simple feed-forward network that outputs Q-values for a discrete action set.
    """
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, action_size)
        )

    def forward(self, x):
        return self.model(x)

###############################################################################
# 3. Utility Functions
###############################################################################
def one_hot_encode(state, state_size):
    """
    Returns a one-hot encoded tensor given a discrete state index.
    """
    state_tensor = torch.zeros(state_size)
    state_tensor[state] = 1.0
    return state_tensor

def choose_action(q_net, state_tensor, epsilon):
    """
    Chooses an action index using an epsilon-greedy policy.

    - q_net: QNetwork (nn.Module)
    - state_tensor: one-hot tensor of shape [state_size]
    - epsilon: float, probability of random action
    """
    if random.random() < epsilon:
        # Random exploration
        action_idx = random.choice([0, 1])
    else:
        # Exploit best action according to QNetwork
        with torch.no_grad():
            q_values = q_net(state_tensor)
            action_idx = torch.argmax(q_values).item()
    return action_idx

def compute_td_loss(q_net, optimizer, state_tensor, action_idx, reward, done, next_state_tensor, gamma):
    """
    Computes TD loss and performs a gradient update (backprop).

    - q_net: QNetwork
    - optimizer: torch optimizer
    - state_tensor: one-hot vector for current state
    - action_idx: chosen action index
    - reward: scalar reward obtained from environment
    - done: boolean, whether episode ended
    - next_state_tensor: one-hot for next state
    - gamma: discount factor
    """
    # Current Q-values for the current state
    q_values = q_net(state_tensor)

    # Compute target using the Bellman equation
    with torch.no_grad():
        next_q_values = q_net(next_state_tensor)
        max_next_q = torch.max(next_q_values).item()
        target = reward + (gamma * max_next_q if not done else 0.0)

    # TD loss = (Q(s,a) - target)^2
    loss = (q_values[action_idx] - target) ** 2

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

###############################################################################
# 4. Training Loop
###############################################################################
def train_agent(
    env,
    q_net,
    optimizer,
    num_episodes=10000,
    gamma=0.999,
    epsilon=0.5,
    print_every=500
):
    """
    Trains a Q-network with an epsilon-greedy policy on the given environment.

    - env: Environment (must have reset() and step(action) methods)
    - q_net: QNetwork to be trained
    - optimizer: Torch optimizer for QNetwork
    - num_episodes: total episodes
    - gamma: discount factor
    - epsilon: exploration rate
    - print_every: print progress every N episodes
    """
    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            # Convert state to one-hot
            state_tensor = one_hot_encode(state, env.state_size)
            
            # Choose action using epsilon-greedy
            action_idx = choose_action(q_net, state_tensor, epsilon)
            action = env.action_space[action_idx]

            # Take step in environment
            next_state, reward, done = env.step(action)

            # Convert next state to one-hot
            next_state_tensor = one_hot_encode(next_state, env.state_size)

            # Compute TD loss and update Q-network
            loss_value = compute_td_loss(
                q_net,
                optimizer,
                state_tensor,
                action_idx,
                reward,
                done,
                next_state_tensor,
                gamma
            )

        # Print stats
        if episode % print_every == 0:
            print(f"🎯 Episodio {episode} completado — Loss: {loss_value}")

    print("✅ Entrenamiento finalizado")

###############################################################################
# 5. Main Script
###############################################################################
if __name__ == "__main__":
    # Initialize environment
    env = SimpleGame()

    # Create Q-network and optimizer
    q_net = QNetwork(state_size=env.state_size, action_size=len(env.action_space))
    optimizer = optim.Adam(q_net.parameters(), lr=0.01)

    # Hyperparameters
    gamma = 0.999
    epsilon = 0.5
    num_episodes = 10000

    # Train your agent
    train_agent(env, q_net, optimizer, num_episodes, gamma, epsilon)

    # After training, you can evaluate or save your model here...
