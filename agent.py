import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple

# Гиперпараметры
ACTION_SPACE = 5
STATE_SIZE = 10
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class Agent:
    def __init__(self, network):
        self.net = network
        self.replay_buffer = deque(maxlen=10000)
        self.buffer = ReplayBuffer()
        self.optimizer = optim.Adam(self.net.parameters(), lr=LR)
        self.epsilon = 0.2

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, ACTION_SPACE - 1)
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            q_values = self.net(state)
            return q_values.argmax().item()

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=torch.device("cpu")))

    def store(self, *args):
        self.buffer.push(*args)

    def train(self, BATCH_SIZE=64, gamma=0.95):
        if len(self.buffer) < BATCH_SIZE:
            return
        batch = self.buffer.sample(BATCH_SIZE)
        batch = Transition(*zip(*batch))

        states = torch.FloatTensor(batch.state).to(DEVICE)
        actions = torch.LongTensor(batch.action).unsqueeze(1).to(DEVICE)
        rewards = torch.FloatTensor(batch.reward).unsqueeze(1).to(DEVICE)
        next_states = torch.FloatTensor(batch.next_state).to(DEVICE)

        q_values = self.net(states).gather(1, actions)
        next_q = self.net(next_states).max(1)[0].detach().unsqueeze(1)
        expected = rewards + GAMMA * next_q

        loss = nn.MSELoss()(q_values, expected)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
