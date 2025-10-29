import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from story_env import StoryEnvGym

# ----------------- Q-Network -----------------
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_size),
        )

    def forward(self, x):
        return self.net(x)

# ----------------- DQN Agent -----------------
class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.9, lr=1e-3,
                 batch_size=32, buffer_size=5000):
        self.q_net = QNetwork(state_size, action_size)
        self.target_net = QNetwork(state_size, action_size)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(2)
        with torch.no_grad():
            q_vals = self.q_net(torch.tensor(state).float().unsqueeze(0))
        return torch.argmax(q_vals).item()

    def memorize(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def replay(self):
        if len(self.buffer) < self.batch_size:
            return
        batch = random.sample(self.buffer, self.batch_size)
        s, a, r, s2, done = zip(*batch)
        s = torch.tensor(np.array(s)).float()
        a = torch.tensor(a)
        r = torch.tensor(r).float()
        s2 = torch.tensor(np.array(s2)).float()
        done = torch.tensor(done).float()

        q_pred = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze()
        with torch.no_grad():
            q_next = self.target_net(s2).max(1)[0]
            q_target = r + self.gamma * q_next * (1 - done)

        loss = nn.MSELoss()(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

# ----------------- Training Loop -----------------
def train_dqn(episodes=100):
    env = StoryEnvGym("../data/story_sample.json")
    agent = DQNAgent(state_size=768, action_size=2)
    rewards = []

    for ep in range(episodes):
        s, _ = env.reset()
        total = 0
        done = False
        while not done:
            a = agent.act(s)
            s2, r, done, _, _ = env.step(a)
            agent.memorize(s, a, r, s2, done)
            agent.replay()
            s = s2
            total += r
        agent.update_target()
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        rewards.append(total)
        print(f"Episode {ep+1:03d} | Reward: {total:.2f} | Epsilon: {agent.epsilon:.3f}")

    torch.save(agent.q_net.state_dict(), "../models/saved_dqn.pt")
    print("✅ Training complete. Model saved.")
    return rewards

if __name__ == "__main__":
    train_dqn(episodes=50)
