import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm as tqdm


class RunningMeanStd:
    def __init__(self, shape, eps=1e-4, clip=5.0):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps
        self.clip = float(clip)
        self.eps = 1e-8

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        if batch_count == 0:
            return
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * (batch_count / tot_count)
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + (delta**2) * (self.count * batch_count / tot_count)
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def update(self, x_batch):
        x = np.asarray(x_batch, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def normalize(self, x):
        x_arr = np.asarray(x, dtype=np.float64)
        shape_was_1d = False
        if x_arr.ndim == 1:
            x_arr = x_arr[None, :]
            shape_was_1d = True
        std = np.sqrt(self.var + self.eps)
        out = (x_arr - self.mean) / std
        out = np.clip(out, -self.clip, self.clip)
        out = out.astype(np.float32)
        return out[0] if shape_was_1d else out

    def denormalize(self, x):
        x_arr = np.asarray(x, dtype=np.float64)
        shape_was_1d = False
        if x_arr.ndim == 1:
            x_arr = x_arr[None, :]
            shape_was_1d = True
        std = np.sqrt(self.var + self.eps)
        out = x_arr * std + self.mean
        out = out.astype(np.float32)
        return out[0] if shape_was_1d else out

    def state_dict(self):
        return {
            "mean": torch.tensor(self.mean),
            "var": torch.tensor(self.var),
            "count": torch.tensor(self.count)
        }
    
    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"].cpu().numpy()
        self.var = state_dict["var"].cpu().numpy()
        self.count = state_dict["count"].cpu().numpy()


class Actor(nn.Module):
    def __init__(self, input_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), 
                                 nn.Linear(64, 64), nn.ReLU(), 
                                 nn.Linear(64, 64), nn.ReLU(), 
                                 nn.Linear(64, action_dim))
        self.max_action = max_action

    def forward(self, x): 
        preact = self.net(x)
        action = preact.tanh() * self.max_action
        return action, preact

class Critic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim + action_dim, 64), nn.ReLU(), 
                                 nn.Linear(64, 64), nn.ReLU(), 
                                 nn.Linear(64, 64), nn.ReLU(), 
                                 nn.Linear(64, 1))
    def forward(self, state, action):
        return self.net(torch.cat([state, action], 1))


class DDPGAgent:
    def __init__(self, state_dim, goal_dim, action_dim, max_action, device="cuda"):
        self.device = device
        input_dim = state_dim + goal_dim
        
        self.actor = Actor(input_dim, action_dim, max_action)
        self.actor_target = Actor(input_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        
        self.critic = Critic(input_dim, action_dim)
        self.critic_target = Critic(input_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        self.max_action = max_action

        self.normalizer = RunningMeanStd(input_dim)
        self.lambda_preact = 1

        self.to(device)

    def to(self, device):
        self.device = device
        self.actor.to(device)
        self.actor_target.to(device)
        self.critic.to(device)
        self.critic_target.to(device)
        return self

    def select_action(self, obs):
        obs = self.normalizer.normalize(obs)
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.actor(obs)
            action = action.cpu().data.numpy().flatten()

        action = action.clip(-self.max_action, self.max_action)
        return action
        
    def train_step(self, buffer, batch_size=256, gamma=0.98, tau=0.05, return_stats=False):
        if len(buffer) < batch_size:
            return
        
        obs, action, reward, next_obs, done = buffer.sample(batch_size)
        obs = torch.FloatTensor(self.normalizer.normalize(obs.cpu().numpy()))
        next_obs = torch.FloatTensor(self.normalizer.normalize(next_obs.cpu().numpy()))

        obs, action = obs.to(self.device), action.to(self.device)
        reward = reward.unsqueeze(1).to(self.device)
        next_obs = next_obs.to(self.device)
        done = done.unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            next_action, _ = self.actor_target(next_obs)
            target_q = reward + gamma * (1 - done) * self.critic_target(next_obs, next_action)
            target_q = torch.clamp(target_q, min=-1.0 / (1.0 - gamma), max=0.0)
            
        current_q = self.critic(obs, action)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        action_pred, preact = self.actor(obs)
        actor_loss = -self.critic(obs, action_pred).mean() + self.lambda_preact * (preact**2).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        if return_stats:
            return {
                "critic_loss": float(critic_loss.item()),
                "actor_loss": float(actor_loss.item()),
                "q_mean": float(current_q.mean().item()),
            }

