import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from her import HERReplayBuffer
import tqdm as tqdm
from dqn import DQNNet


class BitFlippingEnv:
    def __init__(self, n=50):
        self.n = n
        
    def reset(self):
        self.state = np.random.randint(2, size=self.n)
        self.goal = np.random.randint(2, size=self.n)
        return self.state.copy(), self.goal.copy()
        
    def step(self, action):
        self.state[action] = 1 - self.state[action]
        done = np.array_equal(self.state, self.goal)
        reward = 0.0 if done else -1.0
        return self.state.copy(), reward, done, self.state.copy()

def bitflip_reward_func(achieved_goal, desired_goal):
    return 0.0 if np.array_equal(achieved_goal, desired_goal) else -1.0


class BitflipDQNTrainer:
    def __init__(
        self,
        n=50,
        use_her=True,
        her_strategy='future',
        her_k=4,
        buffer_capacity=10**5,
        gamma=0.98,
        batch_size=256,
        epsilon=0.2,
        cycles_per_epoch=16,
        episodes_per_cycle=50,
        updates_per_cycle=40,
        evals_per_epoch=80,
        target_update_freq=250,
        learning_rate=1e-3,
        device=None,
    ):
        self.n = n
        self.use_her = use_her
        self.her_strategy = her_strategy
        self.her_k = her_k
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.cycles_per_epoch = cycles_per_epoch
        self.episodes_per_cycle = episodes_per_cycle
        self.updates_per_cycle = updates_per_cycle
        self.evals_per_epoch = evals_per_epoch
        self.target_update_freq = target_update_freq
        self.max_episode_steps = n

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = BitFlippingEnv(n=n)

        self.model = DQNNet(n * 2, n).to(self.device)
        self.target_model = DQNNet(n * 2, n).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.buffer = HERReplayBuffer(
            buffer_capacity,
            her_strategy,
            k=her_k,
            reward_func=bitflip_reward_func,
        )

        self.update_steps = 0
        self.history = {
            "success_rate": [],
            "avg_episode_len": [],
            "avg_loss": [],
            "avg_q": [],
            "eval_success_rate": [],
            "eval_avg_episode_len": [],
        }

    def _select_action(self, obs, epsilon):
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        if np.random.rand() < epsilon:
            return np.random.randint(self.n)
        with torch.no_grad():
            return self.model(obs_tensor).argmax().item()

    def _concat_state_goal(self, state, goal):
        return np.concatenate([state, goal])

    def train(self, epochs=50, log_callback=None):
        start_epoch = len(self.history["success_rate"])

        for local_epoch in tqdm.tqdm(range(epochs)):
            epoch = start_epoch + local_epoch
            successes = 0
            episode_lengths = []
            losses = []
            mean_qs = []

            for _ in range(self.cycles_per_epoch):
                for _ in range(self.episodes_per_cycle):
                    state, goal = self.env.reset()
                    episode = []
                    steps = 0

                    for _ in range(self.max_episode_steps):
                        steps += 1
                        obs = self._concat_state_goal(state, goal)
                        action = self._select_action(obs, self.epsilon)

                        next_state, reward, done, achieved_goal = self.env.step(action)
                        episode.append((state, action, reward, next_state, achieved_goal, goal, done))
                        state = next_state
                        if done:
                            successes += 1
                            break

                    episode_lengths.append(steps)

                    if self.use_her:
                        self.buffer.push_episode(episode)
                    else:
                        for transition in episode:
                            (s, a, r, s_next, _, g, d) = transition
                            self.buffer.push_transition(s, a, r, s_next, g, d)

                if len(self.buffer) > self.batch_size:
                    for _ in range(self.updates_per_cycle):
                        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)
                        obs = obs.to(self.device)
                        actions = actions.long().to(self.device)
                        rewards = rewards.to(self.device)
                        next_obs = next_obs.to(self.device)
                        dones = dones.to(self.device)

                        q_values = self.model(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
                        with torch.no_grad():
                            next_q = self.target_model(next_obs).max(1)[0]

                            target_q = rewards + self.gamma * (1 - dones) * next_q

                        loss = nn.MSELoss()(q_values, target_q)

                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                        self.update_steps += 1
                        if self.update_steps % self.target_update_freq == 0:
                            self.target_model.load_state_dict(self.model.state_dict())

                        losses.append(loss.item())
                        mean_qs.append(q_values.mean().item())

            episodes_total = self.cycles_per_epoch * self.episodes_per_cycle
            success_rate = successes / episodes_total
            avg_ep_len = float(np.mean(episode_lengths)) if episode_lengths else float("nan")
            avg_loss = float(np.mean(losses)) if losses else float("nan")
            avg_q = float(np.mean(mean_qs)) if mean_qs else float("nan")
            eval_metrics = self.evaluate(num_episodes=self.evals_per_epoch)

            self.history["success_rate"].append(success_rate)
            self.history["avg_episode_len"].append(avg_ep_len)
            self.history["avg_loss"].append(avg_loss)
            self.history["avg_q"].append(avg_q)
            self.history["eval_success_rate"].append(eval_metrics["success_rate"])
            self.history["eval_avg_episode_len"].append(eval_metrics["avg_episode_len"])

            if log_callback is not None:
                log_callback(epoch, self.history)

        return self.history

    def evaluate(self, num_episodes=100):
        if num_episodes == 0:
            return {"success_rate": float("nan"), "avg_episode_len": float("nan")}

        was_training = self.model.training
        self.model.eval()

        successes = 0
        episode_lengths = []
        for _ in tqdm.tqdm(range(num_episodes)):
            state, goal = self.env.reset()
            steps = 0
            for _ in range(self.max_episode_steps):
                steps += 1
                obs = self._concat_state_goal(state, goal)
                obs_tensor = torch.FloatTensor(obs).to(self.device)
                with torch.no_grad():
                    action = self.model(obs_tensor).argmax().item()
                next_state, reward, done, achieved_goal = self.env.step(action)
                state = next_state
                if done:
                    successes += 1
                    break
            episode_lengths.append(steps)

        if was_training:
            self.model.train()

        return {
            "success_rate": successes / num_episodes,
            "avg_episode_len": float(np.mean(episode_lengths)) if episode_lengths else float("nan"),
        }

    def close(self):
        self.env.close()
    

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.cpu().state_dict(),
            'target_model_state_dict': self.target_model.cpu().state_dict(),
            'history': self.history,
        }, path)
    

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict']).to(self.device)
        self.target_model.load_state_dict(checkpoint['target_model_state_dict']).to(self.device)
        self.history = checkpoint['history']
    