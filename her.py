import inspect
import numpy as np
import random
import torch

class HERReplayBuffer:
    def __init__(self, capacity, strategy='future', k=4, reward_func=None, agent_normalizer=None):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.strategy = strategy
        self.k = k
        self.reward_func = reward_func
        if self.strategy not in ("final", "future"):
            raise ValueError(f"Unknown HER strategy: {self.strategy}")
        if self.reward_func is None:
            raise ValueError("reward_func must be provided for HER relabeling.")
        try:
            self._reward_uses_info = (len(inspect.signature(self.reward_func).parameters) >= 3)
        except (TypeError, ValueError):
            self._reward_uses_info = False
        
        self.agent_normalizer = agent_normalizer

    def push_episode(self, episode_transitions):
        T = len(episode_transitions)

        for t in range(T):
            (state, action, reward, next_state,
             achieved_goal_next, desired_goal, done) = episode_transitions[t]

            self._add_transition(state, action, reward, next_state, desired_goal, done)

            additional_goals = []
            if self.strategy == 'final':
                additional_goals = [episode_transitions[-1][4]] * self.k
            elif self.strategy == 'future':
                future_indices = np.random.randint(t, T, size=self.k)
                additional_goals = [episode_transitions[idx][4] for idx in future_indices]

            for new_goal in additional_goals:
                if self._reward_uses_info:
                    new_reward = self.reward_func(achieved_goal_next, new_goal, None)
                else:
                    new_reward = self.reward_func(achieved_goal_next, new_goal)
                new_done = float(new_reward == 0.0)
                self._add_transition(state, action, new_reward, next_state, new_goal, new_done)


    def push_transition(self, state, action, reward, next_state, goal, done):
        self._add_transition(state, action, reward, next_state, goal, done)

    def _add_transition(self, state, action, reward, next_state, goal, done):
        obs = np.concatenate([state, goal])
        next_obs = np.concatenate([next_state, goal])

        transition = (obs, action, reward, next_obs, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

        if self.agent_normalizer is not None:
            self.agent_normalizer.update(obs)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = map(np.array, zip(*batch))
        return (
            torch.FloatTensor(obs),
            torch.FloatTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_obs),
            torch.FloatTensor(dones),
        )

    def __len__(self):
        return len(self.buffer)
