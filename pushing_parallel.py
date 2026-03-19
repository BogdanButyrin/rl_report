import gymnasium as gym
import numpy as np
import torch
import tqdm as tqdm
from her import HERReplayBuffer
from gymnasium.vector import AsyncVectorEnv
from ddpg import DDPGAgent
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)


class FetchPushTrainer:
    def __init__(
        self,
        max_episode_steps=50,
        use_her=True,
        her_strategy="future",
        her_k=4,
        buffer_capacity=10**6,
        gamma=0.98,
        tau=1-(1-0.05)**(1/40),
        batch_size=128,
        noise_ratio=0.05,
        epsilon=0.2,
        cycles_per_epoch=10,
        episodes_per_cycle=16,
        updates_per_cycle=40,
        evals_per_epoch=80,
        device=None,
        num_envs=8,
    ):
        self.max_episode_steps = max_episode_steps
        self.use_her = use_her
        self.her_strategy = her_strategy
        self.her_k = her_k
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.noise_ratio = noise_ratio
        self.epsilon = epsilon
        self.cycles_per_epoch = cycles_per_epoch
        self.episodes_per_cycle = episodes_per_cycle
        self.updates_per_cycle = updates_per_cycle
        self.evals_per_epoch = evals_per_epoch
        self.num_envs = num_envs
        assert episodes_per_cycle % num_envs == 0
        assert evals_per_epoch % num_envs == 0

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        def make_env():
            return gym.make("FetchPush-v4", max_episode_steps=self.max_episode_steps)

        self.envs = AsyncVectorEnv([make_env for _ in range(self.num_envs)])

        
        self.single_env = make_env()
        self.max_episode_steps = self.max_episode_steps or getattr(self.single_env.spec, "max_episode_steps", 50)

        obs_space = self.single_env.observation_space
        self.state_dim = obs_space["observation"].shape[0]
        self.goal_dim = obs_space["desired_goal"].shape[0]
        self.action_dim = self.single_env.action_space.shape[0]
        self.max_action = float(self.single_env.action_space.high[0])

        self.agent = DDPGAgent(
            state_dim=self.state_dim,
            goal_dim=self.goal_dim,
            action_dim=self.action_dim,
            max_action=self.max_action,
            device=self.device,
        )

        base_env = self.single_env.unwrapped
        self.buffer = HERReplayBuffer(
            buffer_capacity,
            strategy=self.her_strategy,
            k=self.her_k,
            reward_func=base_env.compute_reward,
            agent_normalizer=self.agent.normalizer,
        )

        self.history = {
            "success_rate": [],
            "avg_episode_len": [],
            "avg_loss_critic": [],
            "avg_loss_actor": [],
            "avg_q": [],
            "eval_success_rate": [],
            "eval_avg_episode_len": [],
        }

    def _concat_state_goal(self, obs_dict):
        state = obs_dict["observation"]
        desired_goal = obs_dict["desired_goal"]
        return np.concatenate([state, desired_goal])


    def _select_action(self, obs, epsilon=0.0, noise=0.0):
        action = self.agent.select_action(obs)
        if np.random.rand() < epsilon:
            action = np.random.uniform(-self.max_action, self.max_action, (self.action_dim,))
        else:
            action = (action + np.random.normal(0, noise, size=action.shape)).clip(-self.max_action, self.max_action)
        return action
        

    def _extract_is_success(self, infos):
        if isinstance(infos, dict) and "is_success" in infos:
            return np.array(infos["is_success"], dtype=bool)
        if isinstance(infos, (list, tuple)):
            return np.array([info.get("is_success", False) for info in infos], dtype=bool)
        return np.zeros(self.num_envs, dtype=bool)

    def _collect_parallel_batch(self, noise=0.0, epsilon=0.0):
        obs, infos = self.envs.reset()
        episodes = [[] for _ in range(self.num_envs)]

        for t in range(self.max_episode_steps):
            actions = np.zeros((self.num_envs, self.action_dim), dtype=np.float32)
            for i in range(self.num_envs):
                obs_i = {
                    "observation": obs["observation"][i],
                    "desired_goal": obs["desired_goal"][i],
                }
                policy_input = self._concat_state_goal(obs_i)
                actions[i] = self._select_action(policy_input, noise=noise, epsilon=epsilon)

            next_obs, rewards, terminated, truncated, infos = self.envs.step(actions)
            rollout_done = np.logical_or(terminated, truncated)
            bellman_done = terminated.astype(bool)

            for i in range(self.num_envs):
                state = obs["observation"][i]
                desired_goal = obs["desired_goal"][i]
                next_state = next_obs["observation"][i]
                achieved_goal_next = next_obs["achieved_goal"][i]
                episodes[i].append((state, actions[i], rewards[i], next_state, achieved_goal_next, desired_goal, bellman_done[i]))

            obs = next_obs
            if np.all(rollout_done):
                break

        is_success = self._extract_is_success(infos)
        successes = int(is_success.sum())
        episode_lengths = [t + 1] * self.num_envs
        return episodes, episode_lengths, successes


    def train(self, epochs=300, log_callback=None):
        start_epoch = len(self.history["success_rate"])

        for local_epoch in tqdm.tqdm(range(epochs)):
            epoch = start_epoch + local_epoch
            successes = 0
            episode_lengths = []
            losses_critic = []
            losses_actor = []
            q_means = []

            for _ in range(self.cycles_per_epoch):
                for _ in range(self.episodes_per_cycle // self.num_envs):
                    episodes, lengths, succ = self._collect_parallel_batch(noise=self.noise_ratio * 2 * self.max_action, epsilon=self.epsilon)

                    successes += succ
                    episode_lengths.extend(lengths)

                    if self.use_her:
                        for ep in episodes:
                            self.buffer.push_episode(ep)
                    else:
                        for ep in episodes:
                            for transition in ep:
                                (s, a, r, s_next, _, g, d) = transition
                                self.buffer.push_transition(s, a, r, s_next, g, d)

                for _ in range(self.updates_per_cycle):
                    stats = self.agent.train_step(
                        self.buffer,
                        batch_size=self.batch_size,
                        gamma=self.gamma,
                        tau=self.tau,
                        return_stats=True,
                    )
                    if stats is not None:
                        losses_critic.append(stats["critic_loss"])
                        losses_actor.append(stats["actor_loss"])
                        q_means.append(stats["q_mean"])

            total_episodes = self.cycles_per_epoch * self.episodes_per_cycle
            success_rate = successes / total_episodes
            avg_ep_len = float(np.mean(episode_lengths)) if episode_lengths else float("nan")
            avg_loss_critic = float(np.mean(losses_critic)) if losses_critic else float("nan")
            avg_loss_actor = float(np.mean(losses_actor)) if losses_actor else float("nan")
            avg_q = float(np.mean(q_means)) if q_means else float("nan")
            eval_metrics = self.evaluate(num_episodes=self.evals_per_epoch)

            self.history["success_rate"].append(success_rate)
            self.history["avg_episode_len"].append(avg_ep_len)
            self.history["avg_loss_critic"].append(avg_loss_critic)
            self.history["avg_loss_actor"].append(avg_loss_actor)
            self.history["avg_q"].append(avg_q)
            self.history["eval_success_rate"].append(eval_metrics["success_rate"])
            self.history["eval_avg_episode_len"].append(eval_metrics["avg_episode_len"])

            if log_callback is not None:
                log_callback(epoch, self.history)


        return self.history

    def evaluate(self, num_episodes=100):
        if num_episodes == 0:
            return {"success_rate": float("nan"), "avg_episode_len": float("nan")}
        
        assert num_episodes % self.num_envs == 0
        was_training_actor = self.agent.actor.training
        was_training_critic = self.agent.critic.training
        self.agent.actor.eval()
        self.agent.critic.eval()

        successes = 0
        episode_lengths = []

        for _ in tqdm.tqdm(range(num_episodes // self.num_envs)):
            episodes, lengths, succ = self._collect_parallel_batch(noise=0, epsilon=0)
            successes += succ
            episode_lengths.extend(lengths)

        if was_training_actor:
            self.agent.actor.train()
        if was_training_critic:
            self.agent.critic.train()

        return {
            "success_rate": successes / num_episodes,
            "avg_episode_len": float(np.mean(episode_lengths)) if episode_lengths else float("nan"),
        }

    def close(self):
        self.envs.close()
        self.single_env.close()
        
    def save_model(self, path):
        self.agent.to("cpu")
        torch.save({
            "actor_state_dict": self.agent.actor.state_dict(),
            "critic_state_dict": self.agent.critic.state_dict(),
            "target_actor_state_dict": self.agent.actor_target.state_dict(),
            "target_critic_state_dict": self.agent.critic_target.state_dict(),
            "normalizer_state_dict": self.agent.normalizer.state_dict(),
            "history": self.history,
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.agent.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.agent.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.agent.actor_target.load_state_dict(checkpoint["target_actor_state_dict"])
        self.agent.critic_target.load_state_dict(checkpoint["target_critic_state_dict"])
        self.agent.normalizer.load_state_dict(checkpoint["normalizer_state_dict"])
        self.agent.to(self.device)
        self.history = checkpoint["history"]
