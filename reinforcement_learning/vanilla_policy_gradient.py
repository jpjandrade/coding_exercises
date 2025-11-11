import time
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gymnasium as gym


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 2):
        layers += [nn.Linear(sizes[j], sizes[j + 1]), activation()]
    layers += [nn.Linear(sizes[-2], sizes[-1]), output_activation()]
    return nn.Sequential(*layers)


def get_policy_action_distribution(model, observations):
    logits = model(observations)
    return Categorical(logits=logits)


def compute_loss(observations, actions, reward_weights, policy_model):
    """
    Policy gradient loss: negative expected return-weighted log-probabilities.
    Computes log-prob manually from network logits (no .log_prob from distributions).
    """
    observations = observations
    actions = actions
    # Baseline to reduce reward variance.
    reward_weights_normalized = reward_weights - reward_weights.mean()
    logp = get_policy_action_distribution(policy_model, observations).log_prob(actions)
    return -(logp * reward_weights_normalized).mean()


def sample_action(model, observations):
    observations = observations
    return get_policy_action_distribution(model, observations).sample().item()


def reward_to_go(rewards):
    """Compute reward-to-go for each time step in a trajectory."""
    n = len(rewards)
    rtg = [0] * n
    future_reward = 0.0
    for t in reversed(range(n)):
        future_reward += rewards[t]
        rtg[t] = future_reward
    return rtg


def train_epoch(
    env: gym.Env,
    policy_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
):
    batch = {
        "observations": [],
        "actions": [],
        "reward_weights": [],
        "episode_returns": [],
        "episode_lengths": [],
    }

    # First observation comes from starting distribution.
    observation, _ = env.reset()
    episode_rewards = []

    terminated, truncated = False, False

    while len(batch["observations"]) < batch_size:
        batch["observations"].append(observation.copy())
        actions = sample_action(
            policy_model, torch.tensor(observation, dtype=torch.float32)
        )
        observation, reward, terminated, truncated, _ = env.step(actions)
        batch["actions"].append(actions)
        episode_rewards.append(reward)

        if terminated or truncated:
            episode_return, episode_length = sum(episode_rewards), len(episode_rewards)
            batch["episode_returns"].append(episode_return)
            batch["episode_lengths"].append(episode_length)

            # Replace by [episode_return] * episode_length to disable reward to go.
            batch["reward_weights"] += list(reward_to_go(episode_rewards))
            observation, _ = env.reset()
            episode_rewards = []

    # Discard incomplete episodes: remove observations/actions beyond what we have reward weights for.
    # This is relevant if we ended the batch in the middle of an episode.
    num_observations = len(batch["observations"])
    num_reward_weights = len(batch["reward_weights"])
    if num_observations > num_reward_weights:
        batch["observations"] = batch["observations"][:num_reward_weights]
        batch["actions"] = batch["actions"][:num_reward_weights]
        batch["reward_weights"] = batch["reward_weights"][:num_reward_weights]

    optimizer.zero_grad()
    batch_loss = compute_loss(
        observations=torch.tensor(batch["observations"], dtype=torch.float32),
        actions=torch.tensor(batch["actions"], dtype=torch.int32),
        reward_weights=torch.tensor(batch["reward_weights"], dtype=torch.float32),
        policy_model=policy_model,
    )
    batch_loss.backward()
    optimizer.step()
    return batch_loss, batch["episode_returns"], batch["episode_lengths"]


def train(
    env_name="CartPole-v1", hidden_sizes=[32], lr=1e-2, epochs=500, batch_size=50000
):
    env = gym.make(env_name)

    observations_dim = env.observation_space.shape[0]
    number_actions = env.action_space.n

    # Instantiate the policy function (which is a neural network)
    policy_model = mlp(sizes=[observations_dim] + hidden_sizes + [number_actions])
    policy_model  # Move model to GPU
    optimizer = Adam(policy_model.parameters(), lr=lr)
    for epoch in range(epochs):
        start_time = time.time()
        batch_loss, batch_returns, batch_lengths = train_epoch(
            env, policy_model, optimizer, batch_size
        )

        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {batch_loss.item():.3f}, "
            f"Return: {np.mean(batch_returns):.3f}, Length: {np.mean(batch_lengths):.3f}, Time: {epoch_time:.3f}s"
        )

    env.close()
