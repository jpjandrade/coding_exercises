import time
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gymnasium as gym


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    """
    Build a simple neural network.

    Args:
        sizes: List of layer sizes including input and output dimensions.
        activation: Activation function class to use for hidden layers.
        output_activation: Activation function class for the output layer.

    Returns:
        nn.Sequential: A PyTorch sequential model.
    """
    layers = []
    for j in range(len(sizes) - 2):
        layers += [nn.Linear(sizes[j], sizes[j + 1]), activation()]
    layers += [nn.Linear(sizes[-2], sizes[-1]), output_activation()]
    return nn.Sequential(*layers)


def get_policy_action_distribution(model, observations):
    """
    Get the action probability distribution from the policy model.

    Args:
        model: Policy neural network that outputs action logits.
        observations: Tensor with the environment observations.

    Returns:
        Categorical: A categorical distribution over actions.
    """
    logits = model(observations)
    return Categorical(logits=logits)


def sample_action(model, observations):
    """
    Sample an action from the policy model given observations.

    Args:
        model: Policy neural network that outputs action logits.
        observations: Tensor of environment observations.

    Returns:
        int: Sampled action index.
    """
    return get_policy_action_distribution(model, observations).sample().item()


def compute_loss(observations, actions, advantages, policy_model):
    """
    Actor-Critic policy gradient loss using advantages.

    Computes the policy loss by weighting log-probabilities with advantages
    (returns - value baseline). The value estimates are detached to prevent
    gradients from flowing back through the critic.

    Args:
        observations: Tensor of environment observations.
        actions: Tensor of actions taken.
        returns: Tensor of returns (reward-to-go or normalized returns).
        values: Tensor of value function estimates V(s).
        policy_model: Policy neural network (the actor).

    Returns:
        Tensor: The policy gradient loss.
    """
    # Calculate advantage by subtracting value function from returns.
    logp = get_policy_action_distribution(policy_model, observations).log_prob(actions)
    return -(logp * advantages).mean()


def train_epoch(
    env: gym.Env,
    policy_model: nn.Module,
    policy_optimizer: torch.optim.Optimizer,
    value_model: nn.Module,
    value_optimizer: torch.optim.Optimizer,
    batch_size: int,
    gamma: float = 0.99,
):
    """
    Train both actor and critic for one epoch using 1-step TD.

    Args:
        env: Gymnasium environment to interact with.
        policy_model: Policy neural network (actor) to train.
        policy_optimizer: PyTorch optimizer for updating the policy.
        value_model: Value neural network (critic) to train.
        value_optimizer: PyTorch optimizer for updating the value function.
        batch_size: Number of timesteps to collect before updating.
        gamma: Discount factor for future rewards.

    Returns:
        tuple: (batch_loss, value_loss, episode_returns, episode_lengths)
    """
    batch = {
        "observations": [],
        "next_observations": [],
        "values": [],
        "actions": [],
        "rewards": [],
        "dones": [],
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
        next_observation, reward, terminated, truncated, _ = env.step(actions)
        batch["actions"].append(actions)
        batch["rewards"].append(reward)
        batch["next_observations"].append(next_observation.copy())
        batch["dones"].append(terminated or truncated)

        episode_rewards.append(reward)

        observation = next_observation

        if terminated or truncated:
            episode_return, episode_length = sum(episode_rewards), len(episode_rewards)
            batch["episode_returns"].append(episode_return)
            batch["episode_lengths"].append(episode_length)
            observation, _ = env.reset()
            episode_rewards = []

    # Convert batch data to tensors (done once for efficiency)
    obs_tensor = torch.tensor(batch["observations"], dtype=torch.float32)
    next_obs_tensor = torch.tensor(batch["next_observations"], dtype=torch.float32)
    act_tensor = torch.tensor(batch["actions"], dtype=torch.int32)
    rewards_tensor = torch.tensor(batch["rewards"], dtype=torch.float32)
    dones_tensor = torch.tensor(batch["dones"], dtype=torch.float32)

    with torch.no_grad():
        values = value_model(obs_tensor).squeeze(-1)
        next_values = value_model(next_obs_tensor).squeeze(-1)

    # Also known as TD error or 1-step advantage.
    td_target = rewards_tensor + gamma * next_values * (1 - dones_tensor)
    advantages = td_target - values

    # Normalize advantages for policy stability
    advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Step 1: Update policy network (i.e., the actor).
    policy_optimizer.zero_grad()
    batch_loss = compute_loss(
        observations=obs_tensor,
        actions=act_tensor,
        advantages=advantages_normalized,
        policy_model=policy_model,
    )
    batch_loss.backward()
    # Clip gradients to prevent exploding gradients.
    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=0.5)
    policy_optimizer.step()

    # Step 2: Update value network (i.e., the critic).
    # Recompute values to get fresh computation graph for value loss
    value_optimizer.zero_grad()
    values = value_model(obs_tensor).squeeze(-1)
    value_loss = nn.functional.mse_loss(
        values, rewards_tensor + gamma * next_values * (1 - dones_tensor)
    )
    value_loss.backward()
    torch.nn.utils.clip_grad_norm_(value_model.parameters(), max_norm=0.5)
    value_optimizer.step()

    return batch_loss, value_loss, batch["episode_returns"], batch["episode_lengths"]


def train(
    env_name="CartPole-v1",
    hidden_sizes=[64],
    lr=1e-3,
    epochs=500,
    batch_size=5000,
    gamma=0.99,
    seed=None,
):
    """
    Train a policy using Actor-Critic with 1-step TD.

    Args:
        env_name: Name of the Gymnasium environment to train on.
        hidden_sizes: List of hidden layer sizes for the policy network.
        lr: Learning rate for the policy optimizer (value network uses lr*10).
        epochs: Number of training epochs.
        batch_size: Number of timesteps to collect per epoch before updating.
        gamma: Discount factor for future rewards.
        seed: Random seed for reproducibility. If None, uses random behavior.
    """
    # Set seeds for reproducibility if provided.
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    env = gym.make(env_name)

    if seed is not None:
        env.reset(seed=seed)
    else:
        env.reset()

    observations_dim = env.observation_space.shape[0]
    number_actions = env.action_space.n

    # Instantiate the policy function (which is a neural network)
    policy_model = mlp(sizes=[observations_dim] + hidden_sizes + [number_actions])
    value_model = mlp(
        sizes=[observations_dim] + hidden_sizes + [1]
    )  # We output a scalar which is the state value estimate.

    policy_optimizer = Adam(policy_model.parameters(), lr=lr)
    value_optimizer = Adam(value_model.parameters(), lr=lr)

    for epoch in range(epochs):
        start_time = time.time()
        batch_loss, value_loss, batch_returns, batch_lengths = train_epoch(
            env,
            policy_model,
            policy_optimizer,
            value_model,
            value_optimizer,
            batch_size,
            gamma,
        )

        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {batch_loss.item():.3f}, Value Loss: {value_loss.item():.3f}, "
            f"Return: {np.mean(batch_returns):.3f}±{np.std(batch_returns):.3f}, "
            f"Length: {np.mean(batch_lengths):.3f}±{np.std(batch_lengths):.3f}, "
            f"Time: {epoch_time:.3f}s"
        )

    env.close()


if __name__ == "__main__":
    train()
