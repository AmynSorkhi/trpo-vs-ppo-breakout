import gymnasium as gym
import ale_py
import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np

from policy_network import PolicyNetwork


class TRPO:
    """Trust Region Policy Optimization (TRPO) algorithm implementation.

    Attributes:
        env: The Gymnasium environment
        policy: Policy network with actor and critic
        value_optimizer: Optimizer for the value function
        gamma: Discount factor
        lam: GAE lambda parameter
        delta: KL divergence constraint
        cg_iters: Conjugate gradient iterations
        cg_damping: Conjugate gradient damping coefficient
        device: Computation device (GPU or CPU)
    """

    def __init__(self, env, gamma=0.99, lam=0.95, delta=0.01, cg_iters=10, cg_damping=0.1):
        """Initialize TRPO agent.

        Args:
            env: Gymnasium environment
            gamma: Discount factor for rewards
            lam: GAE lambda parameter
            delta: Maximum KL divergence for policy updates
            cg_iters: Number of conjugate gradient iterations
            cg_damping: Damping coefficient for conjugate gradient
        """
        self.env = env
        obs_shape = env.observation_space.shape
        n_actions = env.action_space.n

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork(obs_shape, n_actions).to(self.device)
        self.value_optimizer = Adam(self.policy.critic.parameters(), lr=1e-3)

        self.gamma = gamma
        self.lam = lam
        self.delta = delta
        self.cg_iters = cg_iters
        self.cg_damping = cg_damping

    def train(self, epochs=1000, steps_per_epoch=10):
        """Train the TRPO agent.

        Args:
            epochs: Number of training epochs
            steps_per_epoch: Number of rollouts per epoch

        Returns:
            List of mean total rewards per epoch
        """

        print(f"Starting training {self.env.unwrapped.spec.id} with TRPO with the following parameters:")
        print(f"Device in use: {self.device}")
        print(f"Gamma: {self.gamma}")
        print(f"Lambda: {self.lam}")
        print(f"Delta: {self.delta}")
        print(f"CG iterations: {self.cg_iters}")
        print(f"CG damping: {self.cg_damping}")
        print(f"EPOCHS: {epochs}")
        print(f"STEPS PER EPOCH: {steps_per_epoch}")

        mean_total_rewards = []
        mean_total_rollouts_len = []
        for epoch in range(epochs):
            rollouts = []
            rollout_rewards = []
            rollout_len = []

            # Collect rollouts for this epoch
            for _ in range(steps_per_epoch):
                state, _ = self.env.reset()
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                done = False
                rollout = []

                # Run episode
                while not done:
                    action = self.get_action(state)
                    next_state, reward, done, truncated, info = self.env.step(action)
                    next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                    rollout.append((state, action, reward, next_state))
                    state = next_state

                # Convert rollout to tensors
                states, actions, rewards, next_states = zip(*rollout)
                states = torch.cat(states)
                actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
                next_states = torch.cat(next_states)

                rollouts.append((states, actions, rewards, next_states))
                rollout_rewards.append(sum(rewards.cpu().numpy()))
                rollout_len.append(len(rewards))

            # Update policy and value function
            self.step(rollouts)

            # Track and print progress
            mean_total_rewards.append(np.mean(rollout_rewards))
            mean_total_rollouts_len.append(np.mean(rollout_len))
            print(f"Epoch {epoch} - Mean total reward: {mean_total_rewards[-1]} - Mean rollout length: {mean_total_rollouts_len[-1]}")
        return mean_total_rewards, mean_total_rollouts_len

    def get_action(self, x):
        """Sample action from the current policy.

        Args:
            x: Current state

        Returns:
            Sampled action
        """
        with torch.no_grad():
            dist = self.policy.actor_forward(x)
            return dist.sample().item()

    def step(self, rollouts):
        """Perform one TRPO update step using collected rollouts.

        Args:
            rollouts: List of (states, actions, rewards, next_states) tuples
        """
        # Compute advantages using GAE
        advantages = self.compute_advantages(rollouts)

        # Unpack rollouts
        states, actions, rewards, next_states = zip(*rollouts)
        states = torch.cat(states)
        rewards = torch.cat(rewards)

        # Update value function (critic)
        self.update_value_network(states, rewards)

        # Update policy (actor) using TRPO
        actions = torch.cat(actions)
        self.update_policy(states, actions, advantages)

    def compute_advantages(self, rollouts):
        """Compute advantages using Generalized Advantage Estimation (GAE).

        Args:
            rollouts: List of rollout tuples

        Returns:
            Tensor of computed advantages
        """
        advantages = []
        for states, _, rewards, next_states in rollouts:
            with torch.no_grad():
                # Compute values and next value
                values = self.policy.critic_forward(states).squeeze()
                next_value = self.policy.critic_forward(next_states[-1].unsqueeze(0)).squeeze()

                # Compute returns and deltas
                returns = rewards + self.gamma * next_value
                deltas = returns - values

                # Compute GAE advantages
                advantages.append(self.gae(deltas, self.gamma, self.lam))

        return torch.cat(advantages)

    def gae(self, deltas, gamma, lam):
        """Calculate Generalized Advantage Estimation.

        Args:
            deltas: TD residuals
            gamma: Discount factor
            lam: GAE lambda parameter

        Returns:
            Computed advantages
        """
        advantages = []
        advantage = 0

        # Compute advantages in reverse order
        for delta in deltas.flip(0):
            advantage = delta + gamma * lam * advantage
            advantages.append(advantage)

        # Reverse back to original order
        return torch.stack(advantages).flip(0)

    def update_value_network(self, states, returns):
        """Update the value function using MSE loss.

        Args:
            states: Batch of states
            returns: Computed returns
        """
        returns = returns.detach()
        values = self.policy.critic_forward(states).squeeze()

        # Compute MSE loss
        loss = F.mse_loss(values, returns)

        # Perform gradient step
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()

    def update_policy(self, states, actions, advantages):
        """Update policy using TRPO algorithm.

        Args:
            states: Batch of states
            actions: Batch of actions
            advantages: Computed advantages
        """
        # Get old policy's log probabilities
        with torch.no_grad():
            old_dist = self.policy.actor_forward(states)
            old_log_probs = old_dist.log_prob(actions)

        # Compute current policy's loss
        current_dist = self.policy.actor_forward(states)
        log_probs = current_dist.log_prob(actions)
        loss = -self.surrogate_loss(log_probs, old_log_probs, advantages)

        # Compute policy gradient
        self.policy.actor.zero_grad()
        loss.backward(retain_graph=True)
        g = torch.cat([p.grad.flatten().detach() for p in self.policy.actor.parameters()])

        def hvp(v):
            """Hessian-vector product computation for conjugate gradient."""
            self.policy.actor.zero_grad()
            current_dist_hvp = self.policy.actor_forward(states)

            # Compute KL divergence
            kl = torch.distributions.kl.kl_divergence(old_dist, current_dist_hvp).mean()

            # Compute gradient of KL
            flat_grad_kl = self.flat_grad(kl, self.policy.actor.parameters(), create_graph=True)

            # Compute product with vector v
            kl_v = (flat_grad_kl * v).sum()

            # Compute gradient of the product
            flat_grad_kl_v = self.flat_grad(kl_v, self.policy.actor.parameters(), create_graph=False, retain_graph=True)
            return flat_grad_kl_v + self.cg_damping * v

        # Compute natural policy gradient direction using conjugate gradient
        step_dir = self.conjugate_gradient(hvp, g)

        # Compute step size with constraint
        shs = 0.5 * (step_dir @ hvp(step_dir))
        lagrange_multiplier = torch.sqrt(shs / self.delta + 1e-8)
        full_step = step_dir / (lagrange_multiplier + 1e-8)

        # Save old parameters for potential rollback
        old_actor_params = torch.cat([p.data.view(-1) for p in self.policy.actor.parameters()])

        def line_search():
            """Perform line search to find acceptable step size."""
            for alpha in [0.5 ** i for i in range(10)]:  # Try steps halving each time
                # Update parameters with current alpha
                new_actor_params = old_actor_params + alpha * full_step

                # Update policy parameters
                idx = 0
                for p in self.policy.actor.parameters():
                    p_len = p.numel()
                    p.data.copy_(new_actor_params[idx:idx + p_len].view(p.shape))
                    idx += p_len

                with torch.no_grad():
                    # Compute new policy's performance
                    new_dist = self.policy.actor_forward(states)
                    new_log_probs = new_dist.log_prob(actions)
                    new_loss = self.surrogate_loss(new_log_probs, old_log_probs, advantages)
                    kl = torch.distributions.kl.kl_divergence(old_dist, new_dist).mean()

                # Check KL constraint and performance improvement
                if kl <= self.delta and new_loss >= loss.item():
                    return True
            return False

        # If line search fails, revert to old parameters
        if not line_search():
            idx = 0
            for p in self.policy.actor.parameters():
                p_len = p.numel()
                p.data.copy_(old_actor_params[idx:idx + p_len].view(p.shape))
                idx += p_len

    def surrogate_loss(self, new_log_probs, old_log_probs, advantages):
        """Compute policy gradient surrogate loss.

        Args:
            new_log_probs: Log probabilities under current policy
            old_log_probs: Log probabilities under old policy
            advantages: Computed advantages

        Returns:
            Surrogate loss value
        """
        ratio = torch.exp(new_log_probs - old_log_probs)
        return torch.mean(ratio * advantages)

    def flat_grad(self, y, x, retain_graph=False, create_graph=False):
        """Compute and flatten gradient of y with respect to x.

        Args:
            y: Scalar output
            x: Parameters to compute gradient with respect to
            retain_graph: Whether to retain computation graph
            create_graph: Whether to create computation graph

        Returns:
            Flattened gradient tensor
        """
        if create_graph:
            retain_graph = True
        grads = torch.autograd.grad(y, x, create_graph=create_graph, retain_graph=retain_graph)
        return torch.cat([grad.view(-1) for grad in grads])

    def conjugate_gradient(self, A, b):
        """Solve Ax = b using conjugate gradient method.

        Args:
            A: Linear operator (function)
            b: Right-hand side vector

        Returns:
            Solution vector x
        """
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rs_old = r.dot(r)

        for _ in range(self.cg_iters):
            Ap = A(p)
            alpha = rs_old / (p.dot(Ap) + 1e-8)
            x += alpha * p
            r -= alpha * Ap
            rs_new = r.dot(r)
            if rs_new < 1e-10:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x


if __name__ == "__main__":
    env = gym.make("ALE/Breakout-v5", render_mode='rgb_array')
    trpo = TRPO(env)
    trpo.train()