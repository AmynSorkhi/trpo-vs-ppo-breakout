import gymnasium as gym
import torch
import torch.nn.functional as F
import ale_py
from torch.optim import Adam
import numpy as np
import time
from policy_network import PolicyNetwork
from init_blackout import BreakoutPreprocessor

class TRPO:
    def __init__(self, env, gamma=0.995, lam=0.97, delta=0.2, cg_iters=10, cg_damping=0.1):
        self.env = env
        obs_shape = env.observation_space.shape
        n_actions = env.action_space.n

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork(obs_shape, n_actions).to(self.device)
        self.value_optimizer = Adam(self.policy.critic.parameters(), lr=1e-4)

        self.gamma = gamma
        self.lam = lam
        self.delta = delta
        self.cg_iters = cg_iters
        self.cg_damping = cg_damping

        self.entropy_coeff = 0.03 #NOTE CHANGE TO 0.2, 0.3, 0.5

    def train(self, epochs=1000, steps_per_epoch=50, reward_threshold=2.0):
        print(f"Starting training {self.env.unwrapped.spec.id} with TRPO:")
        print(f"Device: {self.device} | Gamma: {self.gamma} | Lambda: {self.lam} | Delta: {self.delta}")
        print(f"CG iterations: {self.cg_iters} | CG damping: {self.cg_damping}")
        print(f"Epochs: {epochs} | Steps per epoch: {steps_per_epoch}")
        print(f"Reward threshold for convergence: {reward_threshold}\n")

        mean_total_rewards = []
        mean_total_rollouts_len = []
        total_steps = 0
        convergence_epoch = None
        start_time = time.time()

        for epoch in range(epochs):
            epoch_start_time = time.time()
            rollouts = []
            rollout_rewards = []
            rollout_lengths = []

            # Collect rollouts
            for _ in range(steps_per_epoch):
                state, _ = self.env.reset()
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                done = False
                truncated = False
                rollout = []

                while not (done or truncated):
                    action = self.get_action(state)
                    next_state, reward, done, truncated, info = self.env.step(action)
                    total_steps += 1
                    next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                    rollout.append((state, action, reward, next_state))
                    state = next_state

                states, actions, rewards, next_states = zip(*rollout)
                states = torch.cat(states)
                actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
                next_states = torch.cat(next_states)

                rollouts.append((states, actions, rewards, next_states))
                episode_reward = float(rewards.sum().cpu())
                rollout_rewards.append(episode_reward)
                rollout_lengths.append(len(rewards))

            print(f"Collected {len(rollouts)} rollouts; Mean reward: {np.mean(rollout_rewards):.2f}")

            # Update policy and value; update_policy_from_rollouts returns KL and entropy
            kl_metric, entropy_metric = self.update_policy_from_rollouts(rollouts)

            mean_reward = np.mean(rollout_rewards)
            std_reward = np.std(rollout_rewards)
            mean_length = np.mean(rollout_lengths)
            sample_efficiency = mean_reward / total_steps

            mean_total_rewards.append(mean_reward)
            mean_total_rollouts_len.append(mean_length)

            if convergence_epoch is None and mean_reward >= reward_threshold:
                convergence_epoch = epoch

            epoch_duration = time.time() - epoch_start_time
            print(f"Epoch {epoch:03d} - Mean Reward: {mean_reward:.2f} ± {std_reward:.2f} | "
                  f"Mean Rollout Length: {mean_length:.2f} | Total Steps: {total_steps} | "
                  f"Sample Efficiency: {sample_efficiency:.6f} | KL: {kl_metric:.6f} | "
                  f"Entropy: {entropy_metric:.6f} | Epoch Time: {epoch_duration:.2f}s")

        total_time = time.time() - start_time
        print("\nTraining Complete!")
        print(f"Total Epochs: {epochs} | Total Steps: {total_steps} | Total Training Time: {total_time:.2f}s")
        if convergence_epoch is not None:
            print(f"Convergence threshold reached at epoch: {convergence_epoch}")
        else:
            print("Convergence threshold not reached.")
        return mean_total_rewards, mean_total_rollouts_len

    def get_action(self, x):
        with torch.no_grad():
            dist = self.policy.actor_forward(x)
            # Uncomment the next line to log action probabilities for debugging.
            # print("Action probabilities:", dist.probs)
            return dist.sample().item()

    def update_policy_from_rollouts(self, rollouts):
        advantages = self.compute_advantages(rollouts)
        states, actions, rewards, next_states = zip(*rollouts)
        states = torch.cat(states)
        actions = torch.cat(actions)
        self.update_value_network(states, torch.cat(rewards))
        return self.update_policy(states, actions, advantages)

    def compute_advantages(self, rollouts):
        advantages_list = []
        for i, (states, _, rewards, _) in enumerate(rollouts):
            with torch.no_grad():
                values = self.policy.critic_forward(states).squeeze()
                T = len(rewards)
                deltas = []
                for t in range(T):
                    next_value = values[t+1] if t < T - 1 else 0.0
                    delta = rewards[t] + self.gamma * next_value - values[t]
                    deltas.append(delta)
                adv = []
                advantage = 0.0
                for delta in reversed(deltas):
                    advantage = delta + self.gamma * self.lam * advantage
                    adv.insert(0, advantage)
                advantages_list.append(torch.tensor(adv, dtype=torch.float32, device=self.device))
            # if i == 0:
                # Print critic values and rewards for the first rollout.
                # print("Critic values (first rollout):", values.cpu().numpy())
                # print("Rewards (first rollout):", rewards.cpu().numpy())
        # Use raw advantages (without centering/normalizing) and scale them
        advantages = torch.cat(advantages_list)
        print("Raw advantages; mean:", advantages.mean().item(), "std:", advantages.std().item())

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages *= 5
        # scaling_factor = 10.0
        # advantages = advantages * scaling_factor
        # print("Scaled advantages; mean:", advantages.mean().item(), "std:", advantages.std().item())
        return advantages

    def update_value_network(self, states, returns):
        returns = returns.detach()
        values = self.policy.critic_forward(states).squeeze()
        loss = F.mse_loss(values, returns)
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()
        print("Value network updated; loss:", loss.item())

    def update_policy(self, states, actions, advantages):
        with torch.no_grad():
            #BELOW ADDED NOTE
            actor_features = self.policy.conv(states.permute(0, 3, 1, 2))
            logits = self.policy.actor(actor_features)
            print("Sample logits:", logits[0].cpu().numpy())


            old_dist = self.policy.actor_forward(states)
            old_log_probs = old_dist.log_prob(actions)
        print("Old distribution computed. Sample log_prob:", old_log_probs[0].item())

        current_dist = self.policy.actor_forward(states)
        log_probs = current_dist.log_prob(actions)
        surrogate = self.surrogate_loss(log_probs, old_log_probs, advantages)
        print("Surrogate loss (pre-update):", surrogate.item())

        dist_entropy = current_dist.entropy().mean()
        final_loss = -(surrogate - self.entropy_coeff * dist_entropy)
        print("Entropy (pre-update):", dist_entropy.item(), "Final loss:", final_loss.item())

        self.policy.actor.zero_grad()
        final_loss.backward(retain_graph=True)
        g = torch.cat([p.grad.flatten().detach() for p in self.policy.actor.parameters()])
        print("Gradient norm:", torch.norm(g).item())

        def hvp(v):
            self.policy.actor.zero_grad()
            current_dist_hvp = self.policy.actor_forward(states)
            kl = torch.distributions.kl.kl_divergence(old_dist, current_dist_hvp).mean()
            flat_grad_kl = self.flat_grad(kl, self.policy.actor.parameters(), create_graph=True)
            kl_v = (flat_grad_kl * v).sum()
            flat_grad_kl_v = self.flat_grad(kl_v, self.policy.actor.parameters(), create_graph=False, retain_graph=True)
            hvp_val = flat_grad_kl_v + self.cg_damping * v
            # print("HVP intermediate value (norm):", torch.norm(hvp_val).item())
            return hvp_val

        step_dir = self.conjugate_gradient(hvp, g)
        print("Step direction norm:", torch.norm(step_dir).item())
        shs = 0.5 * (step_dir @ hvp(step_dir))
        lagrange_multiplier = torch.sqrt(shs / self.delta + 1e-8)
        full_step = step_dir / (lagrange_multiplier + 1e-8)
        print("Lagrange multiplier:", lagrange_multiplier.item(), "Full step norm:", torch.norm(full_step).item())

        old_actor_params = torch.cat([p.data.view(-1) for p in self.policy.actor.parameters()])
        old_entropy = old_dist.entropy().mean().item()
        old_objective = surrogate.item() - self.entropy_coeff * old_entropy
        print("Old objective (including entropy):", old_objective)

        def line_search():
            for i, alpha in enumerate([0.5 ** i for i in range(20)]):
                new_actor_params = old_actor_params + alpha * full_step
                idx = 0
                for p in self.policy.actor.parameters():
                    p_len = p.numel()
                    p.data.copy_(new_actor_params[idx:idx + p_len].view(p.shape))
                    idx += p_len

                with torch.no_grad():
                    new_dist = self.policy.actor_forward(states)
                    new_log_probs = new_dist.log_prob(actions)
                    new_surrogate = self.surrogate_loss(new_log_probs, old_log_probs, advantages)
                    new_entropy = new_dist.entropy().mean().item()
                    new_objective = new_surrogate.item() - self.entropy_coeff * new_entropy
                    kl = torch.distributions.kl.kl_divergence(old_dist, new_dist).mean()
                    # Print norm difference of parameters (optional)
                    param_diff_norm = torch.norm(new_actor_params - old_actor_params).item()
                print(f"Line search step {i}: alpha={alpha:.4f}, new_obj={new_objective:.6f}, kl={kl.item():.6f}, param_diff_norm={param_diff_norm:.6f}")
                # Accept if KL is below threshold and the new objective is not significantly worse.
                tol = 5e-3  # or another small value NOTE try changing?
                if kl <= self.delta and new_objective >= old_objective - tol:
                    print("tol true")
                    return True

            return False

        if not line_search():
            idx = 0
            for p in self.policy.actor.parameters():
                p_len = p.numel()
                p.data.copy_(old_actor_params[idx:idx + p_len].view(p.shape))
                idx += p_len
            print("Line search failed. Reverting to old parameters.")

        with torch.no_grad():
            new_dist = self.policy.actor_forward(states)
            kl_metric = torch.distributions.kl.kl_divergence(old_dist, new_dist).mean().item()
            entropy_metric = new_dist.entropy().mean().item()
        print("Final KL:", kl_metric, "Final entropy:", entropy_metric)
        return kl_metric, entropy_metric

    def surrogate_loss(self, new_log_probs, old_log_probs, advantages):
        ratio = torch.exp(new_log_probs - old_log_probs)
        return torch.mean(ratio * advantages)

    def flat_grad(self, y, x, retain_graph=False, create_graph=False):
        if create_graph:
            retain_graph = True
        grads = torch.autograd.grad(y, x, create_graph=create_graph, retain_graph=retain_graph)
        return torch.cat([grad.view(-1) for grad in grads])

    def conjugate_gradient(self, A, b):
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rs_old = r.dot(r)
        for i in range(self.cg_iters):
            Ap = A(p)
            alpha = rs_old / (p.dot(Ap) + 1e-8)
            x += alpha * p
            r -= alpha * Ap
            rs_new = r.dot(r)
            # print(f"CG iteration {i}: residual norm = {rs_new.item():.6f}")
            if rs_new < 1e-10:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x

if __name__ == "__main__":
    env = gym.make("ALE/Breakout-v5", render_mode='rgb_array')
    env = gym.wrappers.TimeLimit(env, max_episode_steps=5000)
    env = BreakoutPreprocessor(env)
    trpo = TRPO(env)
    trpo.train()
