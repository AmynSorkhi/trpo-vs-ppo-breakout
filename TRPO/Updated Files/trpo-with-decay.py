import gymnasium as gym
import torch
import torch.nn.functional as F
import ale_py
from torch.optim import Adam
import numpy as np
import time
from policy_network import PolicyNetwork
from init_blackout import BreakoutPreprocessor
import csv
import random
import string
import os
import optuna
from optuna.samplers import GridSampler
import shutil

gym.register_envs(ale_py)  # unnecessary but helpful for IDEs

# Define grid search space
search_space = {
    "gamma": [0.98, 0.99],
    "lam": [0.95, 0.97, 0.99],
    "delta": [0.01, 0.1, 0.5 ,1],
    "cg_iters": [5, 15, 20],
    "cg_damping": [0.05, 0.1 ],
    "epochs": [50, 100, 150],
    "steps_per_epoch": [10, 20, 50, 100, 500],
    "entropy_coeff": [0.1,.01,.05,0.2, 0.5],
    "reward_threshold": [3.0],
    "entropy_decay_rate": [0.99, 0.95, 0.9]
}
sampler = GridSampler(search_space)

# Global folder for optuna run results (set in main if optuna mode is active)
RESULTS_DIR = "."

class TRPO:
    def __init__(self, env, gamma=0.995, lam=0.97, delta=0.2, cg_iters=10, cg_damping=0.1,
                 entropy_decay_rate=0.99, min_entropy_coeff=0.001, use_entropy_penalty=False):
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

        self.entropy_coeff = 0.05  # initial entropy coefficient; adjustable via CLI or Optuna
        self.entropy_decay_rate = entropy_decay_rate
        self.min_entropy_coeff = min_entropy_coeff
        self.use_entropy_penalty = use_entropy_penalty

        self.best_mean_reward = -np.inf

    def train(self, epochs=1000, steps_per_epoch=500, reward_threshold=2.0, log_folder="."):
        print(f"Starting training {self.env.unwrapped.spec.id} with TRPO:")
        print(f"Device: {self.device} | Gamma: {self.gamma} | Lambda: {self.lam} | Delta: {self.delta}")
        print(f"CG iterations: {self.cg_iters} | CG damping: {self.cg_damping}")
        print(f"Epochs: {epochs} | Steps per epoch: {steps_per_epoch}")
        print(f"Reward threshold for convergence: {reward_threshold}\n")

        best_checkpoint_path = os.path.join(log_folder, "best_checkpoint.pth")
        mean_total_rewards = []
        mean_total_rollouts_len = []
        total_steps = 0
        convergence_epoch = None
        start_time = time.time()

        zero_count = 0  # count consecutive epochs with zero mean reward

        def random_string(length=4):
            return ''.join(random.choices(string.ascii_uppercase, k=length))

        log_filename = os.path.join(log_folder, "training_log" + time.strftime("_%Y%m%d_%H%M%S_") + random_string() + ".csv")
        csvfile = open(log_filename, "w", newline="")
        csvwriter = csv.writer(csvfile)
        # Add header (including Epoch Sample Efficiency)
        csvwriter.writerow([
            "Epoch", "Mean Reward", "Std Reward", "Mean Rollout Length", "Total Steps",
            "Sample Efficiency (Cumulative)", "Epoch Sample Efficiency", "KL", "Entropy", "Epoch Time", "EarlyStop",
            f"gamma={self.gamma}", f"lam={self.lam}", f"delta={self.delta}",
            f"cg_iters={self.cg_iters}", f"cg_damping={self.cg_damping}",
            f"entropy_coeff={self.entropy_coeff}"
        ])
        csvfile.flush()

        for epoch in range(epochs):
            epoch_start_time = time.time()
            rollouts = []
            rollout_rewards = []
            rollout_lengths = []

            for _ in range(steps_per_epoch):
                state, _ = self.env.reset()
                state, reward, done, truncated, _ = self.env.step(1)  # FIRE action
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
                next_states = torch.cat(next_states)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
                scale_factor = 2.0  # Reward scaling factor; adjust as needed
                rewards = rewards * scale_factor
                returns = []
                G = 0.0
                for r in reversed(rewards):
                    G = r + self.gamma * G
                    returns.insert(0, G)
                returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
                rollouts.append((states, actions, returns, next_states))
                episode_reward = float(rewards.sum().cpu())
                rollout_rewards.append(episode_reward)
                rollout_lengths.append(len(rewards))

            mean_reward = np.mean(rollout_rewards)
            std_reward = np.std(rollout_rewards)
            mean_length = np.mean(rollout_lengths)
            # Cumulative sample efficiency: mean reward divided by total steps so far.
            sample_efficiency = mean_reward / total_steps
            # Epoch sample efficiency: mean reward divided by steps taken in this epoch.
            epoch_steps = sum(rollout_lengths)
            epoch_sample_efficiency = mean_reward / epoch_steps if epoch_steps != 0 else 0

            mean_total_rewards.append(mean_reward)
            mean_total_rollouts_len.append(mean_length)

            # Check for early stopping: if mean_reward is zero for 3 consecutive epochs.
            if mean_reward == 0:
                zero_count += 1
            else:
                zero_count = 0
            early_stop_note = ""
            if zero_count >= 3:
                early_stop_note = f"Early stop at epoch {epoch}"
                epoch_duration = time.time() - epoch_start_time
                log_line = [epoch, mean_reward, std_reward, mean_length, total_steps,
                           sample_efficiency, epoch_sample_efficiency, 0, 0, epoch_duration, early_stop_note]
                csvwriter.writerow(log_line)
                csvfile.flush()
                break

            if convergence_epoch is None and mean_reward >= reward_threshold:
                convergence_epoch = epoch

            epoch_duration = time.time() - epoch_start_time
            log_line = [epoch, mean_reward, std_reward, mean_length, total_steps,
                       sample_efficiency, epoch_sample_efficiency, 0, 0, epoch_duration, early_stop_note]
            print(f"Epoch {epoch:03d} - Mean Reward: {mean_reward:.2f} ± {std_reward:.2f} | "
                  f"Mean Rollout Length: {mean_length:.2f} | Total Steps: {total_steps} | "
                  f"Cumulative Sample Efficiency: {sample_efficiency:.6f} | "
                  f"Epoch Sample Efficiency: {epoch_sample_efficiency:.6f} | "
                  f"Epoch Time: {epoch_duration:.2f}s")
            csvwriter.writerow(log_line)
            csvfile.flush()

            # Update policy and get KL & entropy metrics
            kl_metric, entropy_metric = self.update_policy_from_rollouts(rollouts)
            with open(log_filename, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Final KL", kl_metric, "Final Entropy", entropy_metric])

            if mean_reward > self.best_mean_reward:
                if os.path.exists(best_checkpoint_path):
                    os.remove(best_checkpoint_path)
                torch.save(self.policy.state_dict(), best_checkpoint_path)
                self.best_mean_reward = mean_reward
                print(f"Saved new best checkpoint with reward {mean_reward:.2f}")
                print("Checkpoint hyperparameters:")
                print(f"gamma: {self.gamma}")
                print(f"lam: {self.lam}")
                print(f"delta: {self.delta}")
                print(f"cg_iters: {self.cg_iters}")
                print(f"cg_damping: {self.cg_damping}")
                print(f"entropy_coeff: {self.entropy_coeff}")

            last_checkpoint_path = os.path.join(log_folder, "last_checkpoint.pth")
            if os.path.exists(last_checkpoint_path):
                os.remove(last_checkpoint_path)
            torch.save(self.policy.state_dict(), last_checkpoint_path)

            # Update entropy coefficient by decay schedule
            self.entropy_coeff = max(self.min_entropy_coeff, self.entropy_coeff * self.entropy_decay_rate)
            # print(f"Updated entropy_coeff: {self.entropy_coeff:.6f}")

        total_time = time.time() - start_time
        print("\nTraining Complete!")
        print(f"Total Epochs: {epoch+1} | Total Steps: {total_steps} | Total Training Time: {total_time:.2f}s")
        if convergence_epoch is not None:
            print(f"Convergence threshold reached at epoch: {convergence_epoch}")
        else:
            print("Convergence threshold not reached.")
        return mean_total_rewards, mean_total_rollouts_len, log_filename

    def get_action(self, x):
        with torch.no_grad():
            dist = self.policy.actor_forward(x)
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
                    next_value = values[t+1] if t < T - 1 else self.policy.critic_forward(states[-1].unsqueeze(0)).item()
                    delta = rewards[t] + self.gamma * next_value - values[t]
                    deltas.append(delta)
                adv = []
                advantage = 0.0
                for delta in reversed(deltas):
                    advantage = delta + self.gamma * self.lam * advantage
                    adv.insert(0, advantage)
                advantages_list.append(torch.tensor(adv, dtype=torch.float32, device=self.device))
        advantages = torch.cat(advantages_list)
        print("Raw advantages; mean:", advantages.mean().item(), "std:", advantages.std().item())
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def update_value_network(self, states, returns):
        returns = returns.detach()
        values = self.policy.critic_forward(states).squeeze()
        loss = F.mse_loss(values, returns)
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()
        # print("Value network updated; loss:", loss.item())

    def update_policy(self, states, actions, advantages):
        with torch.no_grad():
            actor_features = self.policy.conv(states.permute(0, 3, 1, 2))
            logits = self.policy.actor(actor_features)
            logits = torch.clamp(logits, -10, 10)
            # print("Sample logits:", logits[0].cpu().numpy())
            old_dist = self.policy.actor_forward(states)
            old_log_probs = old_dist.log_prob(actions)
        # print("Old distribution computed. Sample log_prob:", old_log_probs[0].item())
        current_dist = self.policy.actor_forward(states)
        log_probs = current_dist.log_prob(actions)
        surrogate = self.surrogate_loss(log_probs, old_log_probs, advantages)
        # print("Surrogate loss (pre-update):", surrogate.item())
        dist_entropy = current_dist.entropy().mean()
        if self.use_entropy_penalty:
            final_loss = -(surrogate - self.entropy_coeff * dist_entropy)
        else:
            final_loss = -(surrogate + self.entropy_coeff * dist_entropy)
        print("Entropy (pre-update):", dist_entropy.item(), "Final loss:", final_loss.item())
        self.policy.actor.zero_grad()
        final_loss.backward(retain_graph=True)
        g = torch.cat([p.grad.flatten().detach() for p in self.policy.actor.parameters()])
        # print("Gradient norm:", torch.norm(g).item())

        def hvp(v):
            self.policy.actor.zero_grad()
            current_dist_hvp = self.policy.actor_forward(states)
            kl = torch.distributions.kl.kl_divergence(old_dist, current_dist_hvp).mean()
            flat_grad_kl = self.flat_grad(kl, self.policy.actor.parameters(), create_graph=True)
            kl_v = (flat_grad_kl * v).sum()
            flat_grad_kl_v = self.flat_grad(kl_v, self.policy.actor.parameters(), create_graph=False, retain_graph=True)
            hvp_val = flat_grad_kl_v + self.cg_damping * v
            return hvp_val

        step_dir = self.conjugate_gradient(hvp, g)
        print("Step direction norm:", torch.norm(step_dir).item())
        shs = 0.5 * (step_dir @ hvp(step_dir))
        lagrange_multiplier = torch.sqrt(shs / self.delta + 1e-8)
        full_step = step_dir / (lagrange_multiplier + 1e-8)
        # print("Lagrange multiplier:", lagrange_multiplier.item(), "Full step norm:", torch.norm(full_step).item())
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
                tol = 1e-3
                if kl <= self.delta and new_objective >= old_objective - tol:
                    # print("tol true")
                    return True
            return False

        if not line_search():
            idx = 0
            for p in self.policy.actor.parameters():
                p_len = p.numel()
                p.data.copy_(old_actor_params[idx:idx + p_len].view(p.shape))
                idx += p_len
            # print("Line search failed. Reverting to old parameters.")

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
            if rs_new < 1e-10:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Train TRPO on Breakout 5 environments.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--lam", type=float, default=0.97, help="Lambda for GAE.")
    parser.add_argument("--delta", type=float, default=0.2, help="TRPO KL divergence threshold.")
    parser.add_argument("--cg_iters", type=int, default=10, help="Number of CG iterations.")
    parser.add_argument("--cg_damping", type=float, default=0.1, help="CG damping factor.")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of training epochs.")
    parser.add_argument("--steps_per_epoch", type=int, default=10, help="Number of steps per epoch.")
    parser.add_argument("--entropy_coeff", type=float, default=0.03, help="Initial coefficient for entropy regularization.")
    parser.add_argument("--reward_threshold", type=float, default=2.0, help="Reward threshold for convergence.")
    parser.add_argument("--entropy_decay_rate", type=float, default=0.99, help="Decay rate for entropy coefficient per epoch.")
    parser.add_argument("--min_entropy_coeff", type=float, default=0.001, help="Minimum entropy coefficient.")
    parser.add_argument("--use_entropy_penalty", action="store_true", help="Use entropy penalty (subtract entropy term) instead of bonus.")
    parser.add_argument("--optuna", action="store_true", help="Run hyperparameter optimization using Optuna.")
    return parser.parse_args()

def objective(trial):
    gamma = trial.suggest_categorical("gamma", [0.97, 0.98, 0.99])
    lam = trial.suggest_categorical("lam", [0.95, 0.96, 0.97, 0.98, 0.99])
    delta = trial.suggest_categorical("delta", [0.1, 0.15, 0.2, 0.25, 0.3])
    cg_iters = trial.suggest_categorical("cg_iters", [5, 10, 12, 15, 20])
    cg_damping = trial.suggest_categorical("cg_damping", [0.05, 0.5, 0.1])
    epochs = trial.suggest_categorical("epochs", [50, 100])
    steps_per_epoch = trial.suggest_categorical("steps_per_epoch", [10, 20, 30, 40, 50])
    entropy_coeff = trial.suggest_categorical("entropy_coeff", [0.01, 0.02, 0.03, 0.05, 0.1])
    reward_threshold = trial.suggest_categorical("reward_threshold", [1.0, 1.5, 2.0, 2.5, 3.0])
    
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    env = gym.wrappers.TimeLimit(env, max_episode_steps=25000)
    env = BreakoutPreprocessor(env, skip=4, stack=4)
    trpo = TRPO(env, gamma=gamma, lam=lam, delta=delta, cg_iters=cg_iters, cg_damping=cg_damping,
                entropy_decay_rate=0.99, min_entropy_coeff=0.001, use_entropy_penalty=False)
    trpo.entropy_coeff = entropy_coeff
    rewards, _, log_filename = trpo.train(epochs=epochs, steps_per_epoch=steps_per_epoch, reward_threshold=reward_threshold, log_folder=RESULTS_DIR)
    final_reward = sum(rewards) / len(rewards)
    
    # Save results for this trial to CSV (in the results folder)
    result_file = os.path.join(RESULTS_DIR, "all_trials_results.csv")
    header = ["Trial", "gamma", "lam", "delta", "cg_iters", "cg_damping", "epochs", "steps_per_epoch",
              "entropy_coeff", "reward_threshold", "final_reward", "log_filename"]
    if not os.path.exists(result_file):
        with open(result_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
    with open(result_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([trial.number, gamma, lam, delta, cg_iters, cg_damping, epochs,
                         steps_per_epoch, entropy_coeff, reward_threshold, final_reward, log_filename])
    
    trial.set_user_attr("log_file", log_filename)
    return final_reward

def load_and_run(checkpoint_path, num_episodes=5, render=True):
    env = gym.make("ALE/Breakout-v5", render_mode='human' if render else 'rgb_array')
    env = gym.wrappers.TimeLimit(env, max_episode_steps=25000)
    env = BreakoutPreprocessor(env, skip=4, stack=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    policy = PolicyNetwork(obs_shape, n_actions).to(device)
    policy.load_state_dict(torch.load(checkpoint_path, map_location=device))
    policy.eval()
    for episode in range(num_episodes):
        state, _ = env.reset()
        state, _, _, _, _ = env.step(1)  # Fire to start
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        done = truncated = False
        total_reward = 0
        while not (done or truncated):
            with torch.no_grad():
                dist = policy.actor_forward(state)
                action = dist.sample().item()
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
            if render and record_dir is None:
                env.render()
            else:
                time.sleep(0.01)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    env.close()

if __name__ == "__main__":
    args = parse_args()
    params = vars(args)
    if args.optuna:
        RESULTS_DIR = time.strftime("%Y-%m-%d-%H%M%S")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=20)
        print("Best hyperparameters:")
        print(study.best_trial.params)
        best_output_file = os.path.join(RESULTS_DIR, "best_optuna_trial.csv")
        with open(best_output_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Hyperparameter", "Value"])
            for key, value in study.best_trial.params.items():
                writer.writerow([key, value])
            writer.writerow(["Best Reward", study.best_trial.value])
        best_log = study.best_trial.user_attrs.get("log_file", "N/A")
        if best_log != "N/A" and os.path.exists(best_log):
            shutil.copy(best_log, os.path.join(RESULTS_DIR, "best_trial_training_log.csv"))
            print("Best trial training log saved to best_trial_training_log.csv")
        print(f"Best trial stats saved to {best_output_file}")
    else:
        env = gym.make("ALE/Breakout-v5", render_mode='rgb_array')
        env = gym.wrappers.TimeLimit(env, max_episode_steps=25000)
        env = BreakoutPreprocessor(env, skip=4, stack=4)
        trpo = TRPO(env, gamma=params["gamma"], lam=params["lam"], delta=params["delta"],
                    cg_iters=params["cg_iters"], cg_damping=params["cg_damping"],
                    entropy_decay_rate=params["entropy_decay_rate"],
                    min_entropy_coeff=params["min_entropy_coeff"],
                    use_entropy_penalty=params["use_entropy_penalty"])
        trpo.entropy_coeff = params["entropy_coeff"]
        trpo.train(epochs=params["epochs"], steps_per_epoch=params["steps_per_epoch"],
                   reward_threshold=params["reward_threshold"])
