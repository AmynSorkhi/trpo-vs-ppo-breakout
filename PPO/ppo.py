import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from policy_network import PolicyNetwork
from init_blackout import BreakoutPreprocessor
from gymnasium.wrappers import RecordVideo
import csv

class PPO:
    def __init__(self, env_name="ALE/Breakout-v5", device='cpu'):
        self.device = device
        self.env_name = env_name
        self.setup_environment()
        self.initialize_network()
        self.setup_tracking_variables()
        
    def setup_environment(self):
        """Initialize the environment with preprocessing"""
        self.env = gym.make(self.env_name, render_mode='rgb_array')
        self.env = BreakoutPreprocessor(self.env, skip=4, stack=4)
        self.action_dim = self.env.action_space.n
        self.obs_space = (4, 84, 84)  # Channels first for PyTorch
        
    def initialize_network(self):
        """Initialize policy network and optimizer"""
        self.policy = PolicyNetwork(self.obs_space, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)
        
    def setup_tracking_variables(self):
        """Initialize variables for tracking training progress"""
        self.best_avg_reward = -np.inf
        self.checkpoint_dir = "./checkpoints_ppo"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.reward_history = []
        self.ep_length_history = []
        self.entropy_history = []
        self.time_per_iteration = []
        
        self.convergence_threshold = 6  
        self.convergence_frame = None
        self.total_env_frames = 0
    
    def collect_experience(self, num_steps=512):
        """
        Collect experience from the environment.
        After resetting, press FIRE (action index 1) to serve the ball.
        """
        states, actions, rewards, dones, log_probs = [], [], [], [], []

        # Reset environment (already preprocessed by wrapper)
        state, _ = self.env.reset()
        # Convert from (84, 84, 4) to (4, 84, 84)
        state = np.moveaxis(state, -1, 0)
        
        # Serve the ball: FIRE action (assumed action index 1)
        state, reward, done, truncated, _ = self.env.step(1)
        state = np.moveaxis(state, -1, 0)  # Convert to channels first

        for _ in range(num_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            state_tensor = state_tensor.to(self.device)
            with torch.no_grad():
                action_dist, _ = self.policy(state_tensor)
                action = action_dist.sample()
                
            next_state, reward, done, truncated, info = self.env.step(action.item())
            # Convert from (84, 84, 4) to (4, 84, 84)
            next_state = np.moveaxis(next_state, -1, 0)

            states.append(state.copy())
            actions.append(action.item())
            rewards.append(reward)
            dones.append(done)
            log_probs.append(action_dist.log_prob(action).item())

            state = next_state  # Update current state

            if done:
                # If episode ends, reset and serve the ball again
                state, _ = self.env.reset()
                state = np.moveaxis(state, -1, 0)
                state, reward, done, truncated, _ = self.env.step(1)
                state = np.moveaxis(state, -1, 0)

        # Convert lists to torch tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).to(self.device)
        log_probs = torch.tensor(np.array(log_probs), dtype=torch.float32).to(self.device)

        return states, actions, rewards, dones, log_probs
    
    @staticmethod
    def compute_advantages(rewards, values, gamma=0.94, lam=0.97):
        """Compute advantages using Generalized Advantage Estimation (GAE)"""
        rewards_np = rewards.cpu().numpy()
        values_np = values.squeeze().detach().cpu().numpy()
        advantages = np.zeros_like(rewards_np)
        last_advantage = 0
        for t in reversed(range(len(rewards_np))):
            if t == len(rewards_np) - 1:
                delta = rewards_np[t] - values_np[t]
            else:
                delta = rewards_np[t] + gamma * values_np[t+1] - values_np[t]
            last_advantage = delta + gamma * lam * last_advantage
            advantages[t] = last_advantage
        return torch.tensor(advantages, dtype=torch.float32).to(rewards.device)
    
    def compute_ppo_loss(self, old_log_probs, states, actions, advantages, returns, epsilon=0.3, c1=0.2, c2=0.01):
        """Compute PPO loss with value loss and entropy bonus"""
        action_dist, new_values = self.policy(states)
        new_log_probs = action_dist.log_prob(actions)

        ratio = torch.exp(new_log_probs - old_log_probs)
        policy_loss = -torch.min(ratio * advantages, torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages).mean()
        value_loss = nn.MSELoss()(new_values.squeeze(), returns)
        entropy_bonus = action_dist.entropy().mean()

        return policy_loss + c1 * value_loss - c2 * entropy_bonus
    
    def update_policy(self, states, actions, old_log_probs, advantages, returns, num_epochs=10, batch_size=512):
        """Perform multiple epochs of policy updates"""
        entropy_vals = []
        num_steps = states.shape[0]
        
        for epoch in range(num_epochs):
            indices = torch.randperm(num_steps)
            for start in range(0, num_steps, batch_size):
                mb_idx = indices[start:start+batch_size]
                action_dist, _ = self.policy(states[mb_idx])
                entropy_vals.append(action_dist.entropy().mean().item())

                loss = self.compute_ppo_loss(
                    old_log_probs[mb_idx],
                    states[mb_idx],
                    actions[mb_idx],
                    advantages[mb_idx],
                    returns[mb_idx]
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        return np.mean(entropy_vals)
    
    def compute_episode_stats(self, rewards, dones):
        """Calculate average reward and episode length"""
        ep_reward, ep_len = 0, 0
        ep_rewards, ep_lengths = [], []
        
        for r, done in zip(rewards, dones):
            ep_reward += r.item() 
            ep_len += 1
            if done:
                ep_rewards.append(ep_reward)
                ep_lengths.append(ep_len)
                ep_reward = 0
                ep_len = 0

        avg_reward = np.mean(ep_rewards) if ep_rewards else 0
        avg_ep_length = np.mean(ep_lengths) if ep_lengths else 0
        std_reward = np.std(ep_rewards) if ep_rewards else 0
        
        return avg_reward, avg_ep_length, std_reward
    
    def save_checkpoint(self, iteration, avg_reward):
        """Save model checkpoint if it's the best so far"""
        if avg_reward > self.best_avg_reward:
            self.best_avg_reward = avg_reward
            torch.save({
                'iteration': iteration,
                'policy_state_dict': self.policy.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_reward': self.best_avg_reward,
            }, f"{self.checkpoint_dir}/best_model3.pth")
    
    def plot_training_metrics(self):
        """Generate and save plots of training metrics"""
        plt.figure()
        plt.plot(self.reward_history)
        plt.title("Mean Episodic Reward")
        plt.xlabel("Iteration")
        plt.ylabel("Avg Reward")
        plt.savefig("reward_plot.png")

        plt.figure()
        plt.plot(self.ep_length_history)
        plt.title("Average Episode Length")
        plt.xlabel("Iteration")
        plt.ylabel("Avg Length")
        plt.savefig("episode_length_plot.png")

        plt.figure()
        plt.plot(self.entropy_history)
        plt.title("Policy Entropy")
        plt.xlabel("Iteration")
        plt.ylabel("Entropy")
        plt.savefig("entropy_plot.png")

        plt.figure()
        plt.plot(self.time_per_iteration)
        plt.title("Time per Iteration (s)")
        plt.xlabel("Iteration")
        plt.ylabel("Seconds")
        plt.savefig("time_plot.png")
    

    def train(self, num_iterations=2, num_steps=2048):
        """Main training loop"""
        # Create CSV file and write header
        with open('training_metrics.csv', 'w', newline='') as csvfile:
            fieldnames = ['iteration', 'avg_reward', 'avg_ep_length', 'entropy', 'time_per_iteration', 'std_reward']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

        for iteration in range(num_iterations):
            start_time = time.time()

            # Collect experience
            states, actions, rewards, dones, old_log_probs = self.collect_experience(num_steps)
            self.total_env_frames += num_steps

            _, old_values = self.policy(states)
            advantages = self.compute_advantages(rewards, old_values)
            returns = advantages + old_values.squeeze().detach()

            # Compute episode stats
            avg_reward, avg_ep_length, std_reward = self.compute_episode_stats(rewards, dones)
            self.reward_history.append(avg_reward)
            self.ep_length_history.append(avg_ep_length)

            # Check convergence
            if self.convergence_frame is None and avg_reward >= self.convergence_threshold:
                self.convergence_frame = self.total_env_frames

            # Policy updates
            entropy = self.update_policy(states, actions, old_log_probs, advantages, returns)
            self.entropy_history.append(entropy)
            iteration_time = time.time() - start_time
            self.time_per_iteration.append(iteration_time)

            print(f"Iteration {iteration} | Avg Reward: {avg_reward:.2f} | std Reward: {std_reward:.2f} | " 
                f"Avg Ep Len: {avg_ep_length:.2f} | Entropy: {entropy:.4f} | "
                f"Time: {iteration_time:.2f}s")

            # Save metrics to CSV
            with open('training_metrics.csv', 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({
                    'iteration': iteration,
                    'avg_reward': avg_reward,
                    'avg_ep_length': avg_ep_length,
                    'entropy': entropy,
                    'time_per_iteration': iteration_time,
                    'std_reward': std_reward
                })

            self.save_checkpoint(iteration, avg_reward)
    
    # Save final model and plots
        torch.save(self.policy.state_dict(), "ppo_breakout3.pth")
        self.plot_training_metrics()
        self.env.close()

    def test(self, model_path="checkpoints_ppo/best_model.pth", num_episodes=3):
        """Test the trained model with rendering"""

        video_folder = "Video"
        os.makedirs(video_folder, exist_ok=True)

        env = gym.make(self.env_name, render_mode="rgb_array")
        env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda ep_id: True)
        env = BreakoutPreprocessor(env, skip=4, stack=4)
        
        # Load the saved model (handling both checkpoint and raw model files)
        policy = PolicyNetwork(self.obs_space, self.action_dim).to(self.device)
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Check if it's a checkpoint dictionary or raw model
            if 'policy_state_dict' in checkpoint:
                policy.load_state_dict(checkpoint['policy_state_dict'])
                print(f"Loaded checkpoint from iteration {checkpoint.get('iteration', 'unknown')}")
            else:
                # Assume it's a raw model file
                policy.load_state_dict(checkpoint)
                
        except Exception as e:
            print(f"Error loading model: {e}")
            env.close()
            return

        policy.eval()
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            state = np.moveaxis(state, -1, 0)  # Convert to channels first
            episode_reward = 0
            done = False
            lives = 5  # Breakout starts with 5 lives
            
            # Initial FIRE action to start the game
            state, _, _, _, _ = env.step(1)
            state = np.moveaxis(state, -1, 0)
            
            while not done:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                    action_dist, _ = policy(state_tensor)
                    action = torch.argmax(action_dist.probs).item()  # Greedy action
                
                next_state, reward, done, _, info = env.step(action)
                next_state = np.moveaxis(next_state, -1, 0)
                
                # Track lives to properly handle game over
                if 'lives' in info and info['lives'] < lives:
                    lives = info['lives']
                    env.step(1)  
                    time.sleep(0.5) 
                state = next_state
                episode_reward += reward
                
                # Render at human-viewable speed
                time.sleep(0.02)
                
            print(f"Episode {episode + 1}: Reward = {episode_reward:.1f}")
            time.sleep(1)  # Pause between episodes
        
        env.close()


def main(mode='train'):
    ppo = PPO()
    if mode == 'train':
        ppo.train(num_iterations=5000, num_steps=2048)
    elif mode == 'test':
        ppo.test(model_path="PPO/Analyse_PPO/result_ppo3/best_model3.pth", num_episodes=5)

if __name__ == '__main__':
    main('test')