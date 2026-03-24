# PPO_VS_TRPO
Reinforcement Learning Algorithm Comparison

# 🧠 TRPO vs PPO: Reinforcement Learning Algorithm Comparison

This project implements and compares two foundational reinforcement learning algorithms — **Trust Region Policy Optimization (TRPO)** and **Proximal Policy Optimization (PPO)** — from scratch. Evaluations are conducted in the high-dimensional **Breakout-v5** environment using Gymnasium, with a focus on sample efficiency, stability, and computational overhead.

> 📍 Project by: Ali Asad, Adam Bayley, Amyn Sorkhilalehloo, Reza Jahantigh, Amirmohammad Ramezannaderi  
> 🏫 Queen’s University  
> 📫 Emails: `{24xt20, 19ahb, 24fl4, 22sy14, 24fbdg}@queensu.ca`

---

## 🔬 Objective

- Reproduce TRPO and PPO from scratch using PyTorch.
- Apply both algorithms to the **Atari Breakout-v5** environment.
- Compare their performance on:
  - ✅ Sample Efficiency
  - ✅ Training Stability
  - ✅ Policy Entropy and KL Divergence
  - ✅ Computational Cost

---

## 🕹 Environment

**Breakout-v5 (ALE)** from the Gymnasium library.  
Preprocessing includes:
- Frame skipping
- Grayscale conversion
- Resizing to 84×84
- Normalization
- Frame stacking (4 frames)

---

## 🧱 Architecture

- **Shared CNN Backbone** for both policy and value networks
- **Actor-Critic Design**
- **TRPO**: KL-divergence constrained second-order optimization
- **PPO**: Clipped surrogate loss, first-order method

---

## 📈 Results Summary

| Metric                    | PPO             | TRPO            |
|--------------------------|------------------|------------------|
| Highest Reward           | ~30             | ~8              |
| Avg. Reward              | ~25             | ~4.5            |
| Mean Episode Length      | 150             | 68              |
| Sample Efficiency        | ~0.01           | <0.0013         |
| Policy Entropy (final)   | ~0.56           | ~0.89           |
| Epochs to Convergence    | <1000           | >3000           |

**Conclusion**: PPO demonstrated superior performance in all categories due to its stability, implementation simplicity, and robust clipped objective.

---
