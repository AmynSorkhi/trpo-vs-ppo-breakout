import gymnasium as gym
import ale_py
import pickle
import time

env = gym.make("ALE/Breakout-v5", render_mode="human")
obs, info = env.reset()

print("Observation Space:", env.observation_space)
print("Action Space:", env.action_space.n)
print("Initial Observation Shape:", obs.shape)

for _ in range(1000):
    env.render()

    # Choose a random action(we can put a function using our method)
    action = env.action_space.sample() 
    # ==============================================================
     
    obs, reward, terminated, truncated, info = env.step(action)  # step func
    time.sleep(0.02)

    if terminated or truncated:
        obs, info = env.reset()

env.close()