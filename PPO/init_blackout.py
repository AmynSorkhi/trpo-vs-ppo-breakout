# init_blackout.py

import gymnasium as gym
import cv2
import numpy as np
from collections import deque

class BreakoutPreprocessor(gym.Wrapper):
    """
    A combined wrapper that:
      - Skips frames (repeat action for 'skip' frames)
      - Converts each resulting frame to grayscale,
      - Resizes to 84x84,
      - Normalizes pixel values [0..1],
      - Stacks the most recent 'stack' frames along the channel dimension.
    """

    def __init__(self, env, skip=4, stack=4):
        super().__init__(env)
        self.skip = skip      # how many frames to repeat for one agent step
        self.stack = stack    # how many frames to stack
        self.frames = deque([], maxlen=stack)

        # Adjust observation space to (84, 84, stack)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(84, 84, self.stack),
            dtype=np.float32
        )

    def _process_frame(self, frame):
        """Grayscale and downscale frame to 84x84, returning float32 [0..1]."""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        processed = self._process_frame(obs)
        self.frames.clear()
        # Push the same processed frame 'stack' times initially
        for _ in range(self.stack):
            self.frames.append(processed)
        # Return stacked frames
        return np.stack(self.frames, axis=-1), info

    def step(self, action):
        """
        Repeat the chosen 'action' for 'skip' frames, accumulating reward.
        Then push the new processed frame into the stack.
        """
        total_reward = 0.0
        done = False
        truncated = False
        info = None

        # Repeat action for 'skip' frames
        for _ in range(self.skip):
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if done or truncated:
                break

        # Process the last observed frame
        processed = self._process_frame(obs)
        self.frames.append(processed)

        # Build final stacked state
        stacked_obs = np.stack(self.frames, axis=-1)
        return stacked_obs, total_reward, done, truncated, info
