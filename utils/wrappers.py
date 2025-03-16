# Import the necessary libraries

import numpy as np
import os
os.environ.setdefault('PATH', '')
from collections import deque
import sys
sys.path.append('D:/Anaconda3/envs/ambienteRL/Lib/site-packages') # Path to the gym library
import gym
from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)

# Wrapper to sample initial states by performing random no-op actions on reset.
class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by performing a random number of no-op actions on reset.
        The no-op action is assumed to be action 0."""
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """Perform a no-op action for a random number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)
    
# Wrapper to handle environments that require an initial 'FIRE' action.
class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that require an initial 'FIRE' action."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)
    
# Wrapper to make end-of-life equivalent to end-of-episode.
class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life equivalent to end-of-episode, but only reset on true game over."""
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted."""
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs
    
# Wrapper to skip frames and take the maximum of the last two observations.
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame."""
        gym.Wrapper.__init__(self, env)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

# Wrapper to clip rewards to {-1, 0, +1}.
class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} based on its sign."""
        return np.sign(reward)

# Wrapper to resize frames to 84x84 pixels and optionally convert to grayscale.
class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        num_colors = 1 if self._grayscale else 3
        new_space = gym.spaces.Box(low=0, high=255, shape=(self._height, self._width, num_colors), dtype=np.uint8)
        if self._key is None:
            self.observation_space = new_space
        else:
            self.observation_space.spaces[self._key] = new_space

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]
        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_AREA)
        if self._grayscale:
            frame = np.expand_dims(frame, -1)
        return frame if self._key is None else obs.copy().update({self._key: frame})

# Wrapper to stack the last `k` frames together.
class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        return np.concatenate(list(self.frames), axis=-1)

# Wrapper to scale pixel values to [0, 1].
class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0

# Utility class to optimize memory usage by storing shared frames once.
class LazyFrames(object):
    def __init__(self, frames):
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        return out.astype(dtype) if dtype is not None else out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]

# Function to apply a series of wrappers for preprocessing the environment.
def custom_wrapper(env):
    """
    Apply a series of wrappers to preprocess the environment for reinforcement learning tasks.
    This function applies the following transformations:
    
    1. MaxAndSkipEnv: Skips every 4 frames and takes the maximum of the last two frames.
    2. WarpFrame: Resizes the observation frames to 84x84 pixels and converts them to grayscale.
    3. FrameStack: Stacks the last 4 frames together to provide temporal context to the agent.
    4. ClipRewardEnv: Clips the rewards to {-1, 0, +1} to reduce the scale of rewards and stabilize training.
    
    Args:
        env (gym.Env): The original Gym environment to wrap.
    
    Returns:
        gym.Env: A wrapped environment with the specified preprocessing applied.
    """
    env = MaxAndSkipEnv(env, skip=4)
    env = WarpFrame(env)
    env = FrameStack(env, 4)
    env = ClipRewardEnv(env)
    return env