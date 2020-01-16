from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import gym
import gym.spaces
import numpy as np


class NormalizeActionSpace(gym.ActionWrapper):
    """Normalize a Box action space to [-1, 1]^n."""

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Box)
        self.action_space = gym.spaces.Box(
            low=-np.ones_like(env.action_space.low),
            high=np.ones_like(env.action_space.low),
        )

    def action(self, action):
        # action is in [-1, 1]
        action = action.copy()

        # -> [0, 2]
        action += 1

        # -> [0, orig_high - orig_low]
        action *= (self.env.action_space.high - self.env.action_space.low) / 2

        # -> [orig_low, orig_high]
        return action + self.env.action_space.low


class AbsorbingWrapper(gym.ObservationWrapper):
  """Wraps an environment to have an indicator dimension.
  The indicator dimension is used to represent absorbing states of MDP.
  If the last dimension is 0. It corresponds to a normal state of the MDP,
  1 corresponds to an absorbing state.
  The environment itself returns only normal states, absorbing states are added
  later.
  This wrapper is used mainly for GAIL, since we need to have explicit
  absorbing states in order to be able to assign rewards.
  """

  def __init__(self, env):
    super(AbsorbingWrapper, self).__init__(env)
    obs_space = self.observation_space
    self.observation_space = gym.spaces.Box(
        shape=(obs_space.shape[0] + 1,),
        low=obs_space.low[0],
        high=obs_space.high[0],
        dtype=obs_space.dtype)

  def observation(self, observation):
    return self.get_non_absorbing_state(observation)

  def get_non_absorbing_state(self, obs):
    """Converts an original state of the environment into a non-absorbing state.
    Args:
      obs: a numpy array that corresponds to a state of unwrapped environment.
    Returns:
      A numpy array corresponding to a non-absorbing state obtained from input.
    """
    return np.concatenate([obs, [0]], -1).astype(self.observation_space.dtype)

  def get_absorbing_state(self):
    """Returns an absorbing state that corresponds to the environment.
    Returns:
      A numpy array that corresponds to an absorbing state.
    """
    obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
    obs[-1] = 1
    return obs

  @property
  def _max_episode_steps(self):
    return self.env._max_episode_steps
