from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
from builtins import *  # NOQA
standard_library.install_aliases()  # NOQA

import collections
import os
import copy
from logging import getLogger

import chainer
from chainer import cuda
import chainer.functions as F
import numpy as np

from chainerrl.agent import AttributeSavingMixin
from chainerrl.agent import BatchAgent
from chainerrl.misc.batch_states import batch_states
from chainerrl.misc.copy_param import synchronize_parameters
from chainerrl.replay_buffer import ReplayUpdater
# from chainerrl.replay_buffer import batch_experiences
from replay_buffer import batch_experiences

import sac
from replay_buffer import AbsorbReplayBuffer

def _mean_or_nan(xs):
    """Return its mean a non-empty sequence, numpy.nan for a empty one."""
    return np.mean(xs) if xs else np.nan

class TemperatureHolder(chainer.Link):
    """Link that holds a temperature as a learnable value.
    Args:
        initial_log_temperature (float): Initial value of log(temperature).
    """

    def __init__(self, initial_log_temperature=0):
        super().__init__()
        with self.init_scope():
            self.log_temperature = chainer.Parameter(
                np.array(initial_log_temperature, dtype=np.float32))

    def __call__(self):
        """Return a temperature as a chainer.Variable."""
        return F.exp(self.log_temperature)


class SQIL(sac.SoftActorCritic):
    """Soft Actor-Critic (SAC).
    See https://arxiv.org/abs/1812.05905
    Args:
        policy (Policy): Policy.
        q_func1 (Link): First Q-function that takes state-action pairs as input
            and outputs predicted Q-values.
        q_func2 (Link): Second Q-function that takes state-action pairs as
            input and outputs predicted Q-values.
        policy_optimizer (Optimizer): Optimizer setup with the policy
        q_func1_optimizer (Optimizer): Optimizer setup with the first
            Q-function.
        q_func2_optimizer (Optimizer): Optimizer setup with the second
            Q-function.
        replay_buffer (ReplayBuffer): Replay buffer
        gamma (float): Discount factor
        gpu (int): GPU device id if not None nor negative.
        replay_start_size (int): if the replay buffer's size is less than
            replay_start_size, skip update
        minibatch_size (int): Minibatch size
        update_interval (int): Model update interval in step
        phi (callable): Feature extractor applied to observations
        soft_update_tau (float): Tau of soft target update.
        logger (Logger): Logger used
        batch_states (callable): method which makes a batch of observations.
            default is `chainerrl.misc.batch_states.batch_states`
        burnin_action_func (callable or None): If not None, this callable
            object is used to select actions before the model is updated
            one or more times during training.
        initial_temperature (float): Initial temperature value. If
            `entropy_target` is set to None, the temperature is fixed to it.
        entropy_target (float or None): If set to a float, the temperature is
            adjusted during training to match the policy's entropy to it.
        temperature_optimizer (Optimizer or None): Optimizer used to optimize
            the temperature. If set to None, Adam with default hyperparameters
            is used.
        act_deterministically (bool): If set to True, choose most probable
            actions in the act method instead of sampling from distributions.
    """

    saved_attributes = (
        'policy',
        'q_func1',
        'q_func2',
        'target_q_func1',
        'target_q_func2',
        'policy_optimizer',
        'q_func1_optimizer',
        'q_func2_optimizer',
        'temperature_holder',
        'temperature_optimizer',
    )

    def __init__(
            self,
            policy,
            q_func1,
            q_func2,
            reward_func,
            policy_optimizer,
            q_func1_optimizer,
            q_func2_optimizer,
            reward_func_optimizer,
            replay_buffer,
            replay_buffer_demo,
            gamma,
            is_discrete,
            gpu=None,
            replay_start_size=10000,
            minibatch_size=100,
            update_interval=1,
            phi=lambda x: x,
            soft_update_tau=5e-3,
            logger=getLogger(__name__),
            batch_states=batch_states,
            burnin_action_func=None,
            initial_temperature=1.,
            entropy_target=None,
            temperature_optimizer=None,
            act_deterministically=True,
            sample_interval=20,
            lamda=1.0,
            penalty_lamda=10.0,
    ):

        self.policy = policy
        self.q_func1 = q_func1
        self.q_func2 = q_func2
        self.reward_func = reward_func
        if gpu is not None and gpu >= 0:
            cuda.get_device_from_id(gpu).use()
            self.policy.to_gpu(device=gpu)
            self.q_func1.to_gpu(device=gpu)
            self.q_func2.to_gpu(device=gpu)
            if self.reward_func:
                self.reward_func.to_gpu(device=gpu)

        self.xp = self.policy.xp
        self.replay_buffer = replay_buffer
        self.replay_buffer_demo = replay_buffer_demo
        self.gamma = gamma
        self.gpu = gpu
        self.phi = phi
        self.soft_update_tau = soft_update_tau
        self.sample_interval = sample_interval
        self.minibatch_size = minibatch_size
        self.logger = logger
        self.policy_optimizer = policy_optimizer
        self.q_func1_optimizer = q_func1_optimizer
        self.q_func2_optimizer = q_func2_optimizer
        self.reward_func_optimizer = reward_func_optimizer
        self.replay_start_size = replay_start_size
        self.batch_states = batch_states
        self.burnin_action_func = burnin_action_func
        self.initial_temperature = initial_temperature
        self.entropy_target = entropy_target
        self.lamda = lamda
        if self.entropy_target is not None:
            self.temperature_holder = TemperatureHolder(
                initial_log_temperature=np.log(initial_temperature))
            if temperature_optimizer is not None:
                self.temperature_optimizer = temperature_optimizer
            else:
                self.temperature_optimizer = chainer.optimizers.Adam()
            self.temperature_optimizer.setup(self.temperature_holder)
            if gpu is not None and gpu >= 0:
                self.temperature_holder.to_gpu(device=gpu)
        else:
            self.temperature_holder = None
            self.temperature_optimizer = None
        self.act_deterministically = act_deterministically

        self.t = 0
        self.prev_episode_end_t = 0
        self.last_state = None
        self.last_action = None

        # Target model
        self.target_q_func1 = copy.deepcopy(self.q_func1)
        self.target_q_func2 = copy.deepcopy(self.q_func2)
        
        # Statistics
        self.q1_record = collections.deque(maxlen=1000)
        self.q2_record = collections.deque(maxlen=1000)
        self.entropy_record = collections.deque(maxlen=1000)
        self.q_func1_loss_record = collections.deque(maxlen=100)
        self.q_func2_loss_record = collections.deque(maxlen=100)

        self.reward_demo_record = collections.deque(maxlen=100)
        self.reward_samp_record = collections.deque(maxlen=100)
        self.reward_loss_record = collections.deque(maxlen=100)

        self.is_discrete = is_discrete
        self.penalty_lamda = penalty_lamda

    def act_and_train(self, obs, reward):

        self.logger.debug('t:%s r:%s', self.t, reward)

        if (self.burnin_action_func is not None
                and self.policy_optimizer.t == 0):
            action = self.burnin_action_func()
        else:
            action = self.select_greedy_action(obs, self.xp.zeros((1, 1), dtype=np.float32))
        self.t += 1

        if self.last_state is not None:
            assert self.last_action is not None
            # Add a transition to the replay buffer
            self.replay_buffer.append(
                state=self.last_state,
                action=self.last_action,
                reward=reward,
                next_state=obs,
                next_action=action,
                is_state_terminal=False)

        self.last_state = obs
        self.last_action = action

        # self.replay_updater.update_if_necessary(self.t)

        return self.last_action

    def act(self, obs):
        return self.select_greedy_action(
            obs, self.xp.zeros((1, 1), dtype=np.float32), deterministic=self.act_deterministically)

    def batch_act(self, batch_obs):
        return self.batch_select_greedy_action(
            batch_obs, self.xp.zeros((len(batch_obs), 1), dtype=np.float32), deterministic=self.act_deterministically)

    def batch_act_and_train(self, batch_obs):
        """Select a batch of actions for training.
        Args:
            batch_obs (Sequence of ~object): Observations.
        Returns:
            Sequence of ~object: Actions.
        """

        if (self.burnin_action_func is not None
                and self.policy_optimizer.t == 0):
            batch_action = [self.burnin_action_func()
                            for _ in range(len(batch_obs))]
        else:
            batch_action = self.batch_select_greedy_action(
                batch_obs, self.xp.zeros((len(batch_obs), 1), dtype=np.float32))

        self.batch_last_obs = list(batch_obs)
        self.batch_last_action = list(batch_action)

        return batch_action

    def batch_select_greedy_action(self, batch_obs, batch_abs, deterministic=False):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            batch_xs = self.batch_states(batch_obs, self.xp, self.phi)
            if deterministic:
                batch_action = self.policy(batch_xs, batch_abs).most_probable.array
            else:
                batch_action = self.policy(batch_xs, batch_abs).sample().array
        return list(cuda.to_cpu(batch_action))

    def select_greedy_action(self, obs, absorb, deterministic=False):
        return self.batch_select_greedy_action(
            [obs], absorb, deterministic=deterministic)[0]

    def batch_observe_and_train(
            self, batch_obs, batch_reward, batch_done, batch_reset):
        for i in range(len(batch_obs)):
            self.t += 1
            if self.batch_last_obs[i] is not None:
                assert self.batch_last_action[i] is not None
                # Add a transition to the replay buffer
                self.replay_buffer.append(
                    state=self.batch_last_obs[i],
                    action=self.batch_last_action[i],
                    reward=batch_reward[i],
                    next_state=batch_obs[i],
                    next_action=None,
                    is_state_terminal=batch_done[i],
                )
                if batch_reset[i] or batch_done[i]:
                    self.batch_last_obs[i] = None
            #self.replay_updater.update_if_necessary(self.t)

    def _calc_target_v(self, state, is_absorb):
        action_distrib = self.policy(state, is_absorb)
        actions, log_prob = \
            action_distrib.sample_with_log_prob()
        # Starting from the goal state we can execute only non-actions.
        actions *= (1 - is_absorb)
        entropy_term = self.temperature * log_prob
        if self.is_discrete:
            q1 = F.select_item(self.target_q_func1(state, is_absorb), actions)
            q2 = F.select_item(self.target_q_func2(state, is_absorb), actions)
        else:
            q1 = self.target_q_func1(state, is_absorb, actions)
            q2 = self.target_q_func2(state, is_absorb, actions)
            entropy_term = entropy_term[..., None]
            entropy_term *= (1 - is_absorb)

        q = F.minimum(q1, q2)
        assert q.shape == entropy_term.shape

        return F.flatten(q - entropy_term)

    def _compute_q_loss(self, batch):
        """B(D, r)"""
        
        batch_reward = self.xp.concatenate([self.xp.ones_like(batch['reward'][:self.minibatch_size]),
                                            self.xp.zeros_like(batch['reward'][self.minibatch_size:])], axis=0)
        batch_state = batch['state']
        batch_next_state = batch['next_state']
        batch_actions = batch['action']
        batch_discount = batch['discount']
        batch_terminal = batch['is_state_terminal']
        batch_absorb = batch['is_state_absorb']
        batch_next_absorb = batch['is_next_state_absorb']
        
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            target_next_v = self._calc_target_v(batch_next_state, batch_next_absorb)

            if self.reward_func:
                # reward for gan
                D = F.sigmoid(self.reward_func(batch_state, batch_absorb, batch_actions))
                batch_reward = F.flatten(F.log(D + 1e-8) - F.log(1 - D + 1e-8)) # + 0.5 * batch_reward / self.temperature / self.lamda

            batch_reward = F.flatten(batch_reward)
            
            self.reward_demo_record.extend(cuda.to_cpu(batch_reward.array[:self.minibatch_size]))
            self.reward_samp_record.extend(cuda.to_cpu(batch_reward.array[self.minibatch_size:]))

            target_q = batch_reward + batch_discount * \
                       (1.0 - batch_terminal) * target_next_v

        if self.is_discrete:
            predict_q1 = F.flatten(F.select_item(self.q_func1(batch_state, batch_absorb), batch_actions))
            predict_q2 = F.flatten(F.select_item(self.q_func2(batch_state, batch_absorb), batch_actions))
        else:
            predict_q1 = F.flatten(self.q_func1(batch_state, batch_absorb, batch_actions))
            predict_q2 = F.flatten(self.q_func2(batch_state, batch_absorb, batch_actions))

        # soft bellman error
        loss1 = 0.5 * F.mean_squared_error(target_q, predict_q1)
        loss2 = 0.5 * F.mean_squared_error(target_q, predict_q2)

        self.q1_record.extend(cuda.to_cpu(predict_q1.array))
        self.q2_record.extend(cuda.to_cpu(predict_q2.array))

        return loss1, loss2

    def _compute_pi_loss(self, batch):
        """Compute loss for actor."""

        batch_absorb = batch['is_state_absorb']
        # Don't update the actor for absorbing states.
        # And skip update if all states are absorbing.
        if all(batch_absorb):
            return 0

        batch_state = batch['state']
        action_distrib = self.policy(batch_state, batch_absorb)
        if self.is_discrete: # for discrete actions
            prob = action_distrib.all_prob
            log_prob = action_distrib.all_log_prob
            q1 = self.q_func1(batch_state, batch_absorb)
            q2 = self.q_func2(batch_state, batch_absorb)
        else:
            actions, log_prob = action_distrib.sample_with_log_prob()
            q1 = self.q_func1(batch_state, batch_absorb, actions)
            q2 = self.q_func2(batch_state, batch_absorb, actions)
            prob = 1
            log_prob = log_prob[...,None]
        q = F.minimum(q1, q2)

        entropy_term = self.temperature * log_prob
        assert q.shape == entropy_term.shape
        loss = F.average((1 - batch_absorb) * prob * (entropy_term - q))

        if self.entropy_target is not None:
            self.update_temperature(log_prob.array)

        # Record entropy
        with chainer.no_backprop_mode():
            try:
                self.entropy_record.extend(
                    cuda.to_cpu(action_distrib.entropy.array))
            except NotImplementedError:
                # Record - log p(x) instead
                self.entropy_record.extend(
                    cuda.to_cpu(-log_prob.array))

        return loss

    def update_reward_func(self, batch_demo, batch_samp):
        batch_demo_state = batch_demo['state']
        batch_demo_absorb = batch_demo['is_state_absorb']
        batch_demo_actions = batch_demo['action']
        batch_samp_state = batch_samp['state']
        batch_samp_absorb = batch_samp['is_state_absorb']
        batch_samp_actions = batch_samp['action']

        # -log(D_demo(x)) for GAN
        y_demo = self.reward_func(batch_demo_state, batch_demo_absorb, batch_demo_actions)
        loss_demo = - F.average(F.log(F.sigmoid(y_demo) + 1e-8))
        
        
        # -log(1 - D_samp(x)) for GAN
        y_samp = self.reward_func(batch_samp_state, batch_samp_absorb, batch_samp_actions)
        loss_samp = - F.average(F.log(1 - F.sigmoid(y_samp) + 1e-8))

        # grad penalty
        eps = self.xp.random.uniform(0, 1, size=len(batch_demo_state)).astype("f")
        if len(batch_samp_state.shape) == 2:
            eps = eps[:, None]
        elif len(batch_samp_state.shape) == 4:
            eps = eps[:, None, None, None]

        s_mid = eps * batch_demo_state + (1.0 - eps) * batch_samp_state
        abs_mid = eps * batch_demo_absorb + (1.0 - eps) * batch_samp_absorb
        act_mid = eps * batch_demo_actions + (1.0 - eps) * batch_samp_actions

        x_mid = np.concatenate((s_mid, abs_mid, act_mid), axis=-1)
        x_mid = chainer.Variable(x_mid)
        y_mid = self.reward_func.forward(x_mid)

        grad, = chainer.grad([y_mid], [x_mid], enable_double_backprop=True)
        penalty = 0.5 * self.penalty_lamda * F.mean(F.batch_l2_norm_squared(grad))
        # penalty = self.penalty_lamda * F.mean_squared_error(grad, self.xp.zeros_like(grad.array))
        loss = loss_demo + loss_samp + penalty
        self.reward_loss_record.append(float(loss.array))
        # self.penalty_record.append(float(penalty.array))
        self.reward_func_optimizer.update(lambda: loss)

    def _compute_bc_loss(self, batch):

        # behavioral cloning
        batch_state = batch['state']
        batch_actions = batch['action']
        batch_absorb = batch['is_state_absorb']

        action_distrib = self.policy(batch_state, batch_absorb)
        action = action_distrib.sample()
        loss = F.mean_squared_error(F.flatten(batch_actions), F.flatten(action))

        return loss

    def pretrain(self, num_iters):
        """Behavioral Cloning"""
        for iter in range(num_iters):
            batch_demo = batch_experiences(
                self.replay_buffer_demo.sample(self.minibatch_size),
                xp=self.xp, phi=self.phi, gamma=self.gamma)
            # actor
            loss_pi = self._compute_bc_loss(batch_demo)
            self.policy_optimizer.update(lambda: loss_pi)
            if iter % 1000 == 0:
                print(f'iteration: {iter}, loss pi: {loss_pi.array}')


    def _calc_absorb_value(self, batch_demo):
        absorb_state = self.xp.zeros((1, batch_demo['state'].shape[1]), dtype=batch_demo['state'].dtype)
        is_absorb = self.xp.ones((1, 1), dtype=self.xp.float32)
        absorb_action = self.xp.zeros((1, 1), dtype=self.xp.float32)
        # absorb_reward = self.reward_func(absorb_state, is_absorb, absorb_action)
        absorb_q1 =  self.q_func1(absorb_state, is_absorb, absorb_action)
        absorb_q2 = self.q_func2(absorb_state, is_absorb, absorb_action)
        # assert absorb_q1.shape == absorb_reward.shape
        # return F.flatten(absorb_reward + self.gamma * F.minimum(absorb_q1, absorb_q2))
        return F.flatten(absorb_q1), F.flatten(absorb_q2)

    def concat_demo(self, batch_demo, batch_sample):
        batch = {}
        for key in batch_demo.keys():
            batch[key] = self.xp.concatenate((batch_demo[key], batch_sample[key]), axis=0)

        return batch

    def train_from_demo(self, episode_len):

        # update reward function
        if self.reward_func:
            for epoch in range(episode_len):
                batch_demo = batch_experiences(
                    self.replay_buffer_demo.sample(self.minibatch_size),
                    xp=self.xp, phi=self.phi, gamma=self.gamma)
                batch_sample = batch_experiences(
                    self.replay_buffer.sample(self.minibatch_size),
                    xp=self.xp, phi=self.phi, gamma=self.gamma)

                self.update_reward_func(batch_demo, batch_sample)

        # update actor critic
        for epoch in range(episode_len):
            batch_demo = batch_experiences(
                self.replay_buffer_demo.sample(self.minibatch_size),
                xp=self.xp, phi=self.phi, gamma=self.gamma)
            batch_sample = batch_experiences(
                self.replay_buffer.sample(self.minibatch_size),
                xp=self.xp, phi=self.phi, gamma=self.gamma)
            batch = self.concat_demo(batch_demo, batch_sample)
            
            loss_q1, loss_q2 = self._compute_q_loss(batch)
            self.q_func1_optimizer.update(lambda: loss_q1)
            self.q_func2_optimizer.update(lambda: loss_q2)

            loss_pi = self._compute_pi_loss(batch_sample)
#             loss_pi = self._compute_pi_loss(batch)
            self.policy_optimizer.update(lambda: loss_pi)

            # # decay learning rate
            if self.policy_optimizer.t % 100000 == 0:
                if self.reward_func_optimizer:
                    self.reward_func_optimizer.alpha *= 0.5
                self.q_func1_optimizer.alpha *= 0.5
                self.q_func2_optimizer.alpha *= 0.5
                self.policy_optimizer.alpha *= 0.5

            # Update stats
            self.q_func1_loss_record.append(float(loss_q1.array))
            self.q_func2_loss_record.append(float(loss_q2.array))

            self.sync_target_network()

    def stop_episode_and_train(self, state, reward, done=False):

        assert self.last_state is not None
        assert self.last_action is not None

        # Add a transition to the replay buffer
        self.replay_buffer.append(
            state=self.last_state,
            action=self.last_action,
            reward=reward,
            next_state=state,
            next_action=self.last_action,
            is_state_terminal=done)


        episode_size = self.t - self.prev_episode_end_t
        self.stop_episode()

        if self.t >= self.minibatch_size and self.t >= self.replay_start_size:
            self.train_from_demo(episode_size)

    def stop_episode(self):
        self.last_state = None
        self.last_action = None

        self.prev_episode_end_t = self.t
        self.replay_buffer.stop_current_episode()

    def get_statistics(self):
        return [
            ('average_q1', _mean_or_nan(self.q1_record)),
            ('average_q2', _mean_or_nan(self.q2_record)),
            ('average_q_func1_loss', _mean_or_nan(self.q_func1_loss_record)),
            ('average_q_func2_loss', _mean_or_nan(self.q_func2_loss_record)),
            ('n_updates', self.policy_optimizer.t),
            ('average_entropy', _mean_or_nan(self.entropy_record)),
            ('temperature', self.temperature),
            ('reward_demo', _mean_or_nan(self.reward_demo_record)),
            ('reward_sample', _mean_or_nan(self.reward_samp_record)),
            # ('reward_loss', _mean_or_nan(self.reward_loss_record)),
        ]

    def save(self, dirname, save_replay=True):
        super().save(dirname)
        if save_replay:
            self.replay_buffer.save(os.path.join(dirname, 'replay'))

    def load(self, dirname, load_replay=True):
        super().load(dirname)
        if load_replay:
            self.replay_buffer.load(os.path.join(dirname, 'replay'))
