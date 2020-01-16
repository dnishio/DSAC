"""A training script of Soft Actor-Critic on OpenAI Gym Mujoco environments.
This script follows the settings of https://arxiv.org/abs/1812.05905 as much
as possible.
"""
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
#from future import standard_library
#standard_library.install_aliases()  # NOQA

import argparse
import functools
import logging
import os
import copy
import sys

import chainer
from chainer import functions as F
from chainer import links as L
from chainer import optimizers
# import roboschool
import gym
import pybullet_envs
from gym.spaces import Discrete, Box
import gym.wrappers
import wrappers
import distribution
import numpy as np

import chainerrl
from chainerrl import experiments
from chainerrl import misc
from chainerrl import replay_buffer
from chainerrl.wrappers import atari_wrappers
from chainerrl.misc.makedirs import makedirs
import sqil
import network_sqil
from replay_buffer import AbsorbReplayBuffer
from chainer.optimizer_hooks import GradientClipping

def concat_obs_and_action(obs, action):
    """Concat observation and action to feed the critic."""
    return F.concat((obs, action), axis=-1)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--env', type=str, choices=['Pendulum-v0', 'AntBulletEnv-v0',
                                                    'HalfCheetahBulletEnv-v0', 'HumanoidBulletEnv-v0', 
                                                    'HopperBulletEnv-v0', 'Walker2DBulletEnv-v0'
                                                    ],
                        help='OpenAI Gym env and Pybullet (roboschool) env to perform algorithm on.')
    parser.add_argument('--num-envs', type=int, default=1,
                        help='Number of envs run in parallel.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 32)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU to use, set to -1 if no GPU.')
    parser.add_argument('--load', type=str, default='',
                        help='Directory to load agent from.')
    parser.add_argument('--load-demo', type=str, default='',
                        help='Directory to load replay buffer of demo from.')
    parser.add_argument('--expert-num-episode', type=int, default=0,
                        help='the number of expert trajectory, if 0, no create demo mode.')
    parser.add_argument('--absorb', action='store_true',
                        help='Add absorb state or do not.')
    parser.add_argument('--lamda', type=float, default=1,
                        help='reguralized lambda')
    parser.add_argument('--reward-func', action='store_true',
                        help='Add absorb state or do not.')
    parser.add_argument('--steps', type=int, default=10 ** 6,
                        help='Total number of timesteps to train the agent.')
    parser.add_argument('--eval-n-runs', type=int, default=10,
                        help='Number of episodes run for each evaluation.')
    parser.add_argument('--eval-interval', type=int, default=5000,
                        help='Interval in timesteps between evaluations.')
    parser.add_argument('--replay-start-size', type=int, default=10000,
                        help='Minimum replay buffer size before ' +
                        'performing gradient updates.')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Minibatch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Learning rate of optimizers')
    parser.add_argument('--render', action='store_true',
                        help='Render env states in a GUI window.')
    parser.add_argument('--demo', action='store_true',
                        help='Just run evaluation, not training.')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='The number of pretrain iterations')
    parser.add_argument('--monitor', action='store_true',
                        help='Wrap env with gym.wrappers.Monitor.')
    parser.add_argument('--log-interval', type=int, default=1000,
                        help='Interval in timesteps between outputting log'
                             ' messages during training')
    parser.add_argument('--logger-level', type=int, default=logging.INFO,
                        help='Level of the root logger.')
    parser.add_argument('--policy-output-scale', type=float, default=1.,
                        help='Weight initialization scale of polity output.')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode.')
    args = parser.parse_args()
    if args.expert_num_episode > 0:
        args.absorb = True

    logging.basicConfig(level=args.logger_level)

    if args.debug:
        chainer.set_debug(True)

    dir_name = f'{args.env}_{args.seed}'
    dir_name += '_demo' if args.demo else ''
    dir_name += '_absorb' if args.absorb else ''
    dir_name += '_reward' if args.reward_func else ''
#     dir_name += f'_{args.lamda}'

    args.outdir = experiments.prepare_output_dir(
        args, args.outdir, argv=sys.argv, time_format=dir_name)
    print('Output files are saved in {}'.format(args.outdir))

    # Set a random seed used in ChainerRL
    misc.set_random_seed(args.seed, gpus=(args.gpu,))

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 32

    def make_env(process_idx, test):
        env = gym.make(args.env)
        # Unwrap TimiLimit wrapper
        assert isinstance(env, gym.wrappers.TimeLimit)
        env = env.env
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[process_idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        env.seed(env_seed)
        if isinstance(env.observation_space, Box):
            # Cast observations to float32 because our model uses float32
            env = chainerrl.wrappers.CastObservationToFloat32(env)
        else:
            env = atari_wrappers.wrap_deepmind(
                atari_wrappers.make_atari(args.env, max_frames=None),
                episode_life=not test,
                clip_rewards=not test)
        # if args.absorb:
        #     if isinstance(env.observation_space, Box):
        #         env = wrappers.AbsorbingWrapper(env)
        #     else:
        #         raise NotImplementedError('Currently, my AbsorobingWrapper is not available for Discrete observation.')
        if isinstance(env.action_space, Box):
            # Normalize action space to [-1, 1]^n
            env = wrappers.NormalizeActionSpace(env)
        if args.monitor:
            env = gym.wrappers.Monitor(env, args.outdir)
        if args.render:
            env = chainerrl.wrappers.Render(env)
        return env

    def make_batch_env(test):
        return chainerrl.envs.MultiprocessVectorEnv(
            [functools.partial(make_env, idx, test)
             for idx, env in enumerate(range(args.num_envs))])

    sample_env = make_env(process_idx=0, test=False)
    timestep_limit = sample_env.spec.tags.get(
        'wrapper_config.TimeLimit.max_episode_steps')
    obs_space = sample_env.observation_space
    action_space = sample_env.action_space
    print('Observation space:', obs_space)
    print('Action space:', action_space)

    if isinstance(obs_space, Box):
        head = network_sqil.FCHead()
        phi = lambda x: x
    else:
        head = network_sqil.CNNHead(n_input_channels=4)
        phi = lambda x: np.asarray(x, dtype=np.float32) / 255

    if isinstance(action_space, Box):
        action_size = action_space.low.size
        policy = network_sqil.GaussianPolicy(copy.deepcopy(head), action_size)
        q_func1 = network_sqil.QSAFunction(copy.deepcopy(head), action_size)
        q_func2 = network_sqil.QSAFunction(copy.deepcopy(head), action_size)
        def burnin_action_func():
            """Select random actions until model is updated one or more times."""
            return np.random.uniform(
                action_space.low, action_space.high).astype(np.float32)

    else:
        action_size = action_space.n

        policy = network_sqil.SoftmaxPolicy(copy.deepcopy(head), action_size)
        q_func1 = network_sqil.QSFunction(copy.deepcopy(head), action_size)
        q_func2 = network_sqil.QSFunction(copy.deepcopy(head), action_size)

        def burnin_action_func():
            return np.random.randint(0, action_size)

    policy_optimizer = optimizers.Adam(args.learning_rate).setup(policy)
    policy_optimizer.add_hook(GradientClipping(40))
    q_func1_optimizer = optimizers.Adam(args.learning_rate).setup(q_func1)
    q_func2_optimizer = optimizers.Adam(args.learning_rate).setup(q_func2)

    if args.reward_func:
        reward_func = network_sqil.FCRewardFunction(action_size)
        reward_func_optimizer = optimizers.Adam(args.learning_rate).setup(reward_func)
    else:
        reward_func = None
        reward_func_optimizer = None
    # Draw the computational graph and save it in the output directory.
    fake_obs = chainer.Variable(
        policy.xp.zeros_like(obs_space.low, dtype=np.float32)[None],
        name='observation')
    fake_action = chainer.Variable(
        policy.xp.zeros_like(action_space.low, dtype=np.float32)[None],
        name='action')
    fake_absorb = chainer.Variable(
        policy.xp.zeros_like([1], dtype=np.float32)[None],
        name='absorb')
    chainerrl.misc.draw_computational_graph(
        [policy(fake_obs, fake_absorb).sample()], os.path.join(args.outdir, 'policy'))
    chainerrl.misc.draw_computational_graph(
        [q_func1(fake_obs, fake_absorb, fake_action)], os.path.join(args.outdir, 'q_func1'))
    chainerrl.misc.draw_computational_graph(
        [q_func2(fake_obs, fake_absorb, fake_action)], os.path.join(args.outdir, 'q_func2'))

    if args.absorb:
        absorb_state = sample_env.observation_space.sample() * 0
        absorb_action = sample_env.action_space.sample() * 0
        rbuf = AbsorbReplayBuffer(5 * 10 ** 5, absorb_state, absorb_action)
    else:
        rbuf = replay_buffer.ReplayBuffer(5 * 10 ** 5)
    rbuf_demo = replay_buffer.ReplayBuffer(5 * 10 ** 5)

    if len(args.load_demo) > 0:
        rbuf_demo.load(os.path.join(args.load_demo, 'replay'))
        if args.absorb:
            from convert_to_absorb_replay import convert
            rbuf_demo = convert(rbuf_demo)
            assert isinstance(rbuf_demo, AbsorbReplayBuffer)
        else:
            assert isinstance(rbuf_demo, replay_buffer.ReplayBuffer)


    # Hyperparameters in http://arxiv.org/abs/1802.09477
    agent = sqil.SQIL(
        policy,
        q_func1,
        q_func2,
        reward_func,
        policy_optimizer,
        q_func1_optimizer,
        q_func2_optimizer,
        reward_func_optimizer,
        rbuf,
        rbuf_demo,
        gamma=0.99,
        is_discrete=isinstance(action_space, Discrete),
        replay_start_size=args.replay_start_size,
        gpu=args.gpu,
        minibatch_size=args.batch_size,
        phi=phi,
        burnin_action_func=burnin_action_func,
        entropy_target=-action_size if isinstance(action_space, Box) else -np.log((1.0 / action_size)) * 0.98,
        temperature_optimizer=chainer.optimizers.Adam(3e-4),
        lamda=args.lamda
    )

    if args.load:
        agent.load(args.load, args.expert_num_episode == 0)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=make_env(process_idx=0, test=True),
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
            max_episode_len=timestep_limit,
        )
        print('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:
        experiments.train_agent_with_evaluation(
            agent=agent,
            env=make_env(process_idx=0, test=False),
            eval_env=make_env(process_idx=0, test=True),
            outdir=args.outdir,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            # log_interval=args.log_interval,
            train_max_episode_len=timestep_limit,
            eval_max_episode_len=timestep_limit,
        )


if __name__ == '__main__':
    main()
