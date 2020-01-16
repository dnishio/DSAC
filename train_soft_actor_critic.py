"""A training script of Soft Actor-Critic on OpenAI Gym Mujoco environments.
This script follows the settings of https://arxiv.org/abs/1812.05905 as much
as possible.
"""
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()  # NOQA

import argparse
import functools
import logging
import os
import sys
import copy

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

from replay_buffer import AbsorbReplayBuffer
import network
import sac

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
    parser.add_argument('--expert-num-episode', type=int, default=0,
                        help='the number of expert trajectory, if 0, no create demo mode.')
    parser.add_argument('--steps', type=int, default=10 ** 6,
                        help='Total number of timesteps to train the agent.')
    parser.add_argument('--eval-n-runs', type=int, default=10,
                        help='Number of episodes run for each evaluation.')
    parser.add_argument('--eval-interval', type=int, default=5000,
                        help='Interval in timesteps between evaluations.')
    parser.add_argument('--replay-start-size', type=int, default=10000,
                        help='Minimum replay buffer size before ' +
                        'performing gradient updates.')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Minibatch size')
    parser.add_argument('--render', action='store_true',
                        help='Render env states in a GUI window.')
    parser.add_argument('--demo', action='store_true',
                        help='Just run evaluation, not training.')
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

    logging.basicConfig(level=args.logger_level)

    if args.debug:
        chainer.set_debug(True)
    if args.expert_num_episode == 0:
        args.outdir = experiments.prepare_output_dir(
            args, args.outdir, argv=sys.argv, time_format=f'{args.env}_{args.seed}')
    else:
        args.outdir = experiments.prepare_output_dir(
            args, args.outdir, argv=sys.argv, time_format=f'{args.env}_{args.expert_num_episode}expert')
        args.replay_start_size = 1e8
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
        head = network.FCHead()
        phi = lambda x: x

    else:
        head = network.CNNHead(n_input_channels=4)
        phi = lambda x: np.asarray(x, dtype=np.float32) / 255

    if isinstance(action_space, Box):
        action_size = action_space.low.size
        policy = network.GaussianPolicy(copy.deepcopy(head), action_size)
        q_func1 = network.QSAFunction(copy.deepcopy(head), action_size)
        q_func2 = network.QSAFunction(copy.deepcopy(head), action_size)

        def burnin_action_func():
            """Select random actions until model is updated one or more times."""
            return np.random.uniform(
                action_space.low, action_space.high).astype(np.float32)

    else:
        action_size = action_space.n

        policy = network.SoftmaxPolicy(copy.deepcopy(head), action_size)
        q_func1 = network.QSFunction(copy.deepcopy(head), action_size)
        q_func2 = network.QSFunction(copy.deepcopy(head), action_size)

        def burnin_action_func():
            return np.random.randint(0, action_size)

    policy_optimizer = optimizers.Adam(3e-4).setup(policy)
    q_func1_optimizer = optimizers.Adam(3e-4).setup(q_func1)
    q_func2_optimizer = optimizers.Adam(3e-4).setup(q_func2)

    # Draw the computational graph and save it in the output directory.
    # fake_obs = chainer.Variable(
    #     policy.xp.zeros_like(obs_space.low, dtype=np.float32)[None],
    #     name='observation')
    # fake_action = chainer.Variable(
    #     policy.xp.zeros_like(action_space.low, dtype=np.float32)[None],
    #     name='action')
    # chainerrl.misc.draw_computational_graph(
    #     [policy(fake_obs)], os.path.join(args.outdir, 'policy'))
    # chainerrl.misc.draw_computational_graph(
    #     [q_func1(fake_obs, fake_action)], os.path.join(args.outdir, 'q_func1'))
    # chainerrl.misc.draw_computational_graph(
    #     [q_func2(fake_obs, fake_action)], os.path.join(args.outdir, 'q_func2'))

    rbuf = replay_buffer.ReplayBuffer(10 ** 6)

    # Hyperparameters in http://arxiv.org/abs/1802.09477
    agent = sac.SoftActorCritic(
        policy,
        q_func1,
        q_func2,
        policy_optimizer,
        q_func1_optimizer,
        q_func2_optimizer,
        rbuf,
        gamma=0.99,
        is_discrete=isinstance(action_space, Discrete),
        replay_start_size=args.replay_start_size,
        gpu=args.gpu,
        minibatch_size=args.batch_size,
        phi=phi,
        burnin_action_func=burnin_action_func,
        entropy_target=-action_size if isinstance(action_space, Box) else -np.log((1.0 / action_size)) * 0.98,
        temperature_optimizer=chainer.optimizers.Adam(3e-4),
    )

    if len(args.load) > 0:
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
    elif args.expert_num_episode > 0:
        episode_r = 0
        env = sample_env
        episode_len = 0
        t = 0
        logger = logging.getLogger(__name__)
        episode_results = []
        try:
            for ep in range(args.expert_num_episode):
                obs = env.reset()
                r = 0
                while True:
                    # a_t
                    action = agent.act_and_train(obs, r)
                    # o_{t+1}, r_{t+1}
                    obs, r, done, info = env.step(action)
                    t += 1
                    episode_r += r
                    episode_len += 1
                    reset = (episode_len == timestep_limit
                             or info.get('needs_reset', False))
                    if done or reset:
                        agent.stop_episode_and_train(obs, r, done=done)
                        logger.info('outdir:%s step:%s episode:%s R:%s',
                                    args.outdir, t, ep, episode_r)
                        episode_results.append(episode_r)
                        episode_r = 0
                        episode_len = 0
                        break
            logger.info('mean: %s', sum(episode_results)/ len(episode_results))
        except (Exception, KeyboardInterrupt):
            raise

        # Save
        save_name = os.path.join(
            os.path.join('demos', f'{args.expert_num_episode}_episode'), args.env)
        makedirs(save_name, exist_ok=True)
        agent.replay_buffer.save(os.path.join(save_name, 'replay'))
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
