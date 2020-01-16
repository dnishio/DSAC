import os
import argparse
import numpy as np
from chainerrl.replay_buffer import ReplayBuffer


def convert_demo_to_replay(demo, limit=4):
    traj_data = np.load(demo, allow_pickle=True)
    rbuf = ReplayBuffer(capacity=100000)

    for ep in range(len(traj_data['obs'][:limit])):
        obss = traj_data['obs'][ep]
        acss = traj_data['acs'][ep]
        rewss = traj_data['rews'][ep]
        for t in range(len(obss)-1):
            # TODO: How do we judge whether last state is terminal or not?
            rbuf.append(
                state=obss[t],
                action=acss[t],
                reward=rewss[t],
                next_state=obss[t+1],
                next_action=acss[t+1],
                is_state_terminal=False,
            )
    return rbuf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str,
                        choices=[
                            'Hopper',
                            'HalfCheetah',
                            'Humanoid',
                            'Walker2d',
                        ],
                        help='mujoco env name')
    parser.add_argument('--load', type=str, default='data',
                        help='.npz file of openai')
    parser.add_argument('--save', type=str,
                        help='replay buffer directory')
    parser.add_argument('--n_epoch', type=int, default=4,
                        help='the number of expert demo epoch')
    args = parser.parse_args()

    load = os.path.join(args.load, f'deterministic.trpo.{args.env}.0.00.npz')
    if not os.path.exists(load):
        raise FileNotFoundError(f'{load} is not Found.')
    save = os.path.join(args.save, f'{args.n_epoch}_episodes', args.env)
    os.makedirs(save, exist_ok=True)
    args = parser.parse_args()
    rbuf = convert_demo_to_replay(load, args.n_epoch)
    print('replay size:', len(rbuf))
    rbuf.save(os.path.join(save, 'replay'))
