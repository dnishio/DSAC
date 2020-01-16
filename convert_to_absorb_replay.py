"""
chainerrl.replay_buffer.ReplayBuffer

"""
import os
import argparse
import numpy as np
from chainerrl.replay_buffer import ReplayBuffer
from replay_buffer import AbsorbReplayBuffer

def convert(rbuf):
    absorb_state = np.zeros(rbuf.memory[0][0]['state'].shape, dtype=rbuf.memory[0][0]['state'].dtype)
    absorb_action = rbuf.memory[0][0]['action'] * 0
    new_rbuf = AbsorbReplayBuffer(rbuf.capacity, absorb_state, absorb_action)

    for elems in rbuf.memory:
        for elem in elems:
            elem['is_state_terminal']=False
            new_rbuf.append(**elem)
            # if new_rbuf.memory[-1][0]['is_state_absorb']:
            #     print(new_rbuf.memory[-1])

    assert len(rbuf) <= len(new_rbuf)
    return new_rbuf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str,
                        help='replay buffer directory')
    parser.add_argument('--save', type=str,
                        help='replay buffer directory')
    args = parser.parse_args()
    rbuf = ReplayBuffer(5 * 10 ** 5)
    rbuf.load(os.path.join(args.load, 'replay'))
    new_rbuf = convert(rbuf)
    new_rbuf.save(os.path.join(args.save, 'replay'))
