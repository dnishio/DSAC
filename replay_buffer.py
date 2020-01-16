import numpy as np
from chainerrl.replay_buffer import ReplayBuffer
from chainerrl.misc.batch_states import batch_states

class AbsorbReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, absorb_state, absorb_action, num_steps=1):
        self.absorb_state = absorb_state
        self.absorb_action = absorb_action
        super().__init__(capacity, num_steps)

    def append(self, state, action, reward, next_state=None, next_action=None,
               is_state_terminal=False, is_state_absorb=False, env_id=0, **kwargs):
        last_n_transitions = self.last_n_transitions[env_id]
        experience = dict(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                next_action=next_action,
                is_state_terminal=False,
                is_state_absorb=False,
                is_next_state_absorb=False,
                **kwargs
            )
        last_n_transitions.append(experience)

                
        if is_state_terminal:
            if len(last_n_transitions) == self.num_steps:
                self.memory.append(list(last_n_transitions))
            # Terminal State -> Absorbing state
            experience = dict(
                state=next_state,
                action=next_action,
                reward=0,
                next_state=self.absorb_state,
                next_action=self.absorb_action,
                is_state_terminal=False,
                is_state_absorb=False,
                is_next_state_absorb=True,
                **kwargs
            )
            last_n_transitions.append(experience)

            if len(last_n_transitions) == self.num_steps:
                self.memory.append(list(last_n_transitions))

            # Absorbing state -> Absorbing state
            experience = dict(
                state=self.absorb_state,
                action=self.absorb_action,
                reward=0,
                next_state=self.absorb_state,
                next_action=self.absorb_action,
                is_state_terminal=False,
                is_state_absorb=True,
                is_next_state_absorb=True,
                **kwargs
            )
            last_n_transitions.append(experience)

        if is_state_terminal:
            while last_n_transitions:
                self.memory.append(list(last_n_transitions))
                del last_n_transitions[0]
            assert len(last_n_transitions) == 0
        else:
            if len(last_n_transitions) == self.num_steps:
                self.memory.append(list(last_n_transitions))


def batch_experiences(experiences, xp, phi, gamma, batch_states=batch_states):
    """Takes a batch of k experiences each of which contains j

    consecutive transitions and vectorizes them, where j is between 1 and n.

    Args:
        experiences: list of experiences. Each experience is a list
            containing between 1 and n dicts containing
              - state (object): State
              - action (object): Action
              - reward (float): Reward
              - is_state_terminal (bool): True iff next state is terminal
              - next_state (object): Next state
        xp : Numpy compatible matrix library: e.g. Numpy or CuPy.
        phi : Preprocessing function
        gamma: discount factor
        batch_states: function that converts a list to a batch
    Returns:
        dict of batched transitions
    """

    batch_exp = {
        'state': batch_states(
            [elem[0]['state'] for elem in experiences], xp, phi),
        'action': xp.asarray([elem[0]['action'] for elem in experiences]),
        'reward': xp.asarray([sum((gamma ** i) * exp[i]['reward']
                                  for i in range(len(exp)))
                              for exp in experiences],
                             dtype=np.float32),
        'next_state': batch_states(
            [elem[-1]['next_state']
             for elem in experiences], xp, phi),
        'is_state_terminal': xp.asarray(
            [any(transition['is_state_terminal']
                 for transition in exp) for exp in experiences],
            dtype=np.float32),
        'discount': xp.asarray([(gamma ** len(elem))for elem in experiences],
                               dtype=np.float32)}
    if all(elem[-1]['next_action'] is not None for elem in experiences):
        batch_exp['next_action'] = xp.asarray(
            [elem[-1]['next_action'] for elem in experiences])

    try:
        batch_exp['is_state_absorb'] = xp.asarray(
                [[any(transition['is_state_absorb']
                     for transition in exp)] for exp in experiences],
                dtype=np.float32)
        batch_exp['is_next_state_absorb'] = xp.asarray(
            [[any(transition['is_next_state_absorb']
                  for transition in exp)] for exp in experiences],
            dtype=np.float32)
    except KeyError:
        batch_exp['is_state_absorb'] = xp.zeros((len(experiences), 1), dtype=np.float32)
        batch_exp['is_next_state_absorb'] = xp.zeros((len(experiences), 1), dtype=np.float32)

    return batch_exp