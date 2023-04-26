'''Code implementing the two-step task environment.'''
# © Thomas Akam, 2023, released under the GPLv3 licence.

import numpy as np

def withprob(p):
    return np.random.rand() < p

# State and action IDs
initiate = 0
sec_step_A = 1
sec_step_B = 2

#Action IDs
choose_A = 3
choose_B = 4

#State IDs
choice = 3
reward_A = 4
reward_B = 5

class Two_step():

    def __init__(self, common_prob=0.8, good_prob=0.8, block_len=[20,40]):
        super(Two_step, self).__init__()
        self.common_prob  = common_prob
        self.good_prob = good_prob
        self.block_len = block_len 
        self.reset()
        self.n_actions = 5 # {0: initiate, 1: choice A, 2: choice B, 3: sec. step A, 4: sec. step B}
        self.n_states  = 6 # {0: initiate, 1: choice, 2: sec. step A, 3: sec. step B, 4: reward A, 5: reward B}

    def step(self, action):
        '''Recieve action and return next state and reward.'''
        reward = 0
        if self.state in (initiate, reward_A, reward_B):
            if action == initiate: # Initate action.
                self.state = choice # Choice state.
                self.trial_n += 1
                self.block_trial += 1
                self.trials_till_reversal -= 1
                if self.trials_till_reversal == 0:
                    self.block_n += 1
                    self.A_good = not self.A_good
                    self.block_trial = 0
                    self.trials_till_reversal = np.random.randint(*self.block_len)
        elif self.state == choice: # Choice state
            if action == choose_A:
                if withprob(self.common_prob):
                    self.state = sec_step_A
                else:
                    self.state = sec_step_B
            elif action == choose_B:
                if withprob(self.common_prob):
                    self.state = sec_step_B
                else:
                    self.state = sec_step_A
        elif self.state == sec_step_A: # Second step A state
            if action == sec_step_A:
                if ((self.A_good and withprob(self.good_prob)) or
                    (not self.A_good and withprob(1-self.good_prob))):
                    reward = 1
                    self.state = reward_A
                else:
                    self.state = initiate
        elif self.state == sec_step_B:
            if action == sec_step_B: 
                if ((self.A_good and withprob(1 - self.good_prob)) or
                    (not self.A_good and withprob(self.good_prob))):
                    reward = 1
                    self.state = reward_B
                else:
                    self.state = initiate
        return self.state, reward
    
    def reset(self):
        '''Reset the state of the environment to an initial state.'''
        self.trial_n = 0
        self.block_n = 0
        self.block_trial = 0
        self.trials_till_reversal = np.random.randint(*self.block_len)
        self.A_good = np.random.rand() > 0.5 # True if A is good, false if B is good.
        self.state = initiate # initiate
        self.prev_state = None
        return self.state

    def sample_appropriate_action(self, state):
        '''Return a randomly selected action that is valid given the state.'''
        if state in (initiate, reward_A, reward_B):
            action = initiate
        elif state == choice:
            if withprob(0.5):
                action = choose_A
            else:
                action = choose_B
        elif state == sec_step_A:
            action = sec_step_A
        elif state == sec_step_B:
            action = sec_step_B
        return action
