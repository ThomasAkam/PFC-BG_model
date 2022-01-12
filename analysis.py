#%%  Imports 

import numpy as np
import pylab as plt
from sklearn.decomposition import PCA
from tensorflow import keras

import Two_step_task as ts

plt.rcParams['pdf.fonttype'] = 42
plt.rc("axes.spines", top=False, right=False)

one_hot = keras.utils.to_categorical

#%% Analysis functions

def plot_performance(episode_buffer, task, last_n=100, fig_no=1):
    ''''Plot the evolution of the number of steps needed to complete each trial and the 
    number of rewards per trial across training.'''
    steps_per_trial = []
    rewards_per_trial = []
    for episode in episode_buffer:
        states, rewards, actions, pfc_input, pfc_states, values, n_trials = episode
        steps_per_trial.append(len(states)/n_trials)
        rewards_per_trial.append(sum(rewards)/n_trials)
    plt.figure(fig_no, clear=True)
    plt.subplot(2,1,1)
    plt.plot(steps_per_trial)
    plt.axhline(3, c='k', ls=':')
    plt.xlim(0,len(episode_buffer))
    plt.ylabel('Steps per trial')
    plt.subplot(2,1,2)
    plt.plot(rewards_per_trial)
    plt.axhline(0.5, c='k', ls='--')
    cor_ch_rr = task.good_prob*task.common_prob+(1-task.good_prob)*(1-task.common_prob) # Reward rate if every choice is correct
    plt.axhline(cor_ch_rr, c='k', ls=':')
    plt.ylabel('Rewards per trial')
    plt.xlabel('Episode #')
    plt.yticks(np.arange(0.4,0.9,0.1))
    plt.xlim(0,len(episode_buffer))
    plt.show()
    
def stay_probability_analysis(episode_buffer, last_n=100, fig_no=2):
    '''Standard two-step task stay probability analysis'''
    stay_probs = []
    for episode in episode_buffer[-last_n:]:
        choices, sec_steps, transitions, outcomes = _get_CSTO(episode)
        stays = choices[1:] == choices[:-1]
        sp_comm_rew = np.mean(stays[ transitions[:-1] &  outcomes[:-1]])
        sp_rare_rew = np.mean(stays[~transitions[:-1] &  outcomes[:-1]])
        sp_comm_non = np.mean(stays[ transitions[:-1] & ~outcomes[:-1]])
        sp_rare_non = np.mean(stays[~transitions[:-1] & ~outcomes[:-1]])
        stay_probs.append(np.array([sp_comm_rew, sp_rare_rew, sp_comm_non, sp_rare_non]))
    plt.figure(fig_no, clear=True)
    plt.bar(np.arange(4), np.mean(stay_probs,0), yerr=np.std(stay_probs,0)/np.sqrt(last_n))
    plt.xticks(np.arange(4), ['CR', 'RR', 'CN', 'RN'])
    plt.ylim(ymin=0)
    plt.ylabel('Stay probability')
    
def _get_CSTO(episode):
    '''Get the choices, second step states, transitions and outcomes for one episode as
    a set of binary vectors.'''
    states, rewards, actions, pfc_input, pfc_states, values, n_trials = episode
    choices, sec_steps, outcomes = [],[],[]
    assert states[0] == ts.choice, 'first state of episode should be choice'
    for s,a in zip(states, actions):
        if s == ts.choice and a == ts.choose_A:
            choices.append(1)
        elif s == ts.choice and a == ts.choose_B:
            choices.append(0)
        elif s == ts.sec_step_A and a == ts.sec_step_A:
            sec_steps.append(1)
        elif s == ts.sec_step_B and a == ts.sec_step_B:
                sec_steps.append(0)
        elif s in (ts.reward_A, ts.reward_B) and a == ts.initiate:
            outcomes.append(1)
        elif s == ts.initiate and a == ts.initiate:
            outcomes.append(0)
    assert len(choices) == len(sec_steps) == len(outcomes), 'something went wrong.'
    choices = np.array(choices, bool)
    sec_steps = np.array(sec_steps, bool)
    transitions = choices == sec_steps
    outcomes = np.array(outcomes, bool)
    return choices, sec_steps, transitions, outcomes
    
def sec_step_value_analysis(episode_buffer, Str_model, task, last_n=10, fig_no=1):
    '''Plot the change in value of second-step states from one trial to the next as a function of the trial outcome
    and whether the outcome occured in the same or different second-step state.'''
    value_updates = np.zeros([last_n, 4])
    for i,episode in enumerate(episode_buffer[-last_n:]):
        states, rewards, actions, pfc_input, pfc_states, values, n_trials = episode
        ssi = np.where(np.isin(states[1:], (ts.sec_step_A, ts.sec_step_B)) & # Indicies of second step states excluding repeated visits on a trial.
                       np.equal(states[:-1], ts.choice))[0]+1 
        # Get values of states sec step A and sec step B.
        _, V_ssA = Str_model([one_hot(np.ones(len(ssi), int)*ts.sec_step_A, task.n_states), np.vstack(pfc_states)[ssi,:]])
        _, V_ssB = Str_model([one_hot(np.ones(len(ssi), int)*ts.sec_step_B, task.n_states), np.vstack(pfc_states)[ssi,:]])
        choices, sec_steps, transitions, outcomes = _get_CSTO(episode)
        dVA = np.diff(V_ssA.numpy().squeeze())
        dVB = np.diff(V_ssB.numpy().squeeze())
        rew_same_dV = np.hstack([dVA[(sec_steps[:-1] == 1) &  outcomes[:-1]],
                                 dVB[(sec_steps[:-1] == 0) &  outcomes[:-1]]])
        rew_diff_dV = np.hstack([dVA[(sec_steps[:-1] == 0) &  outcomes[:-1]],
                                 dVB[(sec_steps[:-1] == 1) &  outcomes[:-1]]])                         
        non_same_dV = np.hstack([dVA[(sec_steps[:-1] == 1) & ~outcomes[:-1]],
                                 dVB[(sec_steps[:-1] == 0) & ~outcomes[:-1]]])
        non_diff_dV = np.hstack([dVA[(sec_steps[:-1] == 0) & ~outcomes[:-1]],
                                 dVB[(sec_steps[:-1] == 1) & ~outcomes[:-1]]])          
        value_updates[i,:] = [np.mean(rew_same_dV), np.mean(rew_diff_dV), np.mean(non_same_dV), np.mean(non_diff_dV)]
    plt.figure(fig_no, clear=True)
    plt.errorbar(np.arange(4), np.mean(value_updates,0), yerr = np.std(value_updates,0), ls='none', marker='o')
    plt.xticks(np.arange(4), ['Rew same', 'Rew diff', 'Non same', 'Non diff'])
    plt.axhline(0,c='k',lw=0.5)
    plt.ylabel('Change in state value')
    plt.xlabel('Trial outcome')
 
    
def plot_PFC_PC1(episode_buffer, task, last_n=3, fig_no=3):
    '''Plot the first principle component of variation in the PFC activity in the choice state across trials'''
    ch_state_PFC_features = []
    for episode in episode_buffer[-last_n:]:
        states, rewards, actions, pfc_input, pfc_states, values, n_trials = episode
        ch_state_PFC_features.append(np.array(pfc_states)[np.array(states)==ts.choice].squeeze())
    ch_state_PFC_features = np.vstack(ch_state_PFC_features) 
    PC1 = PCA(n_components=1).fit(ch_state_PFC_features).transform(ch_state_PFC_features)
    plt.figure(fig_no, clear=True)
    plt.plot(PC1)
    plt.ylabel('First principle component of\nchoice state PFC activity')
    plt.xlabel('Trials')
    plt.xlim(0,len(PC1))
