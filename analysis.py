#%%  Imports 

import numpy as np
import pylab as plt
from sklearn.decomposition import PCA

import Two_step_task as ts

plt.rcParams['pdf.fonttype'] = 42
plt.rc("axes.spines", top=False, right=False)

#%% Analysis functions

def plot_performance(episode_buffer, task, last_n=100, fig_no=1):
    steps_per_trial = []
    rewards_per_trial = []
    for episode in episode_buffer:
        states, rewards, actions, pfc_input, pfc_activity, values, n_trials = episode
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
    # print(f'Ave. rewards per trial, last {last_n} episodes: {np.mean(rewards_per_trial[-last_n:]) :.2f}')
    plt.show()
    
def stay_probability_analysis(episode_buffer, last_n=100, fig_no=2):
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
    
def _get_CSTO(episode, return_inds=False):
    states, rewards, actions, pfc_input, pfc_activity, values, n_trials = episode
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
    
def sec_step_value_analysis(episode_buffer, gamma):
    episode = episode_buffer[-1]
    states, rewards, actions, pfc_input, pfc_activity, values, n_trials = episode
    RPE = np.diff(np.squeeze(values))+rewards[1:]
    sec_step_inds = np.where(np.isin(states, (ts.sec_step_A, ts.sec_step_B)))[0]
    rew_ss_inds = np.where(np.array(rewards)[sec_step_inds[:-1]+1] == 1)
    non_ss_inds = np.where(np.array(rewards)[sec_step_inds[:-1]+1] == 0)
    
    choices, sec_steps, transitions, outcomes = _get_CSTO(episode)
    sec_step_inds = np.where(np.isin(states, (ts.sec_step_A, ts.sec_step_B)))[0]
    sec_step_inds = sec_step_inds[np.hstack([True,~(np.diff(sec_step_inds) == 1)])]
    sec_step_values = np.squeeze(values)[sec_step_inds]
    v_diffs = np.diff(sec_step_values)
    
    same_rew_difs = v_diffs[sec_steps]
    
def plot_PFC_PC1(episode_buffer, task, last_n=3, fig_no=3):
    ch_state_PFC_features = []
    for episode in episode_buffer[-last_n:]:
        states, state_f, rewards, actions, values, pfc_input, n_trials = episode
        ch_state_PFC_features.append(np.array(state_f)[np.array(states)==ts.choice,task.n_states:])
    ch_state_PFC_features = np.vstack(ch_state_PFC_features) 
    PC1 = PCA(n_components=1).fit(ch_state_PFC_features).transform(ch_state_PFC_features)
    plt.figure(fig_no, clear=True)
    plt.plot(PC1)
    plt.ylabel('First principle component of\nchoice state PFC activity')
    plt.xlabel('Trials')
    plt.xlim(0,len(PC1))


    
            
        