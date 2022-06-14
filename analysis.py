#%%  Imports 

import os
import json
import pickle
import numpy as np
import pylab as plt
import tensorflow as tf
from sklearn.decomposition import PCA
from tensorflow import keras
from functools import partial
from collections import namedtuple

import Two_step_task as ts

plt.rcParams['pdf.fonttype'] = 42
plt.rc("axes.spines", top=False, right=False)

one_hot = keras.utils.to_categorical
sse_loss = keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)

Run_data = namedtuple('Run_data', ['params', 'episode_buffer', 'PFC_model', 'Str_model', 'task']) # Holds data from one simulation run.

#%% Load data.
  
def load_run(run_dir):
    '''Load data from a single simulation run.'''
    with open(os.path.join(run_dir,'params.json'), 'r') as fp:
            params = json.load(fp)
    with open(os.path.join(run_dir, 'episodes.pkl'), 'rb') as f: 
        episode_buffer = pickle.load(f)
    PFC_model = keras.models.load_model(os.path.join(run_dir, 'PFC_model'))
    Str_model = keras.models.load_model(os.path.join(run_dir, 'Str_model'))
    task = ts.Two_step(good_prob=params['good_prob'], block_len=params['block_len'])
    return Run_data(params, episode_buffer, PFC_model, Str_model, task)

def load_experiment(exp_dir):
    '''Load data from an experiment comprising multiple simulation runs'''
    run_dirs = os.listdir(exp_dir)
    experiment_data = [load_run(os.path.join(exp_dir, run_dir)) for run_dir in run_dirs]
    return experiment_data

#%% Analysis functions

# Multi-analysis functions

def make_plots(episode_buffer, task, Str_model, PFC_model):
    '''Run all plotting functions for a single simulation run.'''
    plot_performance(episode_buffer, task)
    stay_probability_analysis(episode_buffer)
    sec_step_value_analysis(episode_buffer, Str_model, PFC_model, task)
    plot_PFC_choice_state_activity(episode_buffer, task)
    

def plot_run(run_dir):
    '''Load data for a single simulation run from disk and run plots.'''
    rd = load_run(run_dir)
    make_plots(rd.episode_buffer, rd.task, rd.Str_model, rd.PFC_model)
    
def plot_experiment(exp_dir):
    '''Load data for an experiment comprising multiple simulation runs and 
    make plots showing cross run standard deviation.'''
    experiment_data = load_experiment(exp_dir)
    stay_probability_analysis_exp(experiment_data)
    sec_step_value_analysis_exp(experiment_data)

# Plot performance ------------------------------------------------------------

def plot_performance(episode_buffer, task, fig_no=1):
    ''''Plot the evolution of the number of steps needed to complete each trial and the 
    number of rewards per trial across training.'''
    steps_per_trial = []
    rewards_per_trial = []
    bias = []
    PFC_outcome_pred_prob = []
    for ep in episode_buffer:
        steps_per_trial.append(len(ep.states)/ep.n_trials)
        rewards_per_trial.append(sum(ep.rewards)/ep.n_trials)
        n_A_choices = np.sum((ep.states == ts.choice) & (ep.actions == ts.choose_A))
        n_B_choices = np.sum((ep.states == ts.choice) & (ep.actions == ts.choose_B))
        bias.append(n_A_choices/(n_A_choices+n_B_choices))
        if ep.pred_states is not None:
            PFC_outcome_pred_prob.append(np.mean((ep.pred_states == ep.states)[
                np.isin(ep.states,[ts.reward_A, ts.reward_B, ts.initiate])]))
    plt.figure(fig_no, clear=True)
    plt.subplot(3,1,1)
    plt.plot(steps_per_trial)
    plt.axhline(3, c='k', ls=':')
    plt.xlim(0,len(episode_buffer))
    plt.ylabel('Steps per trial')
    plt.subplot(3,1,2)
    plt.plot(rewards_per_trial)
    if ep.pred_states is not None:plt.plot(PFC_outcome_pred_prob)
    plt.axhline(0.5, c='k', ls='--')
    cor_ch_rr = task.good_prob*task.common_prob+(1-task.good_prob)*(1-task.common_prob) # Reward rate if every choice is correct
    plt.axhline(cor_ch_rr, c='k', ls=':')
    plt.ylim(ymin=0.4)
    plt.ylabel('Rewards per trial')
    plt.yticks(np.arange(0.4,0.9,0.1))
    plt.xlim(0,len(episode_buffer))
    plt.subplot(3,1,3)
    plt.plot(bias)
    plt.ylabel('Bias')
    plt.xlabel('Episode #')
    plt.xlim(0,len(episode_buffer))
    plt.show()
    
# Stay probability analysis ---------------------------------------------------
    
def stay_probability_analysis(episode_buffer, last_n=10, return_means=False, fig_no=2):
    '''Standard two-step task stay probability analysis for a single simulation run.'''
    stay_probs = []
    for ep in episode_buffer[-last_n:]:
        choices, sec_steps, transitions, outcomes = _get_CSTO(ep)
        stays = choices[1:] == choices[:-1]
        sp_comm_rew = np.mean(stays[ transitions[:-1] &  outcomes[:-1]])
        sp_rare_rew = np.mean(stays[~transitions[:-1] &  outcomes[:-1]])
        sp_comm_non = np.mean(stays[ transitions[:-1] & ~outcomes[:-1]])
        sp_rare_non = np.mean(stays[~transitions[:-1] & ~outcomes[:-1]])
        stay_probs.append(np.array([sp_comm_rew, sp_rare_rew, sp_comm_non, sp_rare_non]))
    if return_means:
        return np.mean(stay_probs,0)
    else:
        _stay_probability_plot(np.array(stay_probs), fig_no)
      
def stay_probability_analysis_exp(experiment_data, fig_no=2):
    '''Stay probability analysis for an experiment comprising mulitple simulation runs.'''
    stay_probs = np.zeros([len(experiment_data),4])
    for i, run_data in enumerate(experiment_data):
        stay_probs[i,:] = stay_probability_analysis(run_data.episode_buffer, return_means=True)
    _stay_probability_plot(stay_probs, fig_no)
    
def _stay_probability_plot(stay_probs, fig_no):
    plt.figure(fig_no, clear=True)
    plt.bar(np.arange(4), np.mean(stay_probs,0), yerr=np.std(stay_probs,0))
    plt.xticks(np.arange(4), ['CR', 'RR', 'CN', 'RN'])
    plt.ylim(ymin=0)
    plt.ylabel('Stay probability')
    
def _get_CSTO(ep, return_inds=False):
    '''Get the choices, second step states, transitions and outcomes for one episode as
    a set of binary vectors.'''
    choices, sec_steps, outcomes, ch_inds, ss_inds, oc_inds = [],[],[],[],[],[]
    first_choice_ind = np.where(ep.states==ts.choice)[0][0]
    last_outcome_ind = np.where(np.isin(ep.states, [ts.reward_A, ts.reward_B, ts.initiate]) & (ep.actions == ts.initiate))[0][-1]
    for i, (s,a) in enumerate(zip(ep.states, ep.actions)):
        if i < first_choice_ind:
            continue
        elif s == ts.choice and a in (ts.choose_A, ts.choose_B):
            if a == ts.choose_A:
                choices.append(1)
            else:
                choices.append(0)
            ch_inds.append(i)
        elif s == ts.sec_step_A and a == ts.sec_step_A: 
            sec_steps.append(1)
            ss_inds.append(i)
        elif s == ts.sec_step_B and a == ts.sec_step_B: 
            sec_steps.append(0)
            ss_inds.append(i)
        elif s in (ts.reward_A, ts.reward_B, ts.initiate) and a == ts.initiate: 
            if s == ts.initiate:
                outcomes.append(0)
            else:
                outcomes.append(1)
            oc_inds.append(i)
            if i == last_outcome_ind:
                break
    choices = np.array(choices, bool)
    sec_steps = np.array(sec_steps, bool)
    transitions = choices == sec_steps
    outcomes = np.array(outcomes, bool)
    if return_inds:
        return choices, sec_steps, transitions, outcomes, np.array(ch_inds), np.array(ss_inds), np.array(oc_inds)
    else:
        return choices, sec_steps, transitions, outcomes
    
# Second step value analysis --------------------------------------------------    
    
def sec_step_value_analysis(episode_buffer, Str_model, PFC_model, task, last_n=10, return_means=False, fig_no=3):
    '''For a single simulation run, pot the change in value of second-step states from one trial to the next
    as a function of the trial outcome and whether the second-step state on the next trial is the same or
    different.  Evaluates both second-step states on each trial by generating the apropriate input to the 
    striatum model. '''
    value_updates = np.zeros([last_n, 4])
    for i,ep in enumerate(episode_buffer[-last_n:]):
        _, sec_steps, _, outcomes, _, ss_inds, _ = _get_CSTO(ep, return_inds=True)
        # Generate PFC activity that would have occured for each second step state on each trial.
        Get_pfc_state = keras.Model(inputs=PFC_model.input, # Model variant used to get state of RNN layer.
                                     outputs=PFC_model.get_layer('rnn').output)
        ss_pfc_inputs = ep.pfc_inputs[ss_inds]
        ss_pfc_inputs[:,-1,:task.n_states] = 0
        ss_pfc_inputs[:,-1,ts.sec_step_A]  = 1
        ss_pfc_states_A = Get_pfc_state(ss_pfc_inputs) # PFC activity if second-step reached was A.
        ss_pfc_inputs[:,-1,:task.n_states] = 0
        ss_pfc_inputs[:,-1,ts.sec_step_B]  = 1
        ss_pfc_states_B = Get_pfc_state(ss_pfc_inputs) # PFC activity if second-step reached was B.
        # Compute values of both second step states on each trial.
        _, V_ssA = Str_model([one_hot(np.ones(len(ss_inds), int)*ts.sec_step_A, task.n_states), ss_pfc_states_A])
        _, V_ssB = Str_model([one_hot(np.ones(len(ss_inds), int)*ts.sec_step_B, task.n_states), ss_pfc_states_B])
        # Compute value changes as a function of trial outcome and same/different second-step state.
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
        if return_means:
            return(np.mean(value_updates,0))
        else:
            _sec_step_value_analysis_plot(value_updates, fig_no)
            
def sec_step_value_analysis_exp(experiment_data, fig_no=2):
    '''Second step value analysis for an experiment comprising mulitple simulation runs.'''
    value_updates = np.zeros([len(experiment_data),4])
    for i, rd in enumerate(experiment_data):
        value_updates[i,:] = sec_step_value_analysis(rd.episode_buffer, rd.Str_model, rd.PFC_model, rd.task, return_means=True)
    _sec_step_value_analysis_plot(value_updates, fig_no)
    
def _sec_step_value_analysis_plot(value_updates, fig_no):
    plt.figure(fig_no, clear=True)
    plt.subplot(1,2,1)
    plt.errorbar(np.arange(4), np.mean(value_updates,0), yerr = np.std(value_updates,0), ls='none', marker='o')
    plt.xticks(np.arange(4), ['Rew same', 'Rew diff', 'Non same', 'Non diff'])
    plt.axhline(0,c='k',lw=0.5)
    plt.xlim(-0.5,3.5)
    plt.ylabel('Change in state value')
    plt.xlabel('Trial outcome')
    plt.subplot(1,2,2)
    plt.errorbar(np.arange(2), np.mean(value_updates[:,:2]-value_updates[:,2:],0), 
                 yerr = np.std(value_updates[:,:2]-value_updates[:,2:],0), ls='none', marker='o')
    plt.xticks(np.arange(2), ['same', 'diff'])
    plt.axhline(0,c='k',lw=0.5)
    plt.ylabel('Effect of reward on state value')
    plt.xlabel('State')
    plt.xlim(-0.5,1.5)
    plt.tight_layout()
    
# Plot PFC choice state activity ----------------------------------------------
    
def plot_PFC_choice_state_activity(episode_buffer, task, last_n=3, fig_no=4):
    '''Plot the first principle component of variation in the PFC activity in the choice state across trials'''
    ch_state_PFC_features = []
    for ep in episode_buffer[-last_n:]:
        ch_state_PFC_features.append(ep.pfc_states[ep.states==ts.choice])
    ch_state_PFC_features = np.vstack(ch_state_PFC_features) 
    PC1 = PCA(n_components=1).fit(ch_state_PFC_features).transform(ch_state_PFC_features)
    plt.figure(fig_no, clear=True)
    plt.plot(PC1)
    plt.ylabel('First principle component of\nchoice state PFC activity')
    plt.xlabel('Trials')
    plt.xlim(0,len(PC1))
    
# Simulate optogenetic manipulation.
    
def sim_opto(episode_buffer, Str_model, task, last_n=10):
    stay_probs_cs = np.zeros([last_n, 4])
    stay_probs_os = np.zeros([last_n, 4])
    for i, ep in enumerate(episode_buffer[-last_n:]):
        choices, sec_steps, transitions, outcomes, ch_inds, ss_inds, oc_inds = _get_CSTO(ep, return_inds=True)
        # Get indices of different choice states defined by A/B choice and subsequent common/rare transition.
        A_comm_ch_inds = ch_inds[ choices &  transitions]
        A_rare_ch_inds = ch_inds[ choices & ~transitions]
        B_comm_ch_inds = ch_inds[~choices &  transitions]
        B_rare_ch_inds = ch_inds[~choices & ~transitions]

        ocp = partial(_opto_choice_probs, Str_model, ch_inds, ep, task)
        
        # Caclulate choice stim stay probabilities.
        A_comm_nons_cp_0, A_comm_stim_cp_0 = ocp(A_comm_ch_inds[0:-1:2])
        A_comm_nons_cp_1, A_comm_stim_cp_1 = ocp(A_comm_ch_inds[1:-1:2])
        A_rare_nons_cp_0, A_rare_stim_cp_0 = ocp(A_rare_ch_inds[0:-1:2])
        A_rare_nons_cp_1, A_rare_stim_cp_1 = ocp(A_rare_ch_inds[1:-1:2])
        B_comm_nons_cp_0, B_comm_stim_cp_0 = ocp(B_comm_ch_inds[0:-1:2])
        B_comm_nons_cp_1, B_comm_stim_cp_1 = ocp(B_comm_ch_inds[1:-1:2])
        B_rare_nons_cp_0, B_rare_stim_cp_0 = ocp(B_rare_ch_inds[0:-1:2])
        B_rare_nons_cp_1, B_rare_stim_cp_1 = ocp(B_rare_ch_inds[1:-1:2])
    
        sp_comm_stim = np.mean(np.concatenate([
             np.vstack([A_comm_stim_cp_0, A_comm_stim_cp_1])[:,ts.choose_A],
             np.vstack([B_comm_stim_cp_0, B_comm_stim_cp_1])[:,ts.choose_B]]))
        sp_comm_nons = np.mean(np.concatenate([
             np.vstack([A_comm_nons_cp_0, A_comm_nons_cp_1])[:,ts.choose_A],
             np.vstack([B_comm_nons_cp_0, B_comm_nons_cp_1])[:,ts.choose_B]]))
        sp_rare_stim = np.mean(np.concatenate([
             np.vstack([A_rare_stim_cp_0, A_rare_stim_cp_1])[:,ts.choose_A],
             np.vstack([B_rare_stim_cp_0, B_rare_stim_cp_1])[:,ts.choose_B]]))
        sp_rare_nons = np.mean(np.concatenate([
             np.vstack([A_rare_nons_cp_0, A_rare_nons_cp_1])[:,ts.choose_A],
             np.vstack([B_rare_nons_cp_0, B_rare_nons_cp_1])[:,ts.choose_B]]))   
        
        stay_probs_cs[i,:] = [sp_comm_stim, sp_rare_stim, sp_comm_nons, sp_rare_nons]
        
        # Caclulate outcome stim stay probabilities.
        A_comm_nons_cp_0, A_comm_stim_cp_0 = ocp(A_comm_ch_inds[0:-1:2]+1)
        A_comm_nons_cp_1, A_comm_stim_cp_1 = ocp(A_comm_ch_inds[1:-1:2]+1)
        A_rare_nons_cp_0, A_rare_stim_cp_0 = ocp(A_rare_ch_inds[0:-1:2]+1)
        A_rare_nons_cp_1, A_rare_stim_cp_1 = ocp(A_rare_ch_inds[1:-1:2]+1)
        B_comm_nons_cp_0, B_comm_stim_cp_0 = ocp(B_comm_ch_inds[0:-1:2]+1)
        B_comm_nons_cp_1, B_comm_stim_cp_1 = ocp(B_comm_ch_inds[1:-1:2]+1)
        B_rare_nons_cp_0, B_rare_stim_cp_0 = ocp(B_rare_ch_inds[0:-1:2]+1)
        B_rare_nons_cp_1, B_rare_stim_cp_1 = ocp(B_rare_ch_inds[1:-1:2]+1)
    
        sp_comm_stim = np.mean(np.concatenate([
             np.vstack([A_comm_stim_cp_0, A_comm_stim_cp_1])[:,ts.choose_A],
             np.vstack([B_comm_stim_cp_0, B_comm_stim_cp_1])[:,ts.choose_B]]))
        sp_comm_nons = np.mean(np.concatenate([
             np.vstack([A_comm_nons_cp_0, A_comm_nons_cp_1])[:,ts.choose_A],
             np.vstack([B_comm_nons_cp_0, B_comm_nons_cp_1])[:,ts.choose_B]]))
        sp_rare_stim = np.mean(np.concatenate([
             np.vstack([A_rare_stim_cp_0, A_rare_stim_cp_1])[:,ts.choose_A],
             np.vstack([B_rare_stim_cp_0, B_rare_stim_cp_1])[:,ts.choose_B]]))
        sp_rare_nons = np.mean(np.concatenate([
             np.vstack([A_rare_nons_cp_0, A_rare_nons_cp_1])[:,ts.choose_A],
             np.vstack([B_rare_nons_cp_0, B_rare_nons_cp_1])[:,ts.choose_B]]))   
        
        stay_probs_os[i,:] = [sp_comm_stim, sp_rare_stim, sp_comm_nons, sp_rare_nons]
    
    plt.figure(1, clear=True)
    plt.subplot(1,2,1)
    plt.bar(np.arange(4), np.mean(stay_probs_cs,0), yerr=np.std(stay_probs_cs,0))
    plt.xticks(np.arange(4),['Com. stim', 'Rare stim', 'Com. nons', 'Rare nons'], rotation=-45)
    plt.ylim(0,1)
    plt.subplot(1,2,2)
    plt.bar(np.arange(4), np.mean(stay_probs_os,0), yerr=np.std(stay_probs_os,0))
    plt.xticks(np.arange(4),['Com. stim', 'Rare stim', 'Com. nons', 'Rare nons'], rotation=-45)  
    plt.ylim(0,1)
    
def _opto_choice_probs(Str_model, all_ch_inds, episode, task, stim_inds):
    '''Evalute how training the striatum model using gradients due to artificially evoked opto RPE
    affects choice probabilities on the subseqeunt choice states.''' 
    states, rewards, actions, pfc_input, pfc_states, values, pred_states, n_trials = episode
    orig_weights = Str_model.get_weights()
    
    # Update model weights.
    opto_stim = np.zeros(len(states))
    opto_stim[stim_inds] = 1
    
    SGD_optimiser = keras.optimizers.SGD(learning_rate=0.01)
    
    with tf.GradientTape() as tape:
        # Critic loss.
        choice_probs_g, values_g = Str_model([one_hot(states, task.n_states), tf.concat(pfc_states,0)]) # Gradient of these is tracked wrt Str_model weights.
        critic_loss = sse_loss(values_g, tf.stop_gradient(values_g)+opto_stim)*(100/np.sum(opto_stim))
        # Actor loss.
        log_chosen_probs = tf.math.log(tf.gather_nd(choice_probs_g, [[i,a] for i,a in enumerate(actions)]))
        actor_loss = tf.reduce_sum(-log_chosen_probs*opto_stim)*(100/np.sum(opto_stim))
        # Apply gradients
        grads = tape.gradient(actor_loss+critic_loss, Str_model.trainable_variables)

    SGD_optimiser.apply_gradients(zip(grads, Str_model.trainable_variables))
    choice_probs_stim, _ = Str_model([one_hot(states, task.n_states), tf.concat(pfc_states,0)])
    
    # Evaluate choice probabilities on the next choice states following stim states.
    eval_inds = all_ch_inds[np.searchsorted(all_ch_inds, stim_inds+1)] # Where to evaluate choice probabilities.
    assert np.intersect1d(stim_inds, eval_inds).size == 0, 'Overlaping stim and evaluation trials'
    
    ch_probs_nons  = tf.gather(choice_probs_g, eval_inds).numpy()
    ch_probs_stim = tf.gather(choice_probs_stim, eval_inds).numpy()
    
    Str_model.set_weights(orig_weights) # Reset model weights.
    
    return ch_probs_nons, ch_probs_stim
    
 