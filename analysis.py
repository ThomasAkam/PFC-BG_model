#%%  Imports 

import os
import json
import pickle
import numpy as np
import pandas as pd
import pylab as plt
import seaborn as sns
import tensorflow as tf
import statsmodels.formula.api as smf
from scipy.special import logit
from scipy.stats import ttest_1samp, sem
from sklearn.decomposition import PCA
from tensorflow import keras
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

def load_experiment(exp_dir, good_only=True):
    '''Load data from an experiment comprising multiple simulation runs, if good_only
    is True then only runs for which the reward rate in the last 10 episodes is
   significantly higher than 0.5 are returned.'''
    run_dirs = os.listdir(exp_dir)
    experiment_data = [load_run(os.path.join(exp_dir, run_dir)) for run_dir in run_dirs]
    if good_only:
        experiment_data = [run_data for run_data in experiment_data 
                           if ave_reward_rate(run_data, return_p_value=True) < 0.05]
    return experiment_data

#%% Analysis functions

def ave_reward_rate(run_data, last_n=10, return_p_value=False):
    episode_reward_rates = []
    for ep in run_data.episode_buffer[-last_n:]:
        episode_reward_rates.append(np.sum(ep.rewards)/ep.n_trials)
    if return_p_value:
        return  ttest_1samp(episode_reward_rates,0.5).pvalue
    else:
        return np.mean(episode_reward_rates) #
        

# Multi-analysis functions

def make_plots(episode_buffer, task, Str_model, PFC_model):
    '''Run all plotting functions for a single simulation run.'''
    plot_performance(episode_buffer, task)
    stay_probability_analysis(episode_buffer)
    sec_step_value_analysis(episode_buffer, Str_model, PFC_model, task)
    plot_PFC_choice_state_activity(episode_buffer, task)
    

def plot_run(run_data):
    '''Make plots from a Run_data tuple'''
    make_plots(run_data.episode_buffer, run_data.task, run_data.Str_model, run_data.PFC_model)
    
def plot_experiment(experiment_data):
    '''Load data for an experiment comprising multiple simulation runs and 
    make plots showing cross run standard deviation.'''
    stay_probability_analysis_exp(experiment_data, fig_no=1)
    sec_step_value_analysis_exp(experiment_data, fig_no=2)
    plot_PFC_choice_state_activity_exp(experiment_data, fig_no=3)
    opto_stim_analysis(experiment_data, fig_no=4)

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
    plt.figure(fig_no, figsize=[2.8,2.4], clear=True)
    plt.bar(np.arange(4), np.mean(stay_probs,0), yerr=sem(stay_probs,0), ecolor='r')
    sns.stripplot(data=stay_probs, color='k', size=2)
    plt.xticks(np.arange(4), ['CR', 'RR', 'CN', 'RN'])
    plt.ylim(ymin=0)
    plt.ylabel('Stay probability')
    plt.tight_layout()
    
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
        # Generate PFC activity that would have occured had each second step state been reached on each trial.
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
            
def sec_step_value_analysis_exp(experiment_data, fig_no=3):
    '''Second step value analysis for an experiment comprising mulitple simulation runs.'''
    value_updates = np.zeros([len(experiment_data),4])
    for i, rd in enumerate(experiment_data):
        value_updates[i,:] = sec_step_value_analysis(rd.episode_buffer, rd.Str_model, rd.PFC_model, rd.task, return_means=True)
    _sec_step_value_analysis_plot(value_updates, fig_no)
    
def _sec_step_value_analysis_plot(value_updates, fig_no):
    plt.figure(fig_no, figsize=[6.2,2.4], clear=True)
    plt.subplot(1,2,1)
    sns.stripplot(data=value_updates, color='k', size=3)
    plt.errorbar(np.arange(4), np.mean(value_updates,0), yerr = sem(value_updates,0), ls='none', color='r', elinewidth=2)
    plt.xticks(np.arange(4), ['Rew same', 'Rew diff', 'Non same', 'Non diff'])
    plt.axhline(0,c='k',lw=0.5)
    plt.xlim(-0.5,3.5)
    plt.ylabel('Change in state value')
    plt.xlabel('Trial outcome')
    plt.subplot(1,2,2)
    reward_effect = value_updates[:,:2]-value_updates[:,2:]
    sns.stripplot(data=reward_effect, color='k', size=3)
    plt.errorbar(np.arange(2), np.mean(reward_effect,0), yerr=sem(reward_effect,0), ls='none', color='r', elinewidth=2)
    plt.xticks(np.arange(2), ['same', 'diff'])
    plt.axhline(0,c='k',lw=0.5)
    plt.ylabel('Effect of reward on state value')
    plt.xlabel('State')
    plt.xlim(-0.5,1.5)
    plt.tight_layout()
    
# Plot PFC choice state activity ----------------------------------------------
    
def plot_PFC_choice_state_activity(episode_buffer, task, last_n=3, fig_no=4):
    '''Plot the first principle component of variation in the PFC activity in the choice state across trials'''
    ch_state_PFC_activity = []
    for ep in episode_buffer[-last_n:]:
        ch_state_PFC_activity.append(ep.pfc_states[ep.states==ts.choice])
    ch_state_PFC_activity = np.vstack(ch_state_PFC_activity) 
    PC1 = PCA(n_components=1).fit(ch_state_PFC_activity).transform(ch_state_PFC_activity)
    if not fig_no: return PC1
    plt.figure(fig_no, clear=True)
    plt.plot(PC1)
    plt.ylabel('First principle component of\nchoice state PFC activity')
    plt.xlabel('Trials')
    plt.xlim(0,len(PC1))
    
def plot_PFC_choice_state_activity_exp(experiment_data, fig_no=2):
    '''Plot the first principle component of variation in the PFC activity in the choice state across trials,
    seperately for each run in an experiment.'''
    n_runs = len(experiment_data)
    plt.figure(fig_no, clear=True)
    for i, run_data in enumerate(experiment_data):
        PC1 = plot_PFC_choice_state_activity(run_data.episode_buffer, run_data.task, fig_no=False)
        plt.subplot(n_runs,1,i+1)
        plt.plot(PC1)
        plt.xlim(0,len(PC1))
        if i == n_runs//2:
            plt.ylabel('First principle component of\nchoice state PFC activity')
    plt.xlabel('Trials')
            
  
# Simulate optogenetic manipulation. ------------------------------------------

def opto_stim_analysis(experiment_data, stim_strength=1, last_n=10, fig_no=1):
    '''Evaluate effect of simulated optogenetic stimulation of stay proabilities for
    the last_n episoces of each run in experiment.''' 
    # Simulate choice and outcome time stimulation for each experiment run and analyse effects with linear regression.
    choice_stim_fits  = []
    outcome_stim_fits = []
    print('Simulating opto stim for run: ', end='')
    for r,run_data in enumerate(experiment_data,start=1):
        print(f'{r} ', end='')
        # Simulate opto stim effects.
        episode_cs_dfs = []
        episode_os_dfs = []
        for ep in run_data.episode_buffer[-last_n:]:
            episode_cs_dfs.append(_opto_stay_probs(run_data, ep, 'choice_time' , stim_strength))
            episode_os_dfs.append(_opto_stay_probs(run_data, ep, 'outcome_time', stim_strength))
        choice_stim_df  = pd.concat(episode_cs_dfs)
        outcome_stim_df = pd.concat(episode_os_dfs)

        # Regression analysis of stim effects.
        choice_stim_fits.append( smf.ols(formula='logit_stay_prob ~ transition*outcome*stim', data=choice_stim_df ).fit().params)
        outcome_stim_fits.append(smf.ols(formula='logit_stay_prob ~ transition*outcome*stim', data=outcome_stim_df).fit().params)
    # Plot stim effects
    choice_stim_fits  = pd.concat([fit.to_frame().T for fit in choice_stim_fits])
    outcome_stim_fits = pd.concat([fit.to_frame().T for fit in outcome_stim_fits])

    plt.figure(fig_no,clear=True)
    ax1 = plt.subplot(2,1,1)
    print('\nChoice time stim:')
    _plot_opto_fits(choice_stim_fits, ax1, xticklabels=False)
    ax2 = plt.subplot(2,1,2, sharex=ax1, sharey=ax1)
    print('\nChoice time stim:')
    _plot_opto_fits(outcome_stim_fits, ax2, xticklabels=True)
    plt.tight_layout()

 
def _opto_stay_probs(run_data, ep, stim_type, stim_strength):
    '''Evalute how training the striatum model using gradients due to opto RPE
    on individual trials affects stay probability for one episode (ep).''' 
    
    Str_model, task, params = (run_data.Str_model, run_data.task, run_data.params)
    choices, sec_steps, transitions, outcomes, ch_inds, ss_inds, oc_inds = _get_CSTO(ep, return_inds=True)
    orig_weights = Str_model.get_weights()
    
    # Compute A/B choice probabilities for each trial in the absence of stimulation.
    action_probs = Str_model([one_hot(ep.states, task.n_states), tf.concat(ep.pfc_states,0)])[0].numpy()
    choice_probs_nons = np.stack([action_probs[ch_inds,ts.choose_B],action_probs[ch_inds,ts.choose_A]])

    # Compute A/B choice probabilities following opto stim on individual trials.    
    choice_probs_stim = np.ones(choice_probs_nons.shape)
    SGD_optimiser = keras.optimizers.SGD(learning_rate=params['str_learning_rate'])
    for t in range(choice_probs_nons.shape[1]-1): # Loop over trials.
        if stim_type == 'choice_time':
            i = ch_inds[t] # Index of current trial choice in episode.
        elif stim_type == 'outcome_time':
            i = ss_inds[t] # Index of current trial second-step in episode.
        # Compute gradients due to opto stim.
        with tf.GradientTape() as tape:
                # Critic loss.
                tr_action_probs, tr_value = Str_model(
                    [one_hot(ep.states[i], task.n_states)[np.newaxis,:], ep.pfc_states[i][np.newaxis,:]]) # Action probs and values for single trial.
                critic_loss = -2*stim_strength*tr_value
                # Actor loss.
                log_chosen_prob = tf.math.log(tr_action_probs[0, ep.actions[i]])
                entropy = -tf.reduce_sum(tr_action_probs*tf.math.log(tr_action_probs))
                actor_loss = -log_chosen_prob*stim_strength-entropy*params['entropy_loss_weight']
                # Compute gradients.
                grads = tape.gradient(actor_loss+critic_loss, Str_model.trainable_variables)
        # Update model weights.
        SGD_optimiser.apply_gradients(zip(grads, Str_model.trainable_variables))
        # Compute next trial choice probs.
        j = ch_inds[t+1] # Index in episode of next trial choice.
        nt_action_probs, _ = Str_model([one_hot(ep.states[j], task.n_states)[np.newaxis,:], ep.pfc_states[j][np.newaxis,:]])
        choice_probs_stim[:,t+1] = (nt_action_probs[0,ts.choose_B],nt_action_probs[0,ts.choose_A])
        # Reset model weights.
        Str_model.set_weights(orig_weights)
        
    # Normalise choice probs to sum to 1 (as non-choice actions have non-zero prob).
    choice_probs_nons = choice_probs_nons/np.sum(choice_probs_nons,0)
    choice_probs_stim = choice_probs_stim/np.sum(choice_probs_stim,0)
    
    # Compute stay probabilities
    stay_probs_nons = choice_probs_nons[choices[:-1].astype(int),np.arange(1,len(choices))]
    stay_probs_stim = choice_probs_stim[choices[:-1].astype(int),np.arange(1,len(choices))]
    
    # Make dataframe with predictors sum-to-zero coded (-1,1).
    df_nons = pd.DataFrame({'outcome':2*(outcomes[:-1]-0.5),'transition':2*(transitions[:-1]-0.5),
                            'stim':-1,'stay_prob': stay_probs_nons,'logit_stay_prob':logit(stay_probs_nons)})
    df_stim = pd.DataFrame({'outcome':2*(outcomes[:-1]-0.5),'transition':2*(transitions[:-1]-0.5),
                            'stim':1, 'stay_prob': stay_probs_stim,'logit_stay_prob':logit(stay_probs_stim)})
    
    include_trials = df_nons['stay_prob'].between(0.01,0.99) # Exclude trials with stay probs very close to 0 or 1 to avoid floor/ceiling effects.
    df_nons = df_nons.loc[include_trials,:]
    df_stim = df_stim.loc[include_trials,:]
    
    return pd.concat([df_nons, df_stim])

def _plot_opto_fits(fits_df, ax, xticklabels):
    '''Plot the fit of a linear regression analysis of opto-stim simulation.'''
    fits_df = fits_df*2 # Convert to log odds (as predictors are +1,-1).
    x = np.arange(fits_df.shape[1])
    ax.axhline(0,c='k', linewidth=0.5)
    sns.stripplot(data=fits_df, color='k', size=2, axes=ax)
    ax.errorbar(x,fits_df.mean(), fits_df.sem(),linestyle='none', linewidth=2, color='r')
    if xticklabels:
        ax.set_xticklabels(fits_df.columns.to_list(),rotation=-45, ha='left', rotation_mode='anchor')
        ax.set_ylabel('Δ stay probability (log odds)')
    else:
        plt.setp(ax.get_xticklabels(), visible=False)
    
    ttest = ttest_1samp(fits_df,0)
    stats_df = pd.DataFrame({'predictor': fits_df.columns.to_list(),'t':ttest.statistic,'pvalue':ttest.pvalue})
    print(stats_df)


    
