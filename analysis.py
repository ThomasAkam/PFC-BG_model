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
from statsmodels.stats.proportion import proportions_ztest
from sklearn.decomposition import PCA
from tensorflow import keras
from collections import namedtuple

import two_step_task as ts

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

def ave_reward_rate(run_data, last_n=10, return_p_value=False):
    '''Compute the average reward rate over the last_n episodes of a run and 
    return either reward rate or P value for difference from 0.5.'''
    n_rewards, n_trials = (0,0)
    for ep in run_data.episode_buffer[-last_n:]:
        n_rewards += np.sum(ep.rewards)
        n_trials  += ep.n_trials
    if return_p_value:
        return proportions_ztest(n_rewards,n_trials,0.5,'larger')[1]
    return n_rewards/n_trials

#%% Plot experiment
    
def plot_experiment(experiment_data, last_n=10, save_dir=None):
    '''Run set of analyses on data from an experiment comrising multiple simulation runs.
    If save_dir is specified then the plots, statistics and simulation paramters are 
    saved to this directory.'''
    if save_dir and not os.path.exists(save_dir):
        os.mkdir(save_dir)
        with open(os.path.join(save_dir,'params.txt'), 'w') as f:
            json.dump(experiment_data[0].params, f, indent=4)    
    stay_probability_analysis(experiment_data, last_n, fig_no=1, save_dir=save_dir)
    second_step_value_update_analysis(experiment_data, last_n, fig_no=2, save_dir=save_dir)
    plot_PFC_choice_state_activity(experiment_data, fig_no=3, save_dir=save_dir)
    opto_stim_analysis(experiment_data, fig_no=4, save_dir=save_dir)
    
def _statsprint(s, save_dir):
    '''Print text s to file 'stats.txt' in save_dir and screen.'''
    print(s)
    if save_dir:
        with open(os.path.join(save_dir,'stats.txt'), 'a') as f:
            f.write(str(s)+'\n')
    
#%% Plot performance ----------------------------------------------------------

def plot_performance(episode_buffer, task, fig_no=1):
    ''''Plot the evolution of the number of steps needed to complete each trial and the 
    number of rewards per trial across training for one simulation run.'''
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
    
#%% Stay probability analysis -------------------------------------------------

def stay_probability_analysis(experiment_data, last_n=10, fig_no=1, save_dir=None):
    '''Plot stay probabilities and run logistic regression for stats.'''
    # Compute stay probs and logistic regression fit for each simulation run.
    stay_probs = np.zeros([len(experiment_data),4])
    fits = []
    for i, run_data in enumerate(experiment_data):
        stay_probs[i,:], fit = _get_stay_probs(run_data.episode_buffer, last_n)
        if np.max(stay_probs[i,:]) < 1: # Logistic regression will not converge if any stay probs are 1.
            fits.append(fit)
    # Plotting
    fig = plt.figure(fig_no, figsize=[2.8,2.4], clear=True)
    plt.bar(np.arange(4), np.mean(stay_probs,0), yerr=sem(stay_probs,0), ecolor='r')
    sns.stripplot(data=stay_probs, color='grey', size=2)
    plt.xticks(np.arange(4), ['CR', 'RR', 'CN', 'RN'])
    plt.ylim(ymin=0)
    plt.ylabel('Stay probability')
    plt.tight_layout()
    if save_dir: fig.savefig(os.path.join(save_dir,'stay_probabilities.pdf'))
    # Stats
    _statsprint('Logistic regression analysis of stay probabilities:', save_dir)
    _ttest_v_0(pd.concat([fit.to_frame().T for fit in fits]), save_dir)
    #return fits
    
def _get_stay_probs(episode_buffer, last_n):
    '''Compute stay probabilities and logistic regression fit for the last_n
    episodes of one simulation run.'''
    # Compute stay probabilities.
    stay_probs = []
    dfs = []
    for ep in episode_buffer[-last_n:]:
        choices, sec_steps, transitions, outcomes = _get_CSTO(ep)
        stays = choices[1:] == choices[:-1]
        sp_comm_rew = np.mean(stays[ transitions[:-1] &  outcomes[:-1]])
        sp_rare_rew = np.mean(stays[~transitions[:-1] &  outcomes[:-1]])
        sp_comm_non = np.mean(stays[ transitions[:-1] & ~outcomes[:-1]])
        sp_rare_non = np.mean(stays[~transitions[:-1] & ~outcomes[:-1]])
        stay_probs.append(np.array([sp_comm_rew, sp_rare_rew, sp_comm_non, sp_rare_non]))
        dfs.append(pd.DataFrame({'transition': transitions[:-1]-0.5,
                                    'outcome'   : outcomes[:-1]-0.5,
                                    'stay'      : stays.astype(int)}))
    fit = smf.logit(formula='stay ~ transition*outcome', data=pd.concat(dfs)).fit(disp=False).params
    return np.nanmean(stay_probs,0), fit
    
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
    
#%% Second step value update analysis -----------------------------------------
    
def second_step_value_update_analysis(experiment_data, last_n=10, fig_no=2, save_dir=None):
    '''Plot the change in value of second-step states from one trial to the next as a function of the
    trial outcome and whether the second-step state on the next trial is the same or different.  
    Evaluates both second-step states on each trial by generating the apropriate input to the 
    striatum model. '''
    value_updates = np.zeros([len(experiment_data),4])
    for i, run_data in enumerate(experiment_data):
        value_updates[i,:] = _get_value_updates(run_data, last_n)
    # Plotting
    fig = plt.figure(fig_no, figsize=[6.2,2.4], clear=True)
    plt.subplot(1,2,1)
    sns.stripplot(data=value_updates, color='grey', size=2)
    plt.errorbar(np.arange(4), np.mean(value_updates,0), yerr = sem(value_updates,0), ls='none', color='r', elinewidth=2)
    plt.xticks(np.arange(4), ['Rew same', 'Rew diff', 'Non same', 'Non diff'])
    plt.axhline(0,c='k',lw=0.5)
    plt.xlim(-0.5,3.5)
    plt.ylabel('Change in state value')
    plt.xlabel('Trial outcome')
    plt.subplot(1,2,2)
    reward_effect = value_updates[:,:2]-value_updates[:,2:]
    sns.stripplot(data=reward_effect, color='grey', size=2)
    plt.errorbar(np.arange(2), np.mean(reward_effect,0), yerr=sem(reward_effect,0), ls='none', color='r', elinewidth=2)
    plt.xticks(np.arange(2), ['same', 'diff'])
    plt.axhline(0,c='k',lw=0.5)
    plt.ylabel('Effect of reward on state value')
    plt.xlabel('State')
    plt.xlim(-0.5,1.5)
    plt.tight_layout()
    if save_dir: fig.savefig(os.path.join(save_dir,'sec_step_value_updates.pdf'))
    # Stats
    _statsprint('\nEffect out outcome on same/different second-step state value:', save_dir)
    _ttest_v_0(pd.DataFrame({'Same':reward_effect[:,0],'diff':reward_effect[:,1]}), save_dir)

def _get_value_updates(run_data, last_n=10):
    '''Compute the change in second-step values from one trial to the next for one simulation run.'''
    'params', 'episode_buffer', 'PFC_model', 'Str_model', 'task'
    _, episode_buffer, PFC_model, Str_model, task =  run_data
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
    return(np.mean(value_updates,0))

#%% Plot PFC choice state activity --------------------------------------------

def plot_PFC_choice_state_activity(experiment_data, fig_no=3, save_dir=None):
    '''Plot the projection of PFC activity in the choice state across trials onto its first
    principal component'''
    n_runs = len(experiment_data)
    fig = plt.figure(fig_no, figsize=[4,11], clear=True)
    for i, run_data in enumerate(experiment_data):
        PC1, task_rew_state, choices = _get_PFC_activity(run_data.episode_buffer, run_data.task)  
        plt.subplot(n_runs,1,i+1)
        plt.plot(PC1)
        plt.plot(task_rew_state*0.75+2.25,'g')
        plt.plot(choices*0.75-3, '.r', ms=2)
        plt.xlim(0,300)
        plt.yticks([-1,0,1])
        if i == n_runs//2:
            plt.ylabel('First principle component of\nchoice state PFC activity')
    plt.xlabel('Trials')
    plt.tight_layout()
    if save_dir: fig.savefig(os.path.join(save_dir,'PFC_choice_state_activity.pdf'))
    
def _get_PFC_activity(episode_buffer, task, last_n=3):
    '''Get the projection of PFC activtiy in the choice state across trials onto
    its first principal component.  Also returns the trial-by-trial state of the
    task's reward probabilities and the models choices.'''
    ch_state_PFC_activity = []
    task_rew_state = []
    choices = []
    for ep in episode_buffer[-last_n:]:
        ch_state_PFC_activity.append(ep.pfc_states[ep.states==ts.choice])
        task_rew_state.append(ep.task_rew_states[ep.states==ts.choice])
        choices.append(ep.actions[ep.states==ts.choice] == ts.choose_A)
    ch_state_PFC_activity = np.vstack(ch_state_PFC_activity) 
    PC1 = PCA(n_components=1).fit(ch_state_PFC_activity).transform(ch_state_PFC_activity)
    task_rew_state = np.hstack(task_rew_state) 
    choices = np.hstack(choices)
    return PC1, task_rew_state, choices



#%% Simulate optogenetic manipulation. ----------------------------------------

def opto_stim_analysis(experiment_data, last_n=10, stim_strength=1, stim_prob=0.25, fig_no=4, save_dir=None):
    '''Evaluate effect of simulated optogenetic stimulation of stay proabilities for
    the last_n episoces of each run in experiment.''' 
    # Simulate choice and outcome time stimulation for each experiment run and analyse effects with linear regression.
    choice_stim_fits  = []
    outcome_stim_fits = []
    print('\nSimulating opto stim for run: ', end='')
    for r,run_data in enumerate(experiment_data,start=1):
        print(f'{r} ', end='')
        # Simulate opto stim effects.
        episode_cs_dfs = []
        episode_os_dfs = []
        for ep in run_data.episode_buffer[-last_n:]:
            episode_cs_dfs.append(_opto_stay_probs(run_data, ep, 'choice_time' , stim_strength, stim_prob))
            episode_os_dfs.append(_opto_stay_probs(run_data, ep, 'outcome_time', stim_strength, stim_prob))
        choice_stim_df  = pd.concat(episode_cs_dfs)
        outcome_stim_df = pd.concat(episode_os_dfs)

        # Regression analysis of stim effects.
        choice_stim_fits.append( smf.ols(formula='logit_stay_prob ~ transition*outcome*stim', data=choice_stim_df ).fit().params)
        outcome_stim_fits.append(smf.ols(formula='logit_stay_prob ~ transition*outcome*stim', data=outcome_stim_df).fit().params)

    choice_stim_fits  = pd.concat([fit.to_frame().T for fit in choice_stim_fits ])*2 # Multiply by 2 to convert to log odds (as predictors are +1,-1).
    outcome_stim_fits = pd.concat([fit.to_frame().T for fit in outcome_stim_fits])*2
    
    # Plotting
    fig = plt.figure(fig_no, figsize=[3.7,3.2], clear=True)
    ax1 = plt.subplot(2,1,1)
    _plot_opto_fits(choice_stim_fits, ax1, xticklabels=False)
    ax2 = plt.subplot(2,1,2, sharex=ax1, sharey=ax1)
    _plot_opto_fits(outcome_stim_fits, ax2, xticklabels=True)
    plt.xlim(3.5,7.5)
    plt.ylim(-0.5,1)
    plt.tight_layout()
    fig.text(0.05,0.05, f'Stim_strength:{stim_strength}', fontsize=9)
    if save_dir: fig.savefig(os.path.join(save_dir,'opto_stim_analysis.pdf'))
    # Stats
    _statsprint('\n\nOpto stim analysis.\n\nChoice time stim:', save_dir)
    _ttest_v_0(choice_stim_fits, save_dir)
    _statsprint('\nOutcome time stim:', save_dir)
    _ttest_v_0(outcome_stim_fits, save_dir)

def _opto_stay_probs(run_data, ep, stim_type, stim_strength, stim_prob):
    '''Evalute how training the striatum model using gradients due to opto RPE
    on individual trials affects stay probability for one episode (ep).''' 
    params, _, _, Str_model, task = run_data
    choices, sec_steps, transitions, outcomes, ch_inds, ss_inds, oc_inds = _get_CSTO(ep, return_inds=True)
    orig_weights = Str_model.get_weights()
    
    # Compute A/B choice probabilities for each trial in the absence of stimulation.
    action_probs = Str_model([one_hot(ep.states, task.n_states), tf.concat(ep.pfc_states,0)])[0].numpy()
    choice_probs = np.stack([action_probs[ch_inds,ts.choose_B],action_probs[ch_inds,ts.choose_A]])
    
    # Compute A/B choice probabilities following opto stim for randomly selected set of trials. 
    stim_trials = np.random.rand(choice_probs.shape[1])<stim_prob
    
    SGD_optimiser = keras.optimizers.SGD(learning_rate=params['str_learning_rate'])
    for t, stim_trial in enumerate(stim_trials[:-1]): # Loop over trials.
        if not stim_trial:
            continue
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
        choice_probs[:,t+1] = (nt_action_probs[0,ts.choose_B],nt_action_probs[0,ts.choose_A])
        # Reset model weights.
        Str_model.set_weights(orig_weights)
        
    # Normalise choice probs to sum to 1 (as non-choice actions have non-zero prob).
    choice_probs = choice_probs/np.sum(choice_probs,0)
    
    # Compute stay probabilities
    stay_probs = choice_probs[choices[:-1].astype(int), np.arange(1,len(choices))]
    
    # Make dataframe with predictors sum-to-zero coded (-1,1).
    df = pd.DataFrame({'outcome'   :2*(outcomes   [:-1]-0.5),
                       'transition':2*(transitions[:-1]-0.5),
                       'stim'      :2*(stim_trials[:-1]-0.5),
                       'stay_prob' : stay_probs,
                       'logit_stay_prob':logit(stay_probs)})
    return df

def _plot_opto_fits(fits_df, ax, xticklabels=True):
    '''Plot the fit of a linear regression analysis of opto-stim simulation.'''
    x = np.arange(fits_df.shape[1])
    ax.axhline(0,c='k', linewidth=0.5)
    sns.swarmplot(data=fits_df, color='grey', size=2, axes=ax)
    ax.errorbar(x,fits_df.mean(), fits_df.sem(),linestyle='none', linewidth=2, color='r')
    if xticklabels:
        ax.set_xticklabels(fits_df.columns.to_list(),rotation=-45, ha='left', rotation_mode='anchor')
        ax.set_ylabel('Δ stay probability (log odds)')
    else:
        plt.setp(ax.get_xticklabels(), visible=False)

def _ttest_v_0(df, save_dir):
    '''T-test whether the distribution of values in each column of data frame is significantly different from 0.'''
    ttest = ttest_1samp(df,0)
    sigstars = ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                for p in ttest.pvalue]    
    _statsprint(pd.DataFrame({'mean':df.mean(),'std':df.std(),'t':ttest.statistic,
                'pvalue':ttest.pvalue, 'sig.': sigstars}).round(4), save_dir)
    