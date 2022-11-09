#%% Imports

import os
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import namedtuple

import two_step_task as ts
import analysis as an

one_hot = keras.utils.to_categorical
sse_loss = keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)

Episode = namedtuple('Episode', ['states', 'rewards', 'actions', 'pfc_inputs', 'pfc_states', 'pred_states','task_rew_states', 'n_trials'])

#%% Parameters.

default_params = {
    # Simulation params.
    'n_episodes'  : 500,
    'episode_len' : 100,  # Episode length in trials.
    'max_step_per_episode' : 600,
    'gamma' : 0.9,        # Discount rate

    #Task params.
    'good_prob' : 0.8,
    'block_len' : [20,40],

    # PFC model params.
    'n_back': 30, # Length of history provided as input.
    'n_pfc' : 16,  # Number of PFC units
    'pfc_learning_rate' : 0.01,
    'pred_rewarded_only' : False, # If True PFC input (and prediction target) is rewarded states only.

    # Striatum model params.
    'n_str' : 10, # Number of striatum units
    'str_learning_rate' : 0.05,
    'entropy_loss_weight' : 0.05}

#%% Run simulation.

def run_simulation(save_dir=None, pm=default_params):
    # Initialise random seed to ensure runs using multiprocessing use different random numbers.
    np.random.seed(int.from_bytes(os.urandom(4), 'little'))

    #Instantiate task.
    task = ts.Two_step(good_prob=pm['good_prob'], block_len=pm['block_len'])
    
    # PFC model.
    
    if pm['pred_rewarded_only']: # PFC input is one-hot encoding of observable state on rewarded trias, 0 vector on non-rewarded.
        pfc_input_layer = layers.Input(shape=(pm['n_back'], task.n_states))
        pfc_input_buffer = np.zeros([pm['n_back'], task.n_states], bool)
    else: # PFC input is 1 hot encoding of observable state and previous action.
        pfc_input_layer = layers.Input(shape=(pm['n_back'], task.n_states+task.n_actions)) 
        pfc_input_buffer = np.zeros([pm['n_back'], task.n_states+task.n_actions], bool)
    rnn = layers.GRU(pm['n_pfc'], unroll=True, name='rnn')(pfc_input_layer) # Recurrent layer.
    state_pred = layers.Dense(task.n_states, activation='softmax', name='state_pred')(rnn) # Output layer predicts next state
    PFC_model = keras.Model(inputs=pfc_input_layer, outputs=state_pred)
    pfc_optimizer = keras.optimizers.Adam(learning_rate=pm['pfc_learning_rate'])
    PFC_model.compile(loss='mean_squared_error', optimizer=pfc_optimizer)
    Get_pfc_state = keras.Model(inputs=PFC_model.input, # Model variant used to get state of RNN layer.
                                 outputs=PFC_model.get_layer('rnn').output)

    def update_pfc_input(a,s,r):
        '''Update the inputs to the PFC network given the action, subsequent state and reward.'''
        pfc_input_buffer[:-1,:] = pfc_input_buffer[1:,:]
        pfc_input_buffer[-1,:] = 0
        if pm['pred_rewarded_only']:
            pfc_input_buffer[-1,s] = r # One hot encoding on state on rewarded timesteps, 0 vector on non-rewarded.
        else:   
            pfc_input_buffer[-1,s] = 1               # One hot encoding of state.
            pfc_input_buffer[-1,a+task.n_states] = 1 # One hot encoding of action.
        
    def get_masked_PFC_inputs(pfc_inputs):
        '''Return array of PFC input history with the most recent state masked, 
        used for training as the most recent state is the prediction target.'''
        masked_pfc_inputs = np.array(pfc_inputs)
        masked_pfc_inputs[:,-1,:task.n_states] = 0
        return masked_pfc_inputs
    
    # Striatum model
    
    obs_state = layers.Input(shape=(task.n_states,)) # Observable state features.
    pfc_state = layers.Input(shape=(pm['n_pfc'],))         # PFC activity features.
    combined_features = keras.layers.Concatenate(axis=1)([obs_state, pfc_state])
    relu = layers.Dense(pm['n_str'], activation="relu")(combined_features)
    
    actor = layers.Dense(task.n_actions, activation="softmax")(relu)
    critic = layers.Dense(1)(relu)
    
    Str_model = keras.Model(inputs=[obs_state, pfc_state], outputs=[actor, critic])
    str_optimizer = keras.optimizers.Adam(learning_rate=pm['str_learning_rate'])
    
    # Environment loop.
    
    def store_trial_data(s, r, a, pfc_s, V):
        'Store state, reward and subseqent action, PFC input buffer, PFC activity, value.'
        states.append(s)
        rewards.append(r)
        actions.append(a)
        pfc_inputs.append(pfc_input_buffer.copy())
        pfc_states.append(pfc_s)
        values.append(V)
        task_rew_states.append(task.A_good)
        
    # Run model.
    
    s = task.reset() # Get initial state as integer.
    r = 0
    pfc_s = Get_pfc_state(pfc_input_buffer[np.newaxis,:,:])
    
    episode_buffer = []
    
    for e in range(pm['n_episodes']):
        
        step_n = 0
        start_trial = task.trial_n
        
        # Episode history variables
        states  = []       # int
        rewards = []       # float
        actions = []       # int
        pfc_inputs = []    # (1,30,n_states+n_actions)
        pfc_states = []    # (1,n_pfc)
        values = []        # float
        task_rew_states = [] # bool
           
        while True:
            step_n += 1
            
            # Choose action.
            action_probs, V = Str_model([one_hot(s, task.n_states)[None,:], pfc_s])
            a = np.random.choice(task.n_actions, p=np.squeeze(action_probs))
            
            # Store history.
            store_trial_data(s, r, a, pfc_s, V)
            
            # Get next state and reward.
            s, r = task.step(a)
            
            # Get new pfc state.
            update_pfc_input(a,s,r)
            # pfc_s = Get_pfc_state(pfc_input_buffer[np.newaxis,:,:])                # Get the PFC activity, slow but does not give error message.
            pfc_s = Get_pfc_state.predict_on_batch(pfc_input_buffer[np.newaxis,:,:]) # Get the PFC activity, fast and returns same result but gives an error message.
    
            n_trials = task.trial_n - start_trial
            if n_trials == pm['episode_len'] or step_n >= pm['max_step_per_episode'] and s == 0:
                break # End of episode.  
                
        # Store episode data.
        
        pred_states = np.argmax(PFC_model(get_masked_PFC_inputs(pfc_inputs)),1) # Used only for analysis.
        episode_buffer.append(Episode(np.array(states), np.array(rewards), np.array(actions), np.array(pfc_inputs),
                               np.vstack(pfc_states), np.array(pred_states), np.array(task_rew_states), n_trials))
        
        # Update striatum weights using advantage actor critic (A2C), Mnih et al. PMLR 48:1928-1937, 2016
        
        returns = np.zeros([len(rewards),1], dtype='float32')
        returns[-1] = V
        for i in range(1, len(returns)):
            returns[-i-1] = rewards[-i] + pm['gamma']*returns[-i]
                 
        advantages = (returns - np.vstack(values)).squeeze()
              
        with tf.GradientTape() as tape: # Calculate gradients
            # Critic loss.
            action_probs_g, values_g = Str_model([one_hot(states, task.n_states), np.vstack(pfc_states)]) # Gradient of these is tracked wrt Str_model weights.
            critic_loss = sse_loss(values_g, returns)
            # Actor loss.
            log_chosen_probs = tf.math.log(tf.gather_nd(action_probs_g, [[i,a] for i,a in enumerate(actions)]))
            entropy = -tf.reduce_sum(action_probs_g*tf.math.log(action_probs_g),1)
            actor_loss = tf.reduce_sum(-log_chosen_probs*advantages-entropy*pm['entropy_loss_weight'])
            # Apply gradients
            grads = tape.gradient(actor_loss+critic_loss, Str_model.trainable_variables)
    
        str_optimizer.apply_gradients(zip(grads, Str_model.trainable_variables))
             
        # Update PFC weights.

        if pm['pred_rewarded_only']: # PFC is trained to predict its current input given previous input.
            tl = PFC_model.train_on_batch(np.array(pfc_inputs[:-1]), one_hot(states[1:], task.n_states)*np.array(rewards)[1:,np.newaxis])
        else: # PFC is trained to predict the current state given previous action and state.
            tl = PFC_model.train_on_batch(get_masked_PFC_inputs(pfc_inputs), one_hot(states, task.n_states))
            
        print(f'Episode: {e} Steps: {step_n} Trials: {n_trials} '
              f' Rew. per tr.: {np.sum(rewards)/n_trials :.2f} PFC tr. loss: {tl :.3f}')
        
        if e % 10 == 9: an.plot_performance(episode_buffer, task)
        
    # Save data.    
    
    if save_dir:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        with open(os.path.join(save_dir,'params.json'), 'w') as fp:
            json.dump(pm, fp, indent=4)
        with open(os.path.join(save_dir, 'episodes.pkl'), 'wb') as fe: 
            pickle.dump(episode_buffer, fe)
        PFC_model.save(os.path.join(save_dir, 'PFC_model'))
        Str_model.save(os.path.join(save_dir, 'Str_model'))