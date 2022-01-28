#%% Imports

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import namedtuple

import Two_step_task as ts
import analysis as an

one_hot = keras.utils.to_categorical
sse_loss = keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)

Episode = namedtuple('Episode', ['states', 'rewards', 'actions', 'pfc_input', 'pfc_states', 'values', 'pred_states', 'n_trials'])

#%% Parameters.

# Simulation parameters.
n_episodes = 500
episode_len = 100  # Episode length in trials.
gamma = 0.9        # Discount rate
max_step_per_episode = 600
entropy_loss_weight = 0.05

#Task params.
good_prob = 0.8
block_len = [20,21]

# PFC model parameters.
n_back = 30
n_pfc = 12
pfc_learning_rate = 0.01

# Striatum model parameters.
str_learning_rate = 0.05

#%% Instantiate task.

task = ts.Two_step(good_prob=good_prob, block_len=block_len, punish_invalid=False)

#%% PFC model.

pfcm_input = layers.Input(shape=(n_back, task.n_states)) 
rnn = layers.GRU(n_pfc, unroll=True, name='rnn')(pfcm_input) # Recurrent layer.
rew_pred = layers.Dense(task.n_states, activation='softmax', name='state_pred')(rnn) # Output layer predicts reward by state.
PFC_model = keras.Model(inputs=pfcm_input, outputs=rew_pred)
pfc_optimizer = keras.optimizers.Adam(learning_rate=pfc_learning_rate)
PFC_model.compile(loss="mean_squared_error", optimizer=pfc_optimizer)
Get_pfc_state = keras.Model(inputs=PFC_model.input, # Model variant used to get state of RNN layer.
                             outputs=PFC_model.get_layer('rnn').output)

pfc_input_buffer = np.zeros([n_back, task.n_states])

def update_pfc_input(s,r):
    '''Update the inputs to the PFC network given the state and reward.'''
    global pfc_input_buffer
    pfc_input_buffer = np.roll(pfc_input_buffer,-1, axis=0)
    pfc_input_buffer[-1,:] = 0
    pfc_input_buffer[-1,s] = r               # One hot encoding of old state.

#%% Basal ganglia model

obs_state = layers.Input(shape=(task.n_states,)) # Observable state features.
pfc_state = layers.Input(shape=(n_pfc,))         # PFC activity features.
combined_features = keras.layers.Concatenate(axis=1)([obs_state, pfc_state])
relu = layers.Dense(task.n_states, activation="relu")(combined_features)
#common = keras.layers.add([obs_state, relu]) # Add residual connections from observable features to output of relu layer.
common = keras.layers.Concatenate(axis=1)([obs_state, relu])

actor = layers.Dense(task.n_actions, activation="softmax")(common)
critic = layers.Dense(1)(common)

Str_model = keras.Model(inputs=[obs_state, pfc_state], outputs=[actor, critic])
str_optimizer = keras.optimizers.Adam(learning_rate=str_learning_rate)

#%% Environment loop.

def store_trial_data(s, r, a, pfc_a, V):
    'Store state, reward and subseqent action, PFC input buffer, PFC activity, value.'
    global states, rewards, actions, pfc_input, pfc_states, values
    states.append(s)
    rewards.append(r)
    actions.append(a)
    pfc_input.append(pfc_input_buffer)
    pfc_states.append(pfc_a)
    values.append(V)
    
# Run model.

s = task.reset() # Get initial state as integer.
r = 0
pfc_a = Get_pfc_state(pfc_input_buffer[np.newaxis,:,:])

episode_buffer = []

for e in range(n_episodes):
    
    step_n = 0
    start_trial = task.trial_n
    
    # Episode history variables
    states  = []       # int
    rewards = []       # float
    actions = []       # int
    pfc_input = []     # (1,30,n_states+n_actions)
    pfc_states = []    # (1,n_pfc)
    values = []        # float
    
    while True:
        step_n += 1
        
        choice_probs, V = Str_model([one_hot(s, task.n_states)[None,:], pfc_a])
        
        # Choose action.
        a = np.random.choice(task.n_actions, p=np.squeeze(choice_probs))
        
        # Store history.
        store_trial_data(s, r, a, pfc_a, V)
        
        # Get next state and reward.
        s, r = task.step(a)
        
        # Update the PFC networks inputs.
        update_pfc_input(s,r)
        
        # Get new pfc state.
        pfc_a = Get_pfc_state(pfc_input_buffer[np.newaxis,:,:])
        
        n_trials = task.trial_n - start_trial
        if n_trials == episode_len or step_n >= max_step_per_episode and s == 0:
            break # End of episode.  
            
    # Store episode data.
    
    pred_states = np.argmax(PFC_model.predict(np.array(pfc_input)),1) # Used only for analysis.
    episode_buffer.append(Episode(np.array(states), np.array(rewards), np.array(actions), np.array(pfc_input), 
                           np.vstack(pfc_states), np.vstack(values), None, n_trials))
    
    # Update striatum weights
    
    returns = np.zeros([step_n,1], dtype='float32')
    returns[-1] = V
    for i in range(1, step_n):
        returns[-i-1] = rewards[-i] + gamma*returns[-i]
             
    advantages = (returns - np.vstack(values)).squeeze()
          
    with tf.GradientTape() as tape: # Calculate gradients
        # Critic loss.
        choice_probs_g, values_g = Str_model([one_hot(states, task.n_states), np.vstack(pfc_states)]) # Gradient of these is tracked wrt Str_model weights.
        critic_loss = sse_loss(values_g, returns)
        # Actor loss.
        log_chosen_probs = tf.math.log(tf.gather_nd(choice_probs_g, [[i,a] for i,a in enumerate(actions)]))
        entropy = -tf.reduce_sum(choice_probs_g*tf.math.log(choice_probs_g),1)
        actor_loss = tf.reduce_sum(-log_chosen_probs*advantages-entropy*entropy_loss_weight)
        # Apply gradients
        grads = tape.gradient(actor_loss+critic_loss, Str_model.trainable_variables)

    str_optimizer.apply_gradients(zip(grads, Str_model.trainable_variables))
         
    # Update PFC 
    
    y  = np.zeros([step_n-1, task.n_states])
    y[np.arange(0,step_n-1),states[:-1]] = rewards[1:]
    tl = PFC_model.train_on_batch(np.array(pfc_input[:-1]),y)
        
    print(f'Episode: {e} Steps: {step_n} Trials: {n_trials} '
          f' Rew. per tr.: {np.sum(rewards)/n_trials :.2f} PFC tr. loss: {tl :.3f}')
    
    if e % 10 == 9: an.plot_performance(episode_buffer, task)

# Plotting at end of run.
        
# an.make_plots(episode_buffer, task, Str_model)
    
#%% Save / load data.

data_dir = 'C:\\Users\\Thomas\\Dropbox\\Work\\Two-step DA photometry\\RNN model\\data\\experiment_rew_pred'

def save_data(data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    with open(os.path.join(data_dir, 'episodes.pkl'), 'wb') as f: 
        pickle.dump(episode_buffer, f)
    PFC_model.save(os.path.join(data_dir, 'PFC_model'))
    Str_model.save(os.path.join(data_dir, 'Str_model'))
    
def load_data(data_dir):
    with open(os.path.join(data_dir, 'episodes.pkl'), 'rb') as f: 
        episode_buffer = pickle.load(f)
    PFC_model = keras.models.load_model(os.path.join(data_dir, 'PFC_model'))
    Str_model = keras.models.load_model(os.path.join(data_dir, 'Str_model'))
    return episode_buffer, PFC_model, Str_model
    