#%% Imports

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import Two_step_task as ts
import analysis as an

one_hot = keras.utils.to_categorical
sse_loss = keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)

#%% Parameters.

# Simulation parameters.
n_episodes = 1000
episode_len = 100  # Episode length in trials.
gamma = 0.9        # Discount rate
max_step_per_episode = 600
entropy_loss_weight = 0.03

#Task params.
good_prob = 0.9
block_len = [20,21]

# Model parameters.
n_back = 30
n_rnn = 16
learning_rate = 0.05

#%% Instantiate task.

task = ts.Two_step(good_prob=good_prob, block_len=block_len, punish_invalid=False)

#%% Meta RL model.

model_input = layers.Input(shape=(n_back, task.n_states)) # PFC inputs are 1 hot encoding of states and actions.
rnn = layers.GRU(n_rnn, unroll=True, name='rnn')(model_input) # Recurrent layer.
actor = layers.Dense(task.n_actions, activation="softmax")(rnn)
critic = layers.Dense(1)(rnn)
model = keras.Model(inputs=model_input, outputs=[actor, critic])
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

Get_rnn_state = keras.Model(inputs=model.input, # Model variant used to get state of RNN layer.
                             outputs=model.get_layer('rnn').output)

rnn_input_buffer = np.zeros([n_back, task.n_states])

def update_rnn_input(s):
    '''Update the inputs to the PFC network given the state,reward and action.'''
    global rnn_input_buffer
    rnn_input_buffer = np.roll(rnn_input_buffer,-1, axis=0)
    rnn_input_buffer[-1,:] = 0
    rnn_input_buffer[-1,s] = 1               # One hot encoding of old state.

#%% Environment loop.

def store_trial_data(s, r, a, V):
    'Store state, reward and subseqent action, PFC input buffer, PFC activity, value.'
    global states, rewards, actions, rnn_input, rnn_states, values
    states.append(s)
    rewards.append(r)
    actions.append(a)
    rnn_input.append(rnn_input_buffer)
    rnn_states.append(Get_rnn_state(rnn_input_buffer[np.newaxis,:,:]))
    values.append(V)
    
# Run model.

s = task.reset() # Get initial state as integer.
r = 0
update_rnn_input(s)

episode_buffer = []

for e in range(n_episodes):
    
    step_n = 0
    start_trial = task.trial_n
    
    # Episode history variables
    states  = []       # int
    rewards = []       # float
    actions = []       # int
    rnn_input = []     # (1,30,n_states+n_actions)
    rnn_states = []    # (1,n_pfc)
    values = []        # float
    
    while True:
        step_n += 1
        
        choice_probs, V = model(rnn_input_buffer[np.newaxis,:,:])
        
        # Choose action.
        a = np.random.choice(task.n_actions, p=np.squeeze(choice_probs))
        
        # Store history.
        store_trial_data(s, r, a, V)
        
        # Get next state and reward.
        s, r = task.step(a)
        
        # Update the rnn inputs.
        update_rnn_input(s)
      
        n_trials = task.trial_n - start_trial
        if n_trials == episode_len or step_n >= max_step_per_episode and s == 0:
            break # End of episode.  
            
    # Store episode data.
    
    episode_buffer.append((states, rewards, actions, rnn_input, rnn_states, values, n_trials))
    
    # Update weights
    
    returns = np.zeros([len(rewards),1], dtype='float32')
    returns[-1] = V
    for i in range(1, len(returns)):
        returns[-i-1] = rewards[-i] + gamma*returns[-i]
             
    advantages = (returns - np.vstack(values)).squeeze()
          
    with tf.GradientTape() as tape: # Calculate critic gradients.
        # Critic loss.
        choice_probs_g, values_g = model(np.array(rnn_input)) # Gradient of these is tracked wrt Str_model weights.
        critic_loss = sse_loss(values_g, returns)
        # Actor loss.
        log_chosen_probs = tf.math.log(tf.gather_nd(choice_probs_g, [[i,a] for i,a in enumerate(actions)]))
        entropy = -tf.reduce_sum(choice_probs_g*tf.math.log(choice_probs_g),1)
        actor_loss = tf.reduce_sum(-log_chosen_probs*advantages-entropy*entropy_loss_weight)
        # Apply gradients
        grads = tape.gradient(actor_loss+critic_loss, model.trainable_variables)

    optimizer.apply_gradients(zip(grads, model.trainable_variables))
         
        
    print(f'Episode: {e} Steps: {step_n} Trials: {n_trials} Rew. per tr.: {np.sum(rewards)/n_trials :.2f}')
    
    #if e % 10 == 9:
    #    an.plot_performance(episode_buffer, task)

# Plotting at end of run.
        
# an.make_plots(episode_buffer, task, Str_model)
    
#%% Save / load data.

data_dir = 'C:\\Users\\Thomas\\Dropbox\\Work\\Two-step DA photometry\\RNN model\\data\\experiment_metaRL'

def save_data(data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    with open(os.path.join(data_dir, 'episodes.pkl'), 'wb') as f: 
        pickle.dump(episode_buffer, f)
    model.save(os.path.join(data_dir, 'rnn_model'))
    
def load_data(data_dir):
    with open(os.path.join(data_dir, 'episodes.pkl'), 'rb') as f: 
        episode_buffer = pickle.load(f)
    model = keras.models.load_model(os.path.join(data_dir, 'rnn_model'))
    return episode_buffer, model
    