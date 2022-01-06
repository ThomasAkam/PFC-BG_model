#%% Imports

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pylab as plt
from sklearn.decomposition import PCA

import Two_step_task as ts

#%% Parameters.

# Simulation parameters.
n_episodes = 1000
n_epochs = 3
episode_len = 100  # Episode length in trials.
gamma = 0.9        # Discount rate
max_step_per_episode = 600
entropy_loss_weight = 0.01

#Task params.
good_prob = 0.99
block_len = [20,21]

# PFC model parameters.
n_back = 4
n_pfc = 12
pfc_learning_rate = 0.01

# Striatum model parameters.
n_str = 12
str_learning_rate = 0.05

#%% Instantiate task.

task = ts.Two_step(good_prob=good_prob, block_len=block_len, punish_invalid=False)

#%% PFC model.

pfc_inputs = layers.Input(shape=(n_back, task.n_states+task.n_actions+1)) # Inputs are 1 hot encoding of alternating states and actions.
lstm = layers.LSTM(n_pfc, unroll=True, name='lstm', return_state=True)(pfc_inputs) # Recurrent layer.
state_pred = layers.Dense(task.n_states+1, activation='softmax', name='state_pred')(lstm[0]) # Output layer predicts next state
PFC_model = keras.Model(inputs=pfc_inputs, outputs=state_pred)
pfc_optimizer = keras.optimizers.Adam(learning_rate=pfc_learning_rate)
PFC_model.compile(loss="categorical_crossentropy", optimizer=pfc_optimizer)
# Model variant used to get state of LSTM layer.
PFC_state_view = keras.Model(inputs=PFC_model.input,
                             outputs=PFC_model.get_layer('lstm').output[2])

pfc_input_buffer = np.zeros([n_back, task.n_states+task.n_actions+1])

def update_pfc_inputs(s,r,a):
    '''Update the inputs to the PFC network given the state,reward and action.'''
    global pfc_input_buffer
    pfc_input_buffer = np.roll(pfc_input_buffer,-1, axis=0)
    pfc_input_buffer[-1,:] = 0
    pfc_input_buffer[-1,s] = 1               # One hot encoding of old state.
    pfc_input_buffer[-1,a+task.n_states] = 1 # One hot encoding of action.
    pfc_input_buffer[-1,-1] = r

def get_state_features(s):
    '''Get the state feature vectors combining observable and PFC features.'''
    Sf = np.zeros(task.n_states+n_pfc) # State feature vector combining observable and PFC features.
    Sf[s] = 1 # Observable task state feature.
    # Sf[task.n_states:] = PFC_state_view(pfc_input_buffer[np.newaxis,:,:]) # PFC features
    return Sf

def pretrain_model(n_steps=40000, fig_no=1):
    # Generate data.
    task.reset()
    states = np.zeros(n_steps, int)
    actions = np.zeros(n_steps, int)
    block_info = np.zeros([n_steps,2],int)
    for i in range(n_steps-1):
        actions[i]  = task.sample_appropriate_action(states[i])
        state, reward, info = task.step(actions[i])
        states[i+1] = state
        block_info[i+1,:] = info
    actions[-1]  = task.sample_appropriate_action(states[-1])
    states_actions = np.stack([states, actions+task.n_states]).T.reshape(1,-1).squeeze()
    state_action_seqs = []
    next_states = []
    for i in range(0, len(states_actions) - n_back, 2):
        state_action_seqs.append(states_actions[i:i+n_back])
        next_states.append(states_actions[i+n_back])
    print("Number of sequences:", len(state_action_seqs))
    x = np.stack([keras.utils.to_categorical(sas, task.n_actions+task.n_states) for sas in state_action_seqs])
    y = keras.utils.to_categorical(np.array(next_states), task.n_states)
    # Fit model.
    PFC_model.fit(x[:-1000], y[:-1000], batch_size=100)
    # Plot first PC of lstm state.
    lstm_state = PFC_state_view.predict(x[-1000:])
    choice_inds = np.where(states[-1000:] == ts.choice)[0]
    ch_state = lstm_state[choice_inds,:]
    pca = PCA(n_components=1).fit(ch_state)
    ch_state_pc1 = pca.transform(ch_state)
    plt.figure(fig_no, clear=True)
    plt.plot(ch_state_pc1 , label='lstm state PC1')
    return pca


#%% Basal ganglia model

str_inputs = layers.Input(shape=(task.n_states+n_pfc,))
common = layers.Dense(n_str, activation="relu")(str_inputs)
actor = layers.Dense(task.n_actions, activation="softmax")(common)
critic = layers.Dense(1)(common)

Str_model = keras.Model(inputs=str_inputs, outputs=[actor, critic])
str_optimizer = keras.optimizers.Adam(learning_rate=str_learning_rate)

#%% Environment loop.

sse_loss = keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)

def store_history(s, Sf, r, a):
    'Store state, reward and subseqent action for current timestep.'
    global states, state_f, rewards, actions
    states.append(s)
    state_f.append(Sf)
    rewards.append(r)
    actions.append(a)
    pfc_input.append(pfc_input_buffer)

s = task.reset() # Get initial state as integer.
r = 0
Sf = get_state_features(s)

for e in range(n_episodes):
    
    step_n = 0
    start_trial = task.trial_n
    
    # Episode history variables
    states  = [] # int
    state_f = [] # State feature vectors.
    rewards = []
    actions = [] # int
    pfc_input = []
    
    while True:
        step_n += 1
        
        choice_probs, V = Str_model(tf.convert_to_tensor(Sf[np.newaxis,:]))
        
        # Choose action.
        a = np.random.choice(task.n_actions, p=np.squeeze(choice_probs))
        
        # Store history.
        store_history(s, Sf, r, a)
        
        # Update the PFC networks inputs.
        update_pfc_inputs(s,r,a)
        
        # Get next state and reward.
        s, r = task.step(a)
        
        # Get new state vector conbining observable and PFC features.
        Sf = get_state_features(s)
        
        episode_trials = task.trial_n - start_trial
        if episode_trials == episode_len or step_n >= max_step_per_episode and s == 0:
            break # End of episode.        

    # Update Striatum weights.
    
    returns = np.zeros([len(rewards),1], dtype='float32')
    returns[-1] = r
    for i in range(1, len(returns)):
        returns[-i-1] = rewards[-i] + gamma*returns[-i]
         
    for i in range(n_epochs):
        with tf.GradientTape() as tape:
            choice_probs, values = Str_model(tf.convert_to_tensor(state_f))
            advantages = returns - values
            # Critic loss.
            critic_loss = sse_loss(values, returns)
            # Actor loss
            actor_losses = []
            for i,a in enumerate(actions):
                log_chosen_prob = tf.math.log(choice_probs[i,a])
                entropy = -tf.reduce_sum(choice_probs[i,:]*tf.math.log(choice_probs[i,:]))
                actor_losses.append(-log_chosen_prob*advantages[i]-entropy*entropy_loss_weight)
            actor_loss = tf.reduce_sum(actor_losses)
            # Apply combined loss.    
            grads = tape.gradient(critic_loss+actor_loss, Str_model.trainable_variables)
            str_optimizer.apply_gradients(zip(grads, Str_model.trainable_variables))
            
        # Update PFC weights
        y = np.hstack([keras.utils.to_categorical(states, task.n_states), 
                       np.array(rewards, 'float32')[:,np.newaxis]])
        tl = PFC_model.train_on_batch(np.array(pfc_input),y)
    
    print(f'Episode: {e} Steps: {step_n} Trials: {episode_trials} '
          f' Rew. per tr.: {np.sum(rewards)/episode_trials :.2f} PFC tr. loss: {tl :.3f}')
    
    
    
    
    
    
    
    