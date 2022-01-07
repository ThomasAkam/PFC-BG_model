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
episode_len = 100  # Episode length in trials.
gamma = 0.9        # Discount rate
max_step_per_episode = 600
entropy_loss_weight = 0.01
n_replay = 1

#Task params.
good_prob = 0.9
block_len = [20,21]

# PFC model parameters.
n_back = 30
n_pfc = 16
pfc_learning_rate = 0.005

# Striatum model parameters.
n_str = 12
str_learning_rate = 0.05

#%% Instantiate task.

task = ts.Two_step(good_prob=good_prob, block_len=block_len, punish_invalid=False)

#%% PFC model.

pfc_inputs = layers.Input(shape=(n_back, task.n_states+task.n_actions)) # Inputs are 1 hot encoding of alternating states and actions.
rnn = layers.GRU(n_pfc, unroll=True, name='rnn')(pfc_inputs) # Recurrent layer.
state_pred = layers.Dense(task.n_states, activation='softmax', name='state_pred')(rnn) # Output layer predicts next state
PFC_model = keras.Model(inputs=pfc_inputs, outputs=state_pred)
pfc_optimizer = keras.optimizers.Adam(learning_rate=pfc_learning_rate)
PFC_model.compile(loss="categorical_crossentropy", optimizer=pfc_optimizer)
# Model variant used to get state of RNN layer.
PFC_state_view = keras.Model(inputs=PFC_model.input,
                             outputs=PFC_model.get_layer('rnn').output)

pfc_input_buffer = np.zeros([n_back, task.n_states+task.n_actions])

def update_pfc_inputs(s,a):
    '''Update the inputs to the PFC network given the state,reward and action.'''
    global pfc_input_buffer
    pfc_input_buffer = np.roll(pfc_input_buffer,-1, axis=0)
    pfc_input_buffer[-1,:] = 0
    pfc_input_buffer[-1,s] = 1               # One hot encoding of old state.
    pfc_input_buffer[-1,a+task.n_states] = 1 # One hot encoding of action.

def get_state_features(s):
    '''Get the state feature vectors combining observable and PFC features.'''
    Sf = np.zeros(task.n_states+n_pfc) # State feature vector combining observable and PFC features.
    Sf[s] = 1 # Observable task state feature.
    Spfc = PFC_state_view(pfc_input_buffer[np.newaxis,:,:])
    Sf[task.n_states:] = (Spfc-np.mean(Spfc))/(np.std(Spfc)+0.01) # PFC features
    return Sf

def pretrain_model(n_steps=100000, batch=100, fig_no=1):
    # Generate data.
    task.reset()
    states = np.zeros(n_steps, int)
    actions = np.zeros(n_steps, int)
    s = task.reset()
    for i in range(n_steps):
        states[i] = s
        actions[i]  = task.sample_appropriate_action(s)
        s, r = task.step(actions[i])
    states_v = keras.utils.to_categorical(states, task.n_states)
    actions_v = keras.utils.to_categorical(actions, task.n_actions)
    states_actions = np.hstack([states_v, actions_v])
    x = [] # State-action sequences
    y = [] # Next states
    for i in range(0, len(states_actions) - n_back):
        x.append(states_actions[i:i+n_back,:])
        y.append(states_v[i+n_back,:])
    print("Number of sequences:", len(x))
    x = np.stack(x)
    y = np.stack(y)
    # Fit model holding out last 1000 timesteps for evaluation.
    PFC_model.fit(x[:-1000], y[:-1000], batch_size=batch)
    # Plot reward probabilities on test data.
    y_pred = PFC_model.predict(x[-1000:])
    A_out_inds = np.where(states[-1000:] == ts.sec_step_A)[0][:-1]+1
    B_out_inds = np.where(states[-1000:] == ts.sec_step_B)[0][:-1]+1
    A_reward_probs = y_pred[:,ts.reward_A]
    B_reward_probs = y_pred[:,ts.reward_B]
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(A_out_inds, A_reward_probs[A_out_inds])
    plt.plot(B_out_inds, B_reward_probs[B_out_inds])
    plt.ylabel('Estimated reward probs')
    # Look at features of rnn layer
    rnn_state = PFC_state_view.predict(x[-1000:]) 
    choice_inds = np.where(states[-1000:] == ts.choice)[0]
    ch_state = rnn_state[choice_inds,:]
    pca = PCA(n_components=1).fit(ch_state)
    ch_state_pc1 = pca.transform(ch_state)
    plt.subplot(2,1,2)
    plt.plot(ch_state_pc1 , label='rnn state PC1')
    plt.ylabel('RNN state PC1')
    plt.show()
    tl = PFC_model.train_on_batch(x[-episode_len:],y[-episode_len:])
    print(f'Training loss: {tl :.3f}')
    
# pretrain_model()

#%% Basal ganglia model

str_inputs = layers.Input(shape=(task.n_states+n_pfc,))
common = layers.Dense(n_str, activation="relu")(str_inputs)
actor = layers.Dense(task.n_actions, activation="softmax")(common)
critic = layers.Dense(1)(common)

Str_model = keras.Model(inputs=str_inputs, outputs=[actor, critic])
str_optimizer = keras.optimizers.Adam(learning_rate=str_learning_rate)

#%% Environment loop.

sse_loss = keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)

# Helper functions.

def store_trial_data(s, Sf, r, a):
    'Store state, reward and subseqent action for current timestep.'
    global states, state_f, rewards, actions
    states.append(s)
    state_f.append(Sf)
    rewards.append(r)
    actions.append(a)
    pfc_input.append(pfc_input_buffer)
    
# Run model.

s = task.reset() # Get initial state as integer.
r = 0
Sf = get_state_features(s)

episode_buffer = []

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
        store_trial_data(s, Sf, r, a)
        
        # Update the PFC networks inputs.
        update_pfc_inputs(s,a)
        
        # Get next state and reward.
        s, r = task.step(a)
        
        # Get new state vector conbining observable and PFC features.
        Sf = get_state_features(s)
        
        n_trials = task.trial_n - start_trial
        if n_trials == episode_len or step_n >= max_step_per_episode and s == 0:
            break # End of episode.  
            
    episode_buffer.append((states, state_f, rewards, actions, pfc_input, n_trials))

    # Train networks.
    
    for episode in episode_buffer[-n_replay:]:
        states, state_f, rewards, actions, pfc_input, n_trials = episode

        # Update Striatum weights.
        
        returns = np.zeros([len(rewards),1], dtype='float32')
        #returns[-1] = r
        for i in range(1, len(returns)):
            returns[-i-1] = rewards[-i] + gamma*returns[-i]
                
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
        y = keras.utils.to_categorical(states, task.n_states)
        tl = PFC_model.train_on_batch(np.array(pfc_input),y)
    
    print(f'Episode: {e} Steps: {step_n} Trials: {n_trials} '
          f' Rew. per tr.: {np.sum(rewards)/n_trials :.2f} PFC tr. loss: {tl :.3f}')
    
#%% Analysis

def plot_performance():
    steps_per_trial = []
    rewards_per_trial = []
    for episode in episode_buffer:
        states, state_f, rewards, actions, pfc_input = episode
        n_trials = 0
        for s,a in zip(states, actions):
            n_trials += s == ts.choice and a in (ts.choose_A, ts.choose_B)
        steps_per_trial.append(len(states)/n_trials)
        rewards_per_trial.append(sum(rewards)/n_trials)
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(steps_per_trial)
    plt.ylabel('Steps per trial')
    plt.subplot(2,1,2)
    plt.plot(rewards_per_trial)
    plt.ylabel('Rewards per trial')
    
    