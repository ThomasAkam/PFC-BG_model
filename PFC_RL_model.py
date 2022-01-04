#%% Imports

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import pylab as plt
from sklearn.decomposition import PCA

import Two_step_task as ts

#%% Parameters.

# Simulation parameters.
n_steps = 10000

#Task params.
good_prob = 0.99
block_len = [20,21]

# PFC model parameters.
n_back = 30
n_lstm = 16
pfc_learning_rate = 0.05

# Striatum model parameters.
str_learning_rate = 0.05
ave_reward_learning_rate = 0.01

#%% Instantiate task.

task = ts.Two_step(good_prob=good_prob, block_len=block_len)

#%% PFC model.

pfc_inputs = layers.Input(shape=(n_back, task.n_states+task.n_actions)) # Inputs are 1 hot encoding of alternating states and actions.
lstm = layers.LSTM(n_lstm, unroll=True, name='lstm', return_state=True)(pfc_inputs) # Recurrent layer.
state_pred = layers.Dense(task.n_states, activation='softmax', name='state_pred')(lstm[0]) # Output layer predicts next state
PFC_model = keras.Model(inputs=pfc_inputs, outputs=state_pred)
optimizer = keras.optimizers.Adam(learning_rate=pfc_learning_rate)
PFC_model.compile(loss="categorical_crossentropy", optimizer=optimizer)
# Model variant used to get state of LSTM layer.
PFC_state_view = keras.Model(inputs=PFC_model.input,
                             outputs=PFC_model.get_layer('lstm').output[2])

pfc_input_buffer = np.zeros([n_back, task.n_states+task.n_actions])

def update_pfc_input_buffer(s=None, a=None):
    '''Given an integer state or action update the pfc_input_buffer array.'''
    global pfc_input_buffer
    pfc_input_buffer = np.roll(pfc_input_buffer,-1, axis=0)
    pfc_input_buffer[-1,:] = 0
    if s is not None: # One-hot encoding of state s.
        pfc_input_buffer[-1,s] = 1
    else: # One hot encoding of action a.
        pfc_input_buffer[-1,a+task.n_states] = 1

def get_state_features_orig(s):
    '''Get the new PFC features for states s and return feature vector S_op
    combining observable and PFC features.'''
    update_pfc_input_buffer(s=s)
    S_op = np.zeros(task.n_states+n_lstm) # State feature vector combining observable and PFC features.
    S_op[s] = 1 # Observable task state feature.
    S_op[task.n_states:] = PFC_state_view(pfc_input_buffer[np.newaxis,:,:]).numpy() # PFC features
    return S_op

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

pca = pretrain_model()
    
def get_state_features(s):
    '''Get the new PFC features for states s and return feature vector S_op
    combining observable and PFC features.'''
    update_pfc_input_buffer(s=s)
    S_op = np.zeros(task.n_states+3) # State feature vector combining observable and PFC features.
    S_op[s] = 1 # Observable task state feature.
    pc1 = pca.transform(PFC_state_view(pfc_input_buffer[np.newaxis,:,:]).numpy())
    if s == ts.choice:
        S_op[task.n_states] = pc1
    elif s == ts.sec_step_A:
        S_op[task.n_states + 1] = pc1
    elif s == ts.sec_step_B:
        S_op[task.n_states + 2] = pc1
    return S_op

#%% Striatum model

#w_c = np.zeros(task.n_states+n_lstm)                   # Critic weights.
#w_a = np.zeros([task.n_actions, task.n_states+n_lstm]) # Actor weights.

w_c = np.zeros(task.n_states+3)                   # Critic weights.
w_a = np.zeros([task.n_actions, task.n_states+3]) # Actor weights.

r_ave = 0 # Average reward.
r_per_step = 0

def choice_probs(S,w_a):
    '''Given state feature vector S and actor weights w_a return choice probabilities.'''
    return np.exp(w_a @ S)/np.sum(np.exp(w_a @ S))

def log_chosen_prob_grad(a,S,w_a):
    '''Given integer action a, state feature vector S and actor weights w_a,
    return the gradient of log(choice_probs(S,w_a)[a]) with respect to w_a.'''
    grad = -choice_probs(S,w_a)[:,np.newaxis]*S
    grad[a,:] += S
    return grad

#%% Environment loop.

s = task.reset() # Get initial state as integer.
S_op = get_state_features(s)

for i in range(n_steps):
    
    # Choose action.
    a = np.random.choice(task.n_actions, p=choice_probs(S_op, w_a))
    
    # Get next state and reward.
    s, r, block_info = task.step(a)
    
    # Train PFC to predict next state.
    update_pfc_input_buffer(a=a)
    if False: # i > 100:
        PFC_model.train_on_batch(x=pfc_input_buffer[np.newaxis,:,:],
                             y=keras.utils.to_categorical(s, task.n_states)[np.newaxis,:])
    
    # Get new state features.
    S_new = get_state_features(s)
    
    # Update striatum model.
    d = r - r_ave + w_c @ S_new - w_c @ S_op     # Reward prediction error.
    r_ave = r_ave + ave_reward_learning_rate * d # Update average reward.
    w_c = w_c + str_learning_rate * d * S_op     # Update critic weights.
    w_a = w_a + str_learning_rate * d * log_chosen_prob_grad(a, S_op, w_a) # Update actor weights
    
    S_op = S_new.copy()
    
    r_per_step = r_per_step + ave_reward_learning_rate*(r-r_per_step)
    
    if i % 100 == 0:
        print(f'Step: {i} Reward per step: {r_per_step :.3f} r_ave:{r_ave :.3f}')
    