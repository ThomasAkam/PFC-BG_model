#%% Imports.

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import pylab as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA

import Two_step_task as ts

#%% Utility funcs 

def generate_data(task, n_steps):
    states = np.zeros(n_steps, int)
    actions = np.zeros(n_steps, int)
    s = task.reset()
    for i in range(n_steps):
        states[i] = s
        actions[i]  = task.sample_appropriate_action(s)
        s, r, info = task.step(actions[i])
    states_v = keras.utils.to_categorical(states, task.n_states)
    actions_v = keras.utils.to_categorical(actions, task.n_actions)
    state_action = np.hstack([states_v, actions_v])
    return states, actions, state_action

def plot_reward_probs(states, y_pred, fig_no=1):
    A_out_inds = np.where(states == ts.sec_step_A)[0][:-1]+1
    B_out_inds = np.where(states == ts.sec_step_B)[0][:-1]+1
    A_reward_probs = y_pred[:,ts.reward_A]
    B_reward_probs = y_pred[:,ts.reward_B]
    plt.figure(fig_no, clear=True)
    plt.plot(A_out_inds, A_reward_probs[A_out_inds])
    plt.plot(B_out_inds, B_reward_probs[B_out_inds])
    plt.ylabel('Estimated reward probs')
    
def plot_lstm_state_PC1(states, lstm_state, fig_no=1):
    choice_inds = np.where(states == ts.choice)[0]
    ch_state = lstm_state[choice_inds,:]
    pca = PCA(n_components=1).fit(ch_state)
    ch_state_pc1 = pca.transform(ch_state)
    plt.figure(fig_no, clear=True)
    plt.plot(ch_state_pc1 , label='lstm state PC1')
    plt.ylabel('LSTM state PC1')
    plt.legend()

#%% train model.    
    
def train_model(n_steps=20000, n_back=30, batch=100, n_lstm=16, learning_rate=0.05,
                epochs=1, good_prob=0.9):
    # Get data
    task = ts.Two_step(good_prob=good_prob, block_len=[20,21])
    states, actions, states_actions = generate_data(task, n_steps)
    x = [] # State-action sequences
    y = [] # Next states
    for i in range(0, len(states_actions) - n_back):
        x.append(states_actions[i:i+n_back,:])
        y.append(states_actions[i+n_back, :task.n_states])
    print("Number of sequences:", len(x))
    x = np.stack(x)
    y = np.stack(y)
    # Define model.
    inputs = layers.Input(shape=(n_back, task.n_states+task.n_actions)) # Inputs are 1 hot encoding of alternatig states and actions.
    lstm = layers.LSTM(n_lstm, unroll=True, name='lstm', return_state=True)(inputs) # Recurrent layer.
    state_pred = layers.Dense(task.n_states, activation='softmax', name='state_pred')(lstm[0]) # Output layer predicts next state
    model = keras.Model(inputs=inputs, outputs=state_pred)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer)
    # Model variant used to get output of LSTM layer.
    lstm_output_model = keras.Model(inputs=model.input,
                                 outputs=model.get_layer('lstm').output[1:])
    # Fit model holding out last 1000 timesteps for evaluation.
    model.fit(x[:-1000], y[:-1000], batch_size=batch, epochs=epochs)
    # Plot reward probabilities on test data.
    y_pred = model.predict(x[-1000:])
    plot_reward_probs(states[-1000:], y_pred)
    
    # Look at features of lstm layer
    lstm_output, lstm_state = lstm_output_model.predict(x[-1000:]) 
    plot_lstm_state_PC1(states[-1000:], lstm_state, fig_no=2)

    