import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import pylab as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA

import Two_step_task as ts


def generate_data(task, n_steps):
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
    return states, actions, states_actions, block_info

def plot_outcomes(states, fig_no=1):
    A_out_inds = np.where(states == ts.sec_step_A)[0]+1
    B_out_inds = np.where(states == ts.sec_step_B)[0]+1
    A_outcomes = 0.+(states[A_out_inds] == ts.reward_A)
    B_outcomes = 1.-(states[B_out_inds] == ts.reward_B)
    outcomes = np.hstack([A_outcomes,B_outcomes])
    outcomes = outcomes[np.argsort(np.hstack([A_out_inds, B_out_inds]))]
    plt.figure(fig_no, clear=True)
    #plt.plot(A_out_inds, A_ou , '.')
    #plt.plot(B_out_inds, , '.')
    plt.plot(outcomes,'.')
    plt.plot(gaussian_filter1d(outcomes,4))
    
def eval_reward_prob_accuaracy(states, y_pred, block_info, fig_no=1):
    A_good = block_info[:,1]
    block_trial = block_info[:,0]
    A_out_inds = np.where(states == ts.sec_step_A)[0][:-1]+1
    B_out_inds = np.where(states == ts.sec_step_B)[0][:-1]+1
    A_reward_probs = y_pred[:,ts.reward_A]
    B_reward_probs = y_pred[:,ts.reward_B]
    plt.figure(fig_no, clear=True)
    plt.plot(A_out_inds, A_reward_probs[A_out_inds])
    plt.plot(B_out_inds, B_reward_probs[B_out_inds])
    
    
def eval_lstm_state_PC1(states, lstm_state, lstm_state_r, fig_no=1):
    choice_inds = np.where(states == ts.choice)[0]
    ch_state = lstm_state[choice_inds,:]
    ch_state_r = lstm_state_r[choice_inds,:]
    pca = PCA(n_components=1).fit(ch_state)
    ch_state_pc1 = pca.transform(ch_state)
    ch_state_r_pc1 = pca.transform(ch_state_r)
    plt.figure(fig_no, clear=True)
    plt.plot(ch_state_pc1 , label='lstm state PC1')
    plt.plot(ch_state_r_pc1, label='lstm state r PC1')
    plt.legend()
    
    
def train_model(n_steps=41000, n_back=30, batch=100, n_lstm=16, learning_rate=0.05,
                epochs=1, good_prob=0.9):
    # Get data
    task = ts.Two_step(good_prob=good_prob, block_len=[20,21])
    states, actions, states_actions, block_info = generate_data(task, n_steps)
    
    state_action_seqs = []
    next_states = []
    for i in range(0, len(states_actions) - n_back, 2):
        state_action_seqs.append(states_actions[i:i+n_back])
        next_states.append(states_actions[i+n_back])
    print("Number of sequences:", len(state_action_seqs))
    x = np.stack([keras.utils.to_categorical(sas, task.n_actions+task.n_states) for sas in state_action_seqs])
    y = keras.utils.to_categorical(np.array(next_states), task.n_states)
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
    #model.fit(x[:-1000], y[:-1000], batch_size=batch, epochs=epochs)
    model.fit(x[:20000], y[:20000], batch_size=batch, epochs=epochs)
    # Evaluate outcome prediction quality
    y_pred = model.predict(x[-1000:])
    pred_state = np.argmax(y_pred,1)
    eval_reward_prob_accuaracy(states[-1000:], y_pred, block_info[-1000:,:])
    # Look at features of lstm layer
    lstm_output, lstm_state = lstm_output_model.predict(x[-1000:])
    
    #xr = np.roll(x, 1, axis=1)
    #lstm_output_r, lstm_state_r = lstm_output_model.predict(xr[-1000:])
    
    model.fit(x[20000:40000], y[20000:40000], batch_size=batch, epochs=epochs)
    lstm_output_r, lstm_state_r = lstm_output_model.predict(x[-1000:])
    
    eval_lstm_state_PC1(states[-1000:], lstm_state, lstm_state_r, fig_no=2)
    #return states[-1000:], lstm_output, lstm_state
    