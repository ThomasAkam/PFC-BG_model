#%% Imports.

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import pylab as plt
from sklearn.decomposition import PCA

import Two_step_task as ts

#%% train model.    
    
def train_model(n_steps=20000, n_back=30, batch=100, n_rnn=16, learning_rate=0.01,
                epochs=1, good_prob=0.9):
    # Get data
    task = ts.Two_step(good_prob=good_prob, block_len=[30,31])
    states = np.zeros(n_steps, int)
    rewards = np.zeros([n_steps, 1], 'float32')
    actions = np.zeros(n_steps, int)
    s = task.reset()
    r = 0
    for i in range(n_steps):
        states[i] = s
        rewards[i,0] = r
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
    # Define model.
    inputs = layers.Input(shape=(n_back, task.n_states+task.n_actions)) # Inputs are 1 hot encoding of alternatig states and actions.
    rnn = layers.GRU(n_rnn, unroll=True, name='rnn')(inputs) # Recurrent layer.
    state_pred = layers.Dense(task.n_states, activation='softmax', name='state_pred')(rnn) # Output layer predicts next state
    model = keras.Model(inputs=inputs, outputs=state_pred)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer)
    # Model variant used to get output of rnn layer.
    stateview_model = keras.Model(inputs=model.input,
                                 outputs=model.get_layer('rnn').output)
    # Fit model holding out last 1000 timesteps for evaluation.
    model.fit(x[:-1000], y[:-1000], batch_size=batch, epochs=epochs)
    # Plot reward probabilities on test data.
    y_pred = model.predict(x[-1000:])
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
    rnn_state = stateview_model.predict(x[-1000:]) 
    choice_inds = np.where(states[-1000:] == ts.choice)[0]
    ch_state = rnn_state[choice_inds,:]
    pca = PCA(n_components=1).fit(ch_state)
    ch_state_pc1 = pca.transform(ch_state)
    plt.subplot(2,1,2)
    plt.plot(ch_state_pc1)
    plt.ylabel('RNN state PC1')

