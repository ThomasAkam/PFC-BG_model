'''Script for experimenting with the RNN component of the model without the RL component.'''
# © Thomas Akam, 2023, released under the GPLv3 licence.

#%% Imports.

import numpy as np
import torch 
from torch import nn
from torch.utils.data import DataLoader
from torch import Tensor as tensor
import torch.nn.functional as F
import pylab as plt
from sklearn.decomposition import PCA

import two_step_task as ts

#%% parameters 
    
n_steps=20000
n_back=30
batch=100
n_rnn=16
learning_rate=0.01
epochs=1
good_prob=0.9

#%% Generate data

task = ts.Two_step(good_prob=good_prob, block_len=[30,31])
states = np.zeros(n_steps, int)
actions = np.zeros(n_steps, int)
rewards = np.zeros(n_steps, int)
s = task.reset()
for i in range(n_steps):
    actions[i]  = task.sample_appropriate_action(s)
    s, r = task.step(actions[i])
    states[i] = s
    rewards[i] = r
states=torch.from_numpy(states)
actions=torch.from_numpy(actions)
rewards=torch.from_numpy(rewards)
states_v = F.one_hot(states, task.n_states)*(rewards)[:,None]
actions_v = F.one_hot(actions, task.n_actions)
states_actions = torch.hstack([states_v, actions_v])
states_actions=tensor.numpy(states_actions)
x = [] # State-action sequences
y = [] # Next states
for i in range(0, len(states_actions) - n_back):
    sa = states_actions[i:i+n_back,:].copy()
    y.append(sa[-1,:task.n_states].copy())
    sa[-1,:task.n_states] = 0
    x.append(sa)
print("Number of sequences:", len(x))
x = np.stack(x)
x=torch.from_numpy(x)
x=x.float()
y = np.stack(y)
y=torch.from_numpy(y)
y=y.float()

#%% Define model.
input_size=task.n_states+task.n_actions# Inputs are 1 hot encoding of alternatig states and actions.
seq_length=n_back
hidden_size=n_rnn
num_layers=1
num_classes=task.n_states
class PFC_model(nn.Module):
    # implemented using RNN model
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(PFC_model,self).__init__()
        self.hidden_size= hidden_size
        self.num_layers=num_layers
        self.rnn=nn.GRU(input_size, hidden_size, num_layers, batch_first=True)# Recurrent layer.
        self.state_pred=nn.Linear(hidden_size,num_classes)# Output layer predicts next state
    def forward(self, x):
        h0=torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, hx=self.rnn(x,h0)
        #we want to decode the hidden state from just the last time step
        hidden=out[:,-1,:]
        #the self.rnn returns the code from the last layer of the Gru, i.e. the last hidden state or the PFC_layer
        out=F.softmax(self.state_pred(hidden))
        return out, hidden                                                                                                               
model=PFC_model(input_size, hidden_size, num_layers, num_classes)
loss_fn= nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)


#%% Fit model holding out last 1000 timesteps for evaluation.

trainingdataset=torch.utils.data.TensorDataset(x[:-1000],y[:-1000])
train_loader=torch.utils.data.DataLoader(dataset=trainingdataset, batch_size=batch, shuffle=False)
#training loop
n_total_steps=n_steps
for epoch in range (epochs):
    for i, (w, z) in enumerate(train_loader):
        inputs=w
        #Ensures that the tensor length is the correct length 
        inputs=inputs.reshape(-1, seq_length, input_size)
        labels=z
        #Forward pass
        outputs,__=model(inputs)
        loss=loss_fn(outputs, labels)
        #Backward and optimise
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

with torch.no_grad():
    y_pred, RNN_state=  model(x[-1000:])
states=tensor.numpy(states)


# Plot predicted reward probabilities across trials for test data.
y_pred=tensor.detach(y_pred).numpy()
A_out_inds = np.where(states[-1000:] == ts.sec_step_A)[0][:-1]+2
B_out_inds = np.where(states[-1000:] == ts.sec_step_B)[0][:-1]+2
A_reward_probs = y_pred[:,ts.reward_A]
B_reward_probs = y_pred[:,ts.reward_B]
plt.figure()
plt.subplot(2,1,1)
plt.plot(A_out_inds, A_reward_probs[A_out_inds])
plt.plot(B_out_inds, B_reward_probs[B_out_inds])
plt.ylabel('Estimated reward probs')
# Plot projection of choice state PFC activity onto its first principal component across trials.
rnn_state=tensor.detach(RNN_state).numpy()
choice_inds = np.where(states[-1000:] == ts.choice)[0]+1
ch_state = rnn_state[choice_inds,:]
pca = PCA(n_components=1).fit(ch_state)
ch_state_pc1 = pca.transform(ch_state)
plt.subplot(2,1,2)
plt.plot(ch_state_pc1)
plt.ylabel('RNN state PC1')

