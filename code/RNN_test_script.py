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
states_v = F.one_hot(torch.from_numpy(states), task.n_states)*(torch.from_numpy(rewards))[:,None]
actions_v = F.one_hot(torch.from_numpy(actions), task.n_actions)
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
x=tensor.float(torch.from_numpy(np.stack(x)))
y=tensor.float(torch.from_numpy(np.stack(y)))


#%% Define model.
class PFC_model(nn.Module):
    # implemented using RNN model
    def __init__(self):
        super(PFC_model,self).__init__()
        self.hidden_size= n_rnn
        self.num_layers=1
        self.rnn=nn.GRU(task.n_states+task.n_actions, n_rnn, 1, batch_first=True)# Recurrent layer.
        self.state_pred=nn.Linear(n_rnn,task.n_states)# Output layer predicts next state
    def forward(self, x):
        h0=torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _=self.rnn(x,h0)
        hidden=out[:,-1,:]
        out=F.softmax(self.state_pred(hidden))
        return out, hidden      # hidden used to get output of rnn layer.                                                                                                         
model=PFC_model()
#loss_fn=nn.CrossEntropyLoss()
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
        inputs=inputs.reshape(-1, n_back,task.n_states+task.n_actions )
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

#%% Plotting

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

