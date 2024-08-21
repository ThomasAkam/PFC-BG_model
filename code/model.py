'''Code to implement the model and run a single simulation.'''
# © Thomas Akam, 2023, released under the GPLv3 licence.

#%% Imports

import os
import json
import pickle
import numpy as np
import torch 
from torch import nn
from torch.utils.data import DataLoader
from torch import Tensor as tensor
import torch.nn.functional as F
from collections import namedtuple

import two_step_task as ts
import analysis as an
def one_hot(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]
 # code used to replicate keras.gather_nd function from https://gist.github.com/d4l3k/44548e97ee11153c4be026f62c62d38e
def torch_gather_nd(params: torch.Tensor, indices: torch.Tensor, batch_dim: int = 0) -> torch.Tensor:
    """torch_gather_nd implements tf.gather_nd in PyTorch. This supports multiple batch dimensions as well as multiple channel dimensions."""
    index_shape = indices.shape[:-1]
    num_dim = indices.size(-1)
    tail_sizes = params.shape[batch_dim+num_dim:]

    # flatten extra dimensions
    for s in tail_sizes:
        row_indices = torch.arange(s, device=params.device)
        indices = indices.unsqueeze(-2)
        indices = indices.repeat(*[1 for _ in range(indices.dim()-2)], s, 1)
        row_indices = row_indices.expand(*indices.shape[:-2], -1).unsqueeze(-1)
        indices = torch.cat((indices, row_indices), dim=-1)
        num_dim += 1

    # flatten indices and params to batch specific ones instead of channel specific
    for i in range(num_dim):
        size = int(np.prod(params.shape[batch_dim+i+1:batch_dim+num_dim]))
        
        indices[..., i] *= size

    indices = indices.sum(dim=-1)
    params = params.flatten(batch_dim, -1)
    indices = indices.flatten(batch_dim, -1)

    out = torch.gather(params, dim=batch_dim, index=indices)
    return out.reshape(*index_shape,*tail_sizes)

Episode = namedtuple('Episode', ['states', 'rewards', 'actions', 'pfc_inputs', 'pfc_states', 'pred_states','task_rew_states', 'n_trials'])

#%% Parameters.

default_params = {
    # Simulation params.
    'n_episodes'  : 500,
    'episode_len' : 100,  # Episode length in trials.
    'max_step_per_episode' : 600,
    'gamma' : 0.9,        # Discount rate

    #Task params.
    'good_prob' : 0.8,
    'block_len' : [20,40],

    # PFC model params.
    'n_back': 30, # Length of history provided as input.
    'n_pfc' : 16,  # Number of PFC units
    'pfc_learning_rate' : 0.01,
    'pred_rewarded_only' : False, # If True PFC input (and prediction target) is rewarded states only.

    # Striatum model params.
    'n_str' : 10, # Number of striatum units
    'str_learning_rate' : 0.05,
    'entropy_loss_weight' : 0.05}

#%% Run simulation.

def run_simulation(save_dir=None, pm=default_params):
    # Initialise random seed to ensure runs using multiprocessing use different random numbers.
    np.random.seed(int.from_bytes(os.urandom(4), 'little'))

    #Instantiate task.
    task = ts.Two_step(good_prob=pm['good_prob'], block_len=pm['block_len'])
    
    # PFC model.
    
    if pm['pred_rewarded_only']: # PFC input is one-hot encoding of observable state on rewarded trias, 0 vector on non-rewarded.
        input_size=(task.n_states)
        pfc_input_buffer =torch.zeros([pm['n_back'], task.n_states])
    else: # PFC input is 1 hot encoding of observable state and previous action.
        input_size=(task.n_states+task.n_actions)
        pfc_input_buffer = torch.zeros([pm['n_back'], task.n_states+task.n_actions])
    # Creates a replica of the input_buffer that can be used for analysis 
    pfc_buffer=tensor.detach(pfc_input_buffer).numpy()
    pfc_buffer=np.array(pfc_buffer, dtype=bool) 
    
    #PFC Model
    class PFC(nn.Module):
    # implemented using RNN model
        def __init__(self):
            super(PFC,self).__init__()
            self.hidden_size= pm['n_pfc']
            self.num_layers=1
            self.rnn=nn.GRU(input_size,  pm['n_pfc'], 1, batch_first=True)# Recurrent layer.
            self.state_pred=nn.Linear( pm['n_pfc'],task.n_states)# Output layer predicts next state
        def forward(self, x):
            h0=torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            out, _=self.rnn(x,h0)
            #we want to decode the hidden state from just the last time step
            hidden=out[:,-1,:]
            #the self.rnn returns the code from the last layer of the Gru, i.e. the last hidden state or the PFC_layer
            out=F.softmax(self.state_pred(hidden))
            return out, hidden                                                                                                
    PFC_model=PFC()
    pfc_loss_fn= nn.MSELoss()
    pfc_optimizer=torch.optim.Adam(PFC_model.parameters(), lr=pm['pfc_learning_rate'])

    def update_pfc_input(a,s,r):
        '''Update the inputs to the PFC network given the action, subsequent state and reward.'''
        xy=torch.detach(pfc_input_buffer[1:,:]).clone()
        pfc_input_buffer[:-1,:] = xy
        pfc_input_buffer[-1,:] = 0
        if pm['pred_rewarded_only']:
            pfc_input_buffer[-1,s] = r # One hot encoding on state on rewarded timesteps, 0 vector on non-rewarded.
        else:   
            pfc_input_buffer[-1,s] = 1               # One hot encoding of state.
            pfc_input_buffer[-1,a+task.n_states] = 1 # One hot encoding of action.
    def update_pfc_numpy(a,s,r):   # same code for updating the clone numpy as well
        '''Update the inputs to the PFC network given the action, subsequent state and reward.''' 
        pfc_buffer[:-1,:]=pfc_buffer[1:,:]
        pfc_buffer[-1,:] = 0
        if pm['pred_rewarded_only']:
            pfc_buffer[-1,s] = r # One hot encoding on state on rewarded timesteps, 0 vector on non-rewarded.
        else:   
            pfc_buffer[-1,s] = 1               # One hot encoding of state.
            pfc_buffer[-1,a+task.n_states] = 1

    def get_masked_PFC_inputs(pfc_inputs):
        '''Return array of PFC input history with the most recent state masked, 
        used for training as the most recent state is the prediction target.'''
        masked_pfc_inputs = np.array(pfc_inputs)
        masked_pfc_inputs[:,-1,:task.n_states] = 0
        masked_pfc_inputs=torch.from_numpy(masked_pfc_inputs)
        masked_pfc_inputs=tensor.float(masked_pfc_inputs)
        return masked_pfc_inputs
    
    # Striatum model
    
    class Striatum(nn.Module):
        def __init__(self):
            super(Striatum, self).__init__()
            self.input=nn.Linear((task.n_states+pm['n_pfc']),pm['n_str'])# represents the concatenation of the observal states and PFC activity
            self.actor=nn.Linear(pm['n_str'], task.n_actions)
            self.critic=nn.Linear(pm['n_str'],1)
            self.float()
        
        def forward(self, obs_state, pfc_state):
            y=torch.hstack((obs_state, pfc_state))
            y=F.relu(self.input(y))
            actor=F.softmax(self.actor(y), dim=-1)
            critic=self.critic(y)
            return actor, critic
    
    Str_model=Striatum ()
    str_optimizer=torch.optim.Adam(Str_model.parameters(), lr=pm['str_learning_rate'])
    
    # Environment loop.
    
    def store_trial_data(s, r, a, pfc_s, V):
        'Store state, reward and subseqent action, PFC input buffer, PFC activity, value.'
        states.append(s)
        rewards.append(r)
        actions.append(a)
        pfc_inputs.append(pfc_input_buffer.copy())
        pfc_states.append(pfc_s)
        values.append(V)
        task_rew_states.append(task.A_good)
        
    # Run model.
    
    s = task.reset() # Get initial state as integer.
    r = 0
    _, pfc_s =PFC_model(pfc_input_buffer[None,:,:])
    episode_buffer=[]
    
    for e in range(pm['n_episodes']):
        
        step_n = 0
        start_trial = task.trial_n
        
        # Episode history variables
        states  = []       # int
        rewards = []       # float
        actions = []       # int
        pfc_inputs = []    # (1,30,n_states+n_actions)
        pfc_states = []    # (1,n_pfc)
        values = []        # float
        task_rew_states = [] # bool
           
        while True:
            step_n += 1
            
            # Choose action.
            action_probs, V= Str_model(torch.from_numpy(one_hot(s, task.n_states)[None,:]), torch.detach(pfc_s).clone())
            a =np.random.choice(task.n_actions, p=np.squeeze(tensor.detach(action_probs).numpy()))
            
            # Store history.
            store_trial_data(s, r, a, tensor.detach(pfc_s).numpy(), tensor.detach(V).numpy())
            # Get next state and reward.
            s, r = task.step(a)
            
            # Get new pfc state.
            update_pfc_input(a,s,r)
            update_pfc_numpy(a,s,r)
            _, pfc_s=PFC_model(pfc_input_buffer[None,:,:])

    
            n_trials = task.trial_n - start_trial
            if n_trials == pm['episode_len'] or step_n >= pm['max_step_per_episode'] and s == 0:
                break # End of episode.  
                
        # Store episode data.
        predictions,_=PFC_model(get_masked_PFC_inputs(pfc_inputs))
        pred_states=tensor.detach(predictions).numpy()
        pred_states=np.argmax(pred_states,1)# Used only for analysis.
        episode_buffer.append(Episode(np.array(states), np.array(rewards), np.array(actions), np.array(pfc_inputs),
                               np.vstack(pfc_states), np.array(pred_states), np.array(task_rew_states), n_trials))
        
        # Update striatum weights using advantage actor critic (A2C), Mnih et al. PMLR 48:1928-1937, 2016
        
        returns=np.zeros([len(rewards),1], dtype='float32')
        returns[-1]=tensor.detach(V).numpy()
        for i in range(1, len(returns)):
                returns[-i-1] = rewards[-i] + pm['gamma']*returns[-i] 

        advantages = torch.from_numpy((returns - np.vstack(values)).squeeze())
          
        # Calculate gradients
        action_probs_g, values_g = Str_model(torch.from_numpy(one_hot(states, task.n_states)), 
                                      torch.from_numpy(np.vstack(pfc_states)))# Gradient of these is tracked wrt Str_model weights.
        # Critic loss.
        critic_loss = F.mse_loss(values_g, torch.from_numpy(returns), reduction='sum')
        # Actor loss.
        log_chosen_probs=torch.log(torch_gather_nd(action_probs_g,torch.tensor([[i,a] for i,a in enumerate(actions)])))
        entropy = -torch.sum(action_probs_g*torch.log(action_probs_g),1)
        actor_loss = torch.sum(-log_chosen_probs*advantages-
                                entropy*pm['entropy_loss_weight'])
        policy_loss=actor_loss+critic_loss
        # Apply gradients
        str_optimizer.zero_grad()
        policy_loss.backward()
        # Update weights 
        str_optimizer.step()
        
        # Update PFC weights.

        if pm['pred_rewarded_only']: # PFC is trained to predict its current input given previous input.
            pfc_numpy=np.array(pfc_inputs[:-1])
            x=tensor.float(torch.from_numpy(pfc_numpy))
            y=tensor.float(torch.from_numpy(one_hot(states[1:], task.n_states)*np.array(rewards)[1:,np.newaxis]))
            batchsize=x.size()[0]
            batchdata=torch.utils.data.TensorDataset(x, y)
            batchloader=torch.utils.data.DataLoader(dataset=batchdata, batch_size=batchsize, shuffle=False)
        else: # PFC is trained to predict the current state given previous action and state.
            x=get_masked_PFC_inputs(pfc_inputs)
            batchsize=x.size()[0]
            y=tensor.float(torch.from_numpy(one_hot(states, task.n_states)))
            batchdata=torch.utils.data.TensorDataset(x, y)
            batchloader=torch.utils.data.DataLoader(dataset=batchdata, batch_size=batchsize, shuffle=False)
        
        for i, (w, z) in enumerate(batchloader):
            inputs=w.reshape(-1, pm['n_back'], input_size)
            labels=z
            outputs,__=PFC_model(inputs)
            tl=pfc_loss_fn(outputs, labels)
            pfc_optimizer.zero_grad()
            tl.backward()
            pfc_optimizer.step()
            
        print(f'Episode: {e} Steps: {step_n} Trials: {n_trials} '
              f' Rew. per tr.: {np.sum(rewards)/n_trials :.2f} PFC tr. loss: {tl.item() :.3f}')
        
        if e % 10 == 9: an.plot_performance(episode_buffer, task)
        
    # Save data.    
    
    if save_dir:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        with open(os.path.join(save_dir,'params.json'), 'w') as fp:
            json.dump(pm, fp, indent=4)
        with open(os.path.join(save_dir, 'episodes.pkl'), 'wb') as fe: 
            pickle.dump(episode_buffer, fe)
        
        PATH="model.pt"
        torch.save({
            'PFC_model_state_dict': PFC_model.state_dict(),
            'Str_model_state_dict': Str_model.state_dict(),
            'pfc_optimizer': pfc_optimizer.state_dict(),
            'str_optimizer': str_optimizer.state_dict()
            }, os.path.join(save_dir, PATH))
        