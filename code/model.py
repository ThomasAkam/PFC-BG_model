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

class PFC_model (nn.Module):
        def __init__(self,pm, input_size, task):
            super(PFC_model,self).__init__()
            self.hidden_size= pm['n_pfc']
            self.num_layers=1
            self.rnn=nn.GRU(input_size,  pm['n_pfc'], 1, batch_first=True)# Recurrent layer.
            self.state_pred=nn.Linear( pm['n_pfc'],task.n_states)# Output layer predicts next state
        def forward(self, x):
            h0=torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            out, _=self.rnn(x,h0)
            hidden=out[:,-1,:]
            out=F.softmax(self.state_pred(hidden))
            return out, hidden         #Hidden used to get state of RNN layer 

class Str_model(nn.Module):
        def __init__(self, pm, task):
            super(Str_model, self).__init__()
            self.input=nn.Linear((task.n_states+pm['n_pfc']),pm['n_str'])#Observable state and PFC activity features
            self.actor=nn.Linear(pm['n_str'], task.n_actions)
            self.critic=nn.Linear(pm['n_str'],1)
            self.float()
        def forward(self, obs_state, pfc_state):
            y=torch.hstack((obs_state, pfc_state))
            y=F.relu(self.input(y))
            actor=F.softmax(self.actor(y), dim=-1)
            critic=self.critic(y)
            return actor, critic
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
   
                                                                                          
    pfc_model=PFC_model(pm, input_size, task)
    pfc_loss_fn= nn.MSELoss()
    pfc_optimizer=torch.optim.Adam(pfc_model.parameters(), lr=pm['pfc_learning_rate'])

    def update_pfc_input(a,s,r):
        '''Update the inputs to the PFC network given the action, subsequent state and reward.'''
        pfc_input_buffer[:-1,:] = torch.detach(pfc_input_buffer[1:,:]).clone()
        pfc_input_buffer[-1,:] = 0
        if pm['pred_rewarded_only']:
            pfc_input_buffer[-1,s] = r # One hot encoding on state on rewarded timesteps, 0 vector on non-rewarded.
        else:   
            pfc_input_buffer[-1,s] = 1               # One hot encoding of state.
            pfc_input_buffer[-1,a+task.n_states] = 1 # One hot encoding of action.

    def get_masked_PFC_inputs(pfc_inputs):
        '''Return array of PFC input history with the most recent state masked, 
        used for training as the most recent state is the prediction target.'''
        masked_pfc_inputs = np.array(pfc_inputs)
        masked_pfc_inputs[:,-1,:task.n_states] = 0
        masked_pfc_inputs=tensor.float(torch.from_numpy(masked_pfc_inputs))
        return masked_pfc_inputs
    
    # Striatum model
    str_model=Str_model(pm, task)
    str_optimizer=torch.optim.Adam(str_model.parameters(), lr=pm['str_learning_rate'])
    
    # Environment loop.
    
    def store_trial_data(s, r, a, pfc_s, V):
        'Store state, reward and subseqent action, PFC input buffer, PFC activity, value.'
        states.append(s)
        rewards.append(r)
        actions.append(a)
        pfc_inputs.append(tensor.detach(tensor.detach(pfc_input_buffer).clone()).numpy())
        pfc_states.append(pfc_s)
        values.append(V)
        task_rew_states.append(task.A_good)
        
    # Run model.
    
    s = task.reset() # Get initial state as integer.
    r = 0
    _, pfc_s =pfc_model(pfc_input_buffer[None,:,:])
    
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
            action_probs, V= str_model(F.one_hot(torch.tensor(s), task.n_states)[None,:], torch.detach(pfc_s).clone())
            a =np.random.choice(task.n_actions, p=np.squeeze(tensor.detach(action_probs).numpy()))
            
            # Store history.
            store_trial_data(s, r, a, tensor.detach(pfc_s).numpy(), tensor.detach(V).numpy())
            
            # Get next state and reward.
            s, r = task.step(a)
            
            # Get new pfc state.
            update_pfc_input(a,s,r)

            _, pfc_s=pfc_model(pfc_input_buffer[None,:,:])

    
            n_trials = task.trial_n - start_trial
            if n_trials == pm['episode_len'] or step_n >= pm['max_step_per_episode'] and s == 0:
                break # End of episode.  
                
        # Store episode data.
        
        predictions,_=pfc_model(get_masked_PFC_inputs(pfc_inputs))
        pred_states=np.argmax(tensor.detach(predictions).numpy(),1)# Used only for analysis.
        episode_buffer.append(Episode(np.array(states), np.array(rewards), np.array(actions), np.array(pfc_inputs),
                               np.vstack(pfc_states), np.array(pred_states), np.array(task_rew_states), n_trials))
        
        # Update striatum weights using advantage actor critic (A2C), Mnih et al. PMLR 48:1928-1937, 2016
        
        returns=np.zeros([len(rewards),1], dtype='float32')
        returns[-1]=tensor.detach(V).numpy()
        for i in range(1, len(returns)):
                returns[-i-1] = rewards[-i] + pm['gamma']*returns[-i] 

        advantages = torch.from_numpy((returns - np.vstack(values)).squeeze())
          
        # Calculate gradients
        
        # Critic loss.
        action_probs_g, values_g = str_model(F.one_hot(torch.tensor(states), task.n_states), 
                                             torch.from_numpy(np.vstack(pfc_states)))# Gradient of these is tracked wrt str_model weights.
        critic_loss = F.mse_loss(values_g, torch.from_numpy(returns), reduction='sum')
        # Actor loss.
        chosen_probs=torch.gather(action_probs_g,1, torch.transpose(torch.tensor([actions]),0,1))
        log_chosen_probs=torch.log(torch.transpose(chosen_probs,1,0))
        entropy = -torch.sum(action_probs_g*torch.log(action_probs_g),1)
        actor_loss = torch.sum(-log_chosen_probs*advantages-entropy*pm['entropy_loss_weight'])
        policy_loss=actor_loss+critic_loss
        # Apply gradients
        str_optimizer.zero_grad()
        
        policy_loss.backward()
        
        # Update weights 
        
        str_optimizer.step()
        
        # Update PFC weights.

        if pm['pred_rewarded_only']: # PFC is trained to predict its current input given previous input.
            x=tensor.float(torch.from_numpy(np.array(pfc_inputs[:-1])))
            y=tensor.float(F.one_hot(torch.tensor(states[1:]), task.n_states)*np.array(rewards)[1:,None])
            batchsize=x.size()[0]
            batchdata=torch.utils.data.TensorDataset(x, y)
            batchloader=torch.utils.data.DataLoader(dataset=batchdata, batch_size=batchsize, shuffle=False)
        else: # PFC is trained to predict the current state given previous action and state.
            x=get_masked_PFC_inputs(pfc_inputs)
            batchsize=x.size()[0]
            y=tensor.float(F.one_hot(torch.tensor(states), task.n_states))
            batchdata=torch.utils.data.TensorDataset(x, y)
            batchloader=torch.utils.data.DataLoader(dataset=batchdata, batch_size=batchsize, shuffle=False)
        
        for i, (w, z) in enumerate(batchloader):
            inputs=w.reshape(-1, pm['n_back'], input_size)
            labels=z
            outputs,__=pfc_model(inputs)
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
            'PFC_model_state_dict': pfc_model.state_dict(),
            'Str_model_state_dict': str_model.state_dict(),
            'pfc_optimizer': pfc_optimizer.state_dict(),
            'str_optimizer': str_optimizer.state_dict()
            }, os.path.join(save_dir, PATH))
        