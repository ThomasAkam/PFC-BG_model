'''Script for running and analysing simulation experiments comprised of multiple simulation runs.'''
# © Thomas Akam, 2023, released under the GPLv3 licence.

import os
from concurrent.futures import ProcessPoolExecutor

import analysis as an
import model as md

# Run simulation and save data to disk.

def run_experiment(exp_dir, params, n_runs=12, n_processes=6):
    '''Run an experiment comprising multiple simulation runs in parallel and save the data to disk.'''
    if not os.path.exists(exp_dir):
            os.mkdir(exp_dir)
    run_paths = [os.path.join(exp_dir, f'run_{i}') for i in range(n_runs)]
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        print('Running simulations')
        executor.map(md.run_simulation, run_paths, [params]*n_runs)
        
        
def run_experiments(n_runs=12, n_processes=6):
    '''Run one experiment using the default paramters and one for the model
    version in which the PFC networks state input is gated by reward.'''
    # Run experiment with default parameters.
    exp_dir_def=os.path.join('..','data','experiment_def')
    run_experiment(exp_dir_def, md.default_params, n_runs, n_processes)
    # Run experiment where PFC state input is gated by reward.
    params = md.default_params.copy()
    params['pred_rewarded_only'] = True
    exp_dir_pro=os.path.join('..','data','experiment_pro')
    run_experiment(exp_dir_pro, params, n_runs, n_processes)

# Analyse data that is already saved to disk.

def analyse_experiment(exp_name):
    '''Load data from specified experiment and run analyses, saving
    generated figures to plots directory.'''
    data_dir = os.path.join('..','data',exp_name)
    save_dir = os.path.join('..','plots',exp_name)
    experiment_data = an.load_experiment(data_dir)
    an.plot_experiment(experiment_data, save_dir=save_dir)
    
def analyse_experiments():
    # Analyse experiment with default parameters.
    analyse_experiment('experiment_def')
    # Analyse experiment where PFC state input is gated by reward.
    analyse_experiment('experiment_pro')
    

# Run file to run experiments then analyse the data.
if __name__=='__main__':
    run_experiments()
    analyse_experiments()