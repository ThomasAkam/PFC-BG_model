import os
from concurrent.futures import ProcessPoolExecutor

import model as md

def run_experiment(exp_dir, params, n_runs=16, n_processes=8):
    '''Run an experiment comprising multiple simulation runs in parallel and save the data to disk.'''
    if not os.path.exists(exp_dir):
            os.mkdir(exp_dir)
    run_paths = [os.path.join(exp_dir, f'run_{i}') for i in range(n_runs)]
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        print('Running simulations')
        executor.map(md.run_simulation, run_paths, [params]*n_runs)
        
        
def run_experiments():
    # Run experiment with default parameters.
    exp_dir_def=os.path.join('..','data','experiment_def')
    run_experiment(exp_dir_def, md.default_params, n_runs=16, n_processes=8)
    # Run experiment where PFC only sees rewarded states as input.
    params = md.default_params.copy()
    params['pred_rewarded_only'] = True
    exp_dir_pro=os.path.join('..','data','experiment_pro')
    run_experiment(exp_dir_pro, params, n_runs=16, n_processes=8)