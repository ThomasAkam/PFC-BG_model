import os
from concurrent.futures import ProcessPoolExecutor

import PFC_RL_model as md

def run_experiment(exp_dir=os.path.join('..','data','experiment08_16'), n_runs=18, n_processes=6):
    '''Run an experiment comprising multiple simulation runs in parallel and save the data to disk.'''
    if not os.path.exists(exp_dir):
            os.mkdir(exp_dir)
    run_paths = [os.path.join(exp_dir, f'run_{i}') for i in range(n_runs)]
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        print('Running simulations')
        executor.map(md.run_simulation, run_paths)