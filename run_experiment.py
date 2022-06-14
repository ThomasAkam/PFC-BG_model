import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor

import PFC_RL_model as md

exp_dir = os.path.join('..','data','experiment') 

def run_experiment(n_reps=4, n_processes=4):
    if not os.path.exists(exp_dir):
            os.mkdir(exp_dir)
    run_paths = [os.path.join(exp_dir, f'run_{i}') for i in range(n_reps)]
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        print('Running simulations')
        executor.map(md.run_simulation, run_paths)