import matplotlib
matplotlib.use('agg')
from pyLattice2D.data.create import get_data_batch
from pyLattice2D.utils.record import RecordAndSave
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import sys

lattice_setup = {'type': str(sys.argv[1]),
                 'num': int(sys.argv[2]),
                 'seed': int(sys.argv[3]),
                 'remove_nodes': 1,
                 'remove_edges': 0.2,
                 'move_nodes': [0.15, 1.],
                 'add_edges': [0., 0.2],
                 'BeamCrossArea_A': 2e-5,
                 'YoungsModulus_E': 200*1e9,
                 'oversaturate_edges': None,
                 'edge_mask_init': None,
                 'train_coordinates': False,
                 'num_samples': 20,
                }

experiment_setup = {'displacement': 0.02/30.,
                    'num_steps': 30,
                   'draw_response': False}

if str(sys.argv[1]) == 'honeycomb' or str(sys.argv[1]) == 'reentrant':
    print('change')
    lattice_setup['remove_edges'] = 0.2
    lattice_setup['move_nodes'] = [0.05, 1.0]
    lattice_setup['remove_nodes'] = 0

steps = int(sys.argv[4])
seeds = np.array(np.arange(steps)*1e7, dtype=int) + lattice_setup['seed']

for seed in tqdm(seeds):
    lattice_setup['seed'] = seed
    get_data_batch(lattice_setup, experiment_setup, 10, foldername = str(sys.argv[5]))


