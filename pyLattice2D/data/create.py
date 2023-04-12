from pyLattice2D.models.Lattice import DifferentiableLattice
from pyLattice2D.models.FEM import FEModel
from pyLattice2D.methods.mechanical_properties import get_Yprop
import numpy as np
import random
import matplotlib.pyplot as plt
from copy import deepcopy
from pyLattice2D.lattices.base import Lattice
from multiprocessing import Pool
from pyLattice2D.utils.record import RecordAndSave
import os


class LatticeSeeder:
    def __init__(self, lattice_setup, experiment_setup):
        '''
        Class used for generating a dataset.
        '''
        self.lattice_setup = lattice_setup
        self.experiment = experiment_setup
        
        np.random.seed(self.lattice_setup['seed'])
        self.seeds = np.random.randint(1e9, size=self.lattice_setup['num_samples'])
        
    def create_lattice_from_seed(self, index):
        '''
        Create a random lattice and measure its mechanical properties. 

        Index: ID of the seed to be used stored in self.seeds
        '''
        np.random.seed(self.seeds[index])
        random.seed(self.seeds[index])
        
        lsetup = deepcopy(self.lattice_setup)
        lsetup['seed'] = self.seeds[index]
        if lsetup['type'] == 'voronoi':
            lsetup['delta_ratio'] = np.random.random()*0.75
            lsetup['num'] = np.random.randint(20)+150
        else:
            lsetup['remove_edges'] *= np.random.random()
            lsetup['move_nodes'][0] *= np.random.random()
            lsetup['move_nodes'][1] *= np.random.random()
            lsetup['remove_nodes'] = np.random.randint(lsetup['remove_nodes']+1)
        
        try:
            model = FEModel()
            lat = DifferentiableLattice(lsetup)
               
            experiment_setup = {'forced_nodes': lat.original.top_nodes,
                            'static_nodes': lat.original.bottom_nodes,
                            'displacement': self.experiment['displacement'],
                            'num_steps': self.experiment['num_steps'],
                            'draw_response': self.experiment['draw_response']}

            dens = lat.density
            E, sigma, delta = get_Yprop(lat, model, experiment_setup)
            
            if ((E < 0) == True) or (E.isnan() == True) or (sigma.isnan() == True):
                raise Exception('Measured values invalid. Skip lattice')
            
            return {'Stiff': E.detach().numpy(), 'Poisson': sigma.detach().numpy(), 'Density': dens.detach().numpy(),\
                    'Displacement': delta.detach().numpy(), 'Coordinates': lat.coordinates.detach().numpy(),\
                    'Edges': lat.original.edges, 'UnpCoordinates': lat.original._unperturbed_coordinates}
        
        except:
            return None
        
def folder_structure(foldername, lattice_type):
    '''
    Create folder structure for storing the dataset.
    '''
    path = '../data/{}'.format(foldername)
    if os.path.exists(path) == False:
        os.makedirs(path)
    if os.path.exists('{}/images/{}'.format(path, lattice_type)) == False:
        os.makedirs('{}/images/{}'.format(path, lattice_type))
        
    return path
        
def get_data_batch(lattice_setup, experiment_setup, cores, foldername = 'Unnamed'):
    '''
    Create a batch of data.
    '''
    path = folder_structure(foldername, lattice_setup['type'])
    counter = 0
    loops = 0
    recorder = RecordAndSave(path, lattice_setup['type'])
    try:
        offset = len(recorder[list(recorder.data.keys())[0]])
    except:
        offset = 0
    while counter < lattice_setup['num_samples']:
        print(loops)
        lattice_setup['seed'] += loops
        ls = LatticeSeeder(lattice_setup, experiment_setup)
        with Pool(cores) as p:
            res = p.map(ls.create_lattice_from_seed, np.arange(lattice_setup['num_samples']-counter))
        for values in res:
            if values is not None:
                recorder.add(values)
                plot_data(lattice_setup, path, counter+offset,\
                         values['Edges'], values['Coordinates'],\
                         values['UnpCoordinates'], values['Displacement'])
                plt.close()
                counter += 1                
        loops += 1
    recorder.save()
    
def plot_data(lattice_setup, path, ID, edges, coordinates, unperturbed_coordinates, delta):
    new_setup = deepcopy(lattice_setup)
    new_setup['lattice'] = Lattice(edges, coordinates, unperturbed_coordinates)
    lat = DifferentiableLattice(new_setup)

    lat.draw_lattice(path='{}/images/{}/{}'.format(path, lattice_setup['type'], ID))
    lat.draw_lattice(delta, path='{}/images/{}/{}_deformed'.format(path, lattice_setup['type'], ID))
