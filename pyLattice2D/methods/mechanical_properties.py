from pyLattice2D.utils.draw import draw_linear_fit
import numpy as np
import torch

def get_Yprop(lattice, model, experiment_setup, draw=True):
    '''
    Perform a full compression experiment, with several steps where the top layer
    gets compressed further and further.

    Returns the effective elastic modulus, Poisson's ratio and the deformations for all nodes.

    Optionally, the stress-strain and strain-strain curve can be plotted.
    '''
    total_strain = torch.zeros(experiment_setup['num_steps']+1)

    top_and_bottom = set(lattice.original.top_nodes).union(set(lattice.original.bottom_nodes))
    left_nodes = list(set(lattice.original.left_nodes).difference(top_and_bottom))
    right_nodes = list(set(lattice.original.right_nodes).difference(top_and_bottom))
    width_changes = torch.zeros(experiment_setup['num_steps']+1)
    
    delta = 0
    for i in range(experiment_setup['num_steps']):
        dr, strain = model(lattice, experiment_setup, delta)
        total_strain[i+1] = total_strain[i] + strain
        delta += dr
        width_changes[i+1] = delta[right_nodes,0].mean()\
                        -delta[left_nodes,0].mean()

    total_stress = torch.arange(0, experiment_setup['num_steps']+1)*experiment_setup['displacement']

    effective_modulus = total_strain.sum()/total_stress.sum()
    poissons_ratio = width_changes.sum()/total_stress.sum()

    if draw == True:
        xfit = np.linspace(0, total_stress[-1], 100)
        yfit = effective_modulus.detach().numpy()*xfit
        draw_linear_fit(total_stress.detach().numpy(), total_strain.detach().numpy(),\
                       xfit, yfit,\
                       xlabel = 'x-stress', ylabel = 'y-strain')

        xfit = np.linspace(0, total_stress[-1], 100)
        yfit = poissons_ratio.detach().numpy()*xfit
        draw_linear_fit(total_stress.detach().numpy(), width_changes.detach().numpy(),\
                       xfit, yfit,\
                       xlabel = 'y-stress', ylabel = 'x-stress')

    return effective_modulus, poissons_ratio, delta