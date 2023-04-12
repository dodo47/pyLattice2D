from pyLattice2D.lattices.triangle import TriangleLattice
from pyLattice2D.lattices.voronoi import VoronoiLattice
from pyLattice2D.lattices.honeycomb import KagomeLattice, HoneycombLattice, ReentrantHoneycombLattice
from pyLattice2D.lattices.square import SquareLattice
import numpy as np

def get_lattice_object(lattice_setup, seed):
    '''
    Convenience function for creating a vanilla (non-differentiable) lattice object.
    '''
    if lattice_setup['type'] == 'voronoi':
        lattice = VoronoiLattice(delta_ratio = lattice_setup['delta_ratio'], num_layers = lattice_setup['num'], box_width= 1, rotate=False, seed=seed)
    elif lattice_setup['type'] == 'square':
        lattice = SquareLattice(num_layers = lattice_setup['num'], box_width= 1, rotate=False, seed=seed)
    elif lattice_setup['type'] == 'triangle':
        lattice = TriangleLattice(num_layers = lattice_setup['num'], box_width= 1, rotate=False, seed=seed)
    elif lattice_setup['type'] == 'honeycomb':
        lattice = HoneycombLattice(num_layers = lattice_setup['num'], box_width= 1, rotate=False, seed=seed)
    elif lattice_setup['type'] == 'kagome':
        lattice = KagomeLattice(num_layers = lattice_setup['num'], box_width= 1, rotate=False, seed=seed)
    elif lattice_setup['type'] == 'reentrant':
        lattice = ReentrantHoneycombLattice(num_layers = lattice_setup['num'], box_width= 1, rotate=False, seed=seed)
    else:
        raise Exception('Lattice type {} not supported!'.format(lattice_setup['type']))
    return lattice

def perturb_lattice(lattice_setup, lattice):
    '''
    Convenience function for perturbing the base tiling of a (non-differentaible) lattice object.

    E.g., nodes can be randomly moved/removed, edges can be randomly removed/added.
    '''
    if lattice_setup['move_nodes'][1] > 0 and lattice_setup['move_nodes'][0] > 0:
        lattice.move_nodes(lattice_setup['move_nodes'][0], lattice_setup['move_nodes'][1], mode='uniform')

    if lattice_setup['remove_nodes'] > 0:
        lattice.randomly_delete_nodes(lattice_setup['remove_nodes'])

    if lattice_setup['remove_edges'] > 0:
        lattice.delete_edges(lattice_setup['remove_edges'])

    if lattice_setup['add_edges'][0] > 0:
        for i in range(lattice_setup['add_edges'][0]):
            lattice.add_random_edge(lattice_setup['add_edges'][1])
    lattice.lattice.edges = np.array(lattice.lattice.edges)       
    assert(lattice.lattice.valid)

def get_BeamShapeFactor(BeamCrossArea):
    '''
    Calculate I for the lattice beams.
    '''
    return BeamCrossArea**2/12.

def analytical_stiffness(density, latticeType, YoungsModulus= 200*1e9):
    '''
    Convenience function returning the analytical stiffness for different lattice types.
    '''
    if latticeType == 'square':
        return 0.5*density*YoungsModulus
    elif latticeType == 'triangle':
        return density*YoungsModulus/3.
    elif latticeType == 'kagome':
        return density*YoungsModulus/3.
    elif latticeType == 'honeycomb':
        return 3/2. * density**3 * YoungsModulus
    elif latticeType == 'reentrant':
        return 81./128 * density**3 * YoungsModulus
    else:
        raise Exception('Analytical solution for lattice type {} not supported!'.format(lattice_setup['type']))