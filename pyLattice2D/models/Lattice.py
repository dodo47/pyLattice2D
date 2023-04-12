from pyLattice2D.fem_solver.constraints_and_deformations import Constraints, ExternalDisplacement
from pyLattice2D.lattices.base import Lattice
from pyLattice2D.utils.validity import no_crossings
from pyLattice2D.lattices.utils import get_lattice_object, perturb_lattice
from pyLattice2D.utils.draw import draw_graph
from pyLattice2D.utils.math import isBetween
from pyLattice2D.fem_solver.utils import super_fn, heaviside, get_real_mask
from collections import defaultdict
import torch
import dgl
import dgl.function as fn
import numpy as np
import seaborn as sns
import matplotlib
from tqdm import trange

class DifferentiableLattice(Lattice):
    def __init__(self, lattice_setup):
        '''
        Differentiable lattice object.

        Used together with FEModel to enable differentiable finite element simulations.
        '''
        super().__init__()
        self.type = lattice_setup['type']
        # allow loading of lattice tiling
        if 'lattice' in lattice_setup.keys():
            try:
                self.original = lattice_setup['lattice'].lattice
            except:
                self.original = lattice_setup['lattice']
        else:
            # create lattice tiling from scratch
            lattice = get_lattice_object(lattice_setup, lattice_setup['seed'])
            perturb_lattice(lattice_setup, lattice) # add perturbations to the lattice topology
            self.original = lattice.lattice
        self.orig_num_edges = self.original.num_edges

        # node coordinates of the lattice
        self.coordinates = self.init_coordinates(lattice_setup['train_coordinates'])

        self.BeamCrossArea_A = lattice_setup['BeamCrossArea_A']
        self.YoungsModulus_E = lattice_setup['YoungsModulus_E']

        # oversaturate means: allow new edges to be added that have not been part of the base tiling
        if lattice_setup['oversaturate_edges'] is not None:
            self.add_additional_edges(lattice_setup['oversaturate_edges']['r'], lattice_setup['oversaturate_edges']['p'])
            self.edge_constraints = self.get_edge_constraints()
        else:
            self.edge_constraints = None

        # edge mask is used to add/remove edges during inverse design
        if lattice_setup['edge_mask_init'] is not None:
            self.threshold_function = self.set_threshold_function(lattice_setup['edge_mask_init'])
            self.edge_mask = self.init_edge_mask(lattice_setup['edge_mask_init'])
        else:
            self.edge_mask = None
            self.threshold_function = None

        # Turn lattice into a graph
        self.graph = dgl.graph((self.original.edges[:,0], self.original.edges[:,1]), num_nodes=self.original.num_nodes)
        # Object for adding constraints to the FE
        self.Constrainer = Constraints(self.original.num_nodes)
        # Object for enforcing node displacements (externally induced)
        self.Displacer = ExternalDisplacement(self.original.num_nodes)

    def load_lattice(self, edges, coordinates):
        self.original.edges = edges
        self.orig_num_edges = self.original.num_edges
        self.coordinates = torch.Tensor(coordinates)
        self.edge_mask = None
        self.edge_constraints = None

        self.graph = dgl.graph((self.original.edges[:,0], self.original.edges[:,1]), num_nodes=self.original.num_nodes)

        self.Constrainer = Constraints(self.original.num_nodes)
        self.Displacer = ExternalDisplacement(self.original.num_nodes)

    @property
    def num_edges(self):
        raise Exception('Not implemented yet!')
    
    @property
    def num_cells(self):
        raise Exception('Not implemented yet!')
    
    @property
    def valid(self):
        raise Exception('Not implemented yet!')

    @property
    def active_edges(self):
        '''
        Return edges that are not masked out.
        '''
        if self.edge_mask is None:
            return self.original.edges
        else:
            return self.original.edges[self.active_edge_mask]

    @property
    def active_mask(self):
        '''
        Return mask values of edges that are not masked out.
        '''
        if self.edge_mask is None:
            return None
        else:
            return self.edge_mask[self.active_edge_mask]

    @property
    def active_edge_mask(self):
        '''
        Return array containing a boolean entry indicating the existence of each edge.
        '''
        active_mask = self.threshold_function(self.edge_mask)
        if self.edge_constraints is not None:
            active_mask = get_real_mask(active_mask, self.edge_constraints, self.original.num_edges)
        active_mask = np.array(active_mask.detach().numpy(), dtype='bool')
        return active_mask

    @property
    def active_edge_mask_torch(self):
        '''
        Return torch array containing a boolean entry indicating the existence of each edge.
        '''
        if self.edge_mask is None:
            active_mask = 1
        else:
            active_mask = self.threshold_function(self.edge_mask)
            if self.edge_constraints is not None:
                active_mask = get_real_mask(active_mask, self.edge_constraints, self.original.num_edges)
        return active_mask

    def set_threshold_function(self, setup):
        '''
        Set the threshold function (surrogate gradient) to be used
        for the edge masking when applying automatic differentiation.
        '''
        if setup['train'] == False:
            def threshold_function(x):
                return heaviside(x)
            return threshold_function
        else:
            if setup['gradient']['function'] == 'super':
                def threshold_function(x):
                    return super_fn(x, alpha=setup['gradient']['alpha'])
            elif setup['gradient']['function'] == 'original':
                def threshold_function(x):
                    return heaviside(x)
        return threshold_function

    @property
    def density(self):
        '''
        Calculate relative density of the lattice.
        '''
        with self.graph.local_scope():
            self.graph.ndata['coordinates'] = self.coordinates
            self.graph.apply_edges(fn.u_sub_v('coordinates', 'coordinates', 'delta'))
            L = (torch.norm(self.graph.edata['delta'], dim=1)*self.active_edge_mask_torch).sum()

        material_height = torch.max(self.coordinates[:,1])-torch.min(self.coordinates[:,1])
        material_width = torch.max(self.coordinates[:,0])-torch.min(self.coordinates[:,0])
        material_area = material_height*material_width

        return L/material_area*np.sqrt(self.BeamCrossArea_A)

    def init_coordinates(self, train):
        '''
        Initialize the node coordinates.

        If train is True, the coordinates will be trainable using gradient descent.
        '''
        coordinates = torch.Tensor(self.original.coordinates)
        if train == True:
            coordinates.requires_grad = True
        return coordinates

    def init_edge_mask(self, setup):
        '''
        Initialize the edge mask.
        '''
        base_mask = torch.ones(self.original.num_edges)

        if setup['random'] == True:
            base_factor = torch.Tensor(np.random.random(self.original.num_edges))
        else:
            base_factor = 1.

        base_factor = base_factor + setup['offset']

        edge_mask = base_mask * base_factor * setup['modulating_factor']
        edge_mask[self.orig_num_edges:] *= 0

        if setup['train'] == True:
            edge_mask.requires_grad = True

        return edge_mask

    def add_additional_edges(self, r, p):
        '''
        Add additional edges to the lattice randomly.
        The newly added edges are initially masked out.
        '''
        edge_list = self.original.edges.tolist()

        for snode in range(self.original.num_nodes):
            # only allow new edges between nodes within a certain radius
            candidates = list(set(list(np.where(np.sqrt(np.sum((self.original.coordinates - self.original.coordinates[snode])**2, axis=1)) <= r)[0])).difference(set([snode])))

            for i in candidates:
                edge_crosses_node = False
                for k in range(self.original.num_nodes):
                    if k != snode and k != i:
                        # check that an edge is not crossing a node
                        if isBetween(self.original.coordinates[snode], self.original.coordinates[i], self.original.coordinates[k]):
                            edge_crosses_node = True
                            break
                # only add a new edge if it does not cross a node and is not already in the edge list
                if edge_crosses_node == False and [snode,i] not in edge_list and [i, snode] not in edge_list:
                    # to avoid a high amount of edges: allow for sparsification by randomly sampling
                    if np.random.random() <= p:
                        edge_list.append([snode,i])

        self.original.edges = np.array(edge_list)

    def get_edge_constraints(self):
        '''
        Calculate which edges in the graph/lattice cross each other.
        '''
        print('Checking for crossing edges...')
        edge_constraints = defaultdict(set)

        for k in trange(len(self.original.edges)):
            for i in range(k, self.original.num_edges):
                if i == k:
                    pass
                else:
                    two_edges = np.array(self.original.edges)[[k,i]]
                    if (two_edges[0] != two_edges[1]).all():
                        if no_crossings(two_edges, np.array(self.original.coordinates)) == False:
                            edge_constraints[k].add(i)
                            edge_constraints[i].add(k)

        return edge_constraints
    
    def update_edge_constraints(self):
        '''
        Update the dictionary containing crossing edges (e.g., after moving nodes).
        '''
        print('Checking for crossing edges...')
        edge_constraints = defaultdict(set)

        for k in trange(len(self.original.edges)):
            for i in range(k, self.original.num_edges):
                if i == k:
                    pass
                else:
                    two_edges = np.array(self.original.edges)[[k,i]]
                    if (two_edges[0] != two_edges[1]).all():
                        if no_crossings(two_edges, self.coordinates.detach().numpy()) == False:
                            edge_constraints[k].add(i)
                            edge_constraints[i].add(k)

        self.edge_constraints = edge_constraints

    def set_constraints(self, experiment_setup):
        '''
        Set constraints of the FE simulation.
        '''
        self.Constrainer.reset_constraints()

        constraints_to_add = defaultdict(dict)
        for i in experiment_setup['forced_nodes']:
            constraints_to_add[i] = {'x': True, 'y': True}
        for i in experiment_setup['static_nodes']:
            constraints_to_add[i] = {'x': True, 'y': True}

        self.Constrainer.add_constraints_from_dict(constraints_to_add)

    def set_displacements(self, experiment_setup):
        '''
        Set external displacements for the FE simulation.
        '''
        self.Displacer.reset_displacement()

        for nodes in experiment_setup['forced_nodes']:
            self.Displacer.set_displacement(nodes, y=-experiment_setup['displacement']) 
            
    def draw_lattice(self, dr=0, original_coords = True, path = None):
        if type(dr) == torch.Tensor:
            disp = dr.detach().numpy()
        else:
            disp = dr
        if original_coords == True:
            coords = self.original.coordinates
        else:
            coords = self.coordinates.detach().numpy()
        draw_graph(self.active_edges, coords+disp, node_size=0, numbers=False, path = path)


    def draw_edge_mask(self, colormap = 'Oranges'):
        if self.edge_mask is not None:
            cmap = matplotlib.cm.get_cmap(colormap)

            fig = plt.figure()

            plt.vlines(-0.5, -0.5,0.5, alpha=0.75, color = 'lightsteelblue', linestyle = '--')
            plt.vlines(0.5, -0.5,0.5, alpha=0.75, color = 'lightsteelblue', linestyle = '--')

            plt.hlines(-0.5, -0.5,0.5, alpha=0.75, color = 'lightsteelblue', linestyle = '--')
            plt.hlines(0.5, -0.5,0.5, alpha=0.75, color = 'lightsteelblue', linestyle = '--')


            colors = self.active_mask.detach().numpy()
            colors = colors + np.min(colors)
            colors = colors/np.max(colors)
            edge_list = self.active_edges

            count = 0
            for i,j in edge_list:
                plt.plot([self.original.coordinates[i][0], self.original.coordinates[j][0]],\
                         [self.original.coordinates[i][1], self.original.coordinates[j][1]], color = cmap(colors[count]), zorder=1)
                count += 1

            plt.xticks([])
            plt.yticks([])
            sns.despine(fig=fig, bottom=True, left=True, top=True, right=True)
