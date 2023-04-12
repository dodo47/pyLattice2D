import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from pyLattice2D.utils.math import L1_dist
from pyLattice2D.utils.draw import draw_graph
from pyLattice2D.utils.validity import is_graph_valid
import random
import sys
eps = sys.float_info.epsilon/2

class Lattice:
    def __init__(self, edges = None, coordinates = None, unperturbed_coordinates = None):
        '''
        Initial lattice object.

        A lattice is specified here by its node coordinates and edges (beams).
        '''
        self.edges = edges
        self.coordinates = coordinates
        if unperturbed_coordinates is None:
            self._unperturbed_coordinates = deepcopy(coordinates)
        else:
            self._unperturbed_coordinates = unperturbed_coordinates
    
    @property
    def num_nodes(self):
        '''
        Return the number of nodes.
        '''
        return len(self.coordinates)

    @property
    def num_edges(self):
        '''
        Return the number of edges.
        '''
        return len(self.edges)
    
    @property
    def num_cells(self):
        '''
        Return the number of closed cells in the lattice.
        '''
        return self.num_edges - self.num_nodes + 1

    @property
    def density_factor(self):
        '''
        Return the relative density factor (without beam area) of the lattice. 
        '''
        beam_length = 0
        for i,j in self.edges:
            beam_length += np.sqrt(np.sum((self.coordinates[i]-self.coordinates[j])**2))
        material_height = np.max(self.coordinates[:,1])-np.min(self.coordinates[:,1])
        material_width = np.max(self.coordinates[:,0])-np.min(self.coordinates[:,0])
        material_area = material_height*material_width

        return beam_length/material_area

    @property
    def two_outer_left_columns(self):
        '''
        Return the node IDs of the nodes that make up the two outer left columns of the material.
        '''
        inner_column = list(np.where(np.fabs(self._unperturbed_coordinates[:,0] - np.sort(list(set(list(self._unperturbed_coordinates[:,0]))))[1])<1e-6)[0])
        all_nodes = inner_column + self.left_nodes

        return all_nodes
    
    @property
    def two_outer_right_columns(self):
        '''
        Return the node IDs of the nodes that make up the two outer right columns of the material.
        '''
        inner_column = list(np.where(np.fabs(self._unperturbed_coordinates[:,0] -np.sort(list(set(list(self._unperturbed_coordinates[:,0]))))[-2])<1e-6)[0])
        all_nodes = inner_column + self.right_nodes
        
        return all_nodes
    
    @property
    def outer_nodes(self):
        '''
        Return the node IDs of all nodes that are part of the outer surface of the material.
        '''
        return self.top_nodes+self.bottom_nodes+self.right_nodes+self.left_nodes
    
    @property
    def top_nodes(self):
        '''
        Return the node IDs of all nodes that are part of the top layer.
        '''
        ymax = np.max(self.coordinates[:,1])
        top_nodes = []
        for i in range(self.num_nodes):
            if L1_dist(self.coordinates[i][1], ymax) <= eps:
                top_nodes.append(i)
        return top_nodes

    @property
    def bottom_nodes(self):
        '''
        Return the node IDs of all nodes that are part of the bottom layer.
        '''
        ymin = np.min(self.coordinates[:,1])
        bottom_nodes = []
        for i in range(self.num_nodes):
            if L1_dist(self.coordinates[i][1], ymin) <= eps:
                bottom_nodes.append(i)
        return bottom_nodes

    @property
    def right_nodes(self):
        '''
        Return the node IDs of all nodes that are part of the top layer.
        '''
        xmax = np.max(self.coordinates[:,0])
        right_nodes = []
        for i in range(self.num_nodes):
            if L1_dist(self.coordinates[i][0], xmax) <= eps:
                right_nodes.append(i)
        return right_nodes

    @property
    def left_nodes(self):
        '''
        Return the node IDs of all nodes that are part of the top layer.
        '''
        xmin = np.min(self.coordinates[:,0])
        left_nodes = []
        for i in range(self.num_nodes):
            if L1_dist(self.coordinates[i][0], xmin) <= eps:
                left_nodes.append(i)
        return left_nodes

    @property
    def valid(self):
        '''
        Return whether the graph is a valid graph (i.e, no edges are crossing, no disconnected nodes).
        '''
        return is_graph_valid(self.edges, self.coordinates)

    def draw(self, dx = 0, path = None):
        draw_graph(self.edges, self.coordinates+dx, node_size=0, numbers=False, path = path)

class Base:
    def __init__(self, num_layers, box_width, rotate = False, norm_width = False, seed = 42424242):
        '''
        Base class for all lattice tilings.

        Creates a lattice in a unit box (contains scaling functions).
        Also implements perturbations of the lattice, e.g., node deletions, node movements, edge deletions, etc.

        To implement different base tilings, the functions
            - self.create_base_lattice()
            - self.transform_base_lattice()
            - self.reentrant_transformation()
        have to be specified accordingly.
        '''
        np.random.seed(seed)
        random.seed(seed)
        self.num_layers = num_layers
        self.norm_width = norm_width
        self.box_width = box_width
        self.rotate = rotate

        # add lattice
        self.lattice = Lattice()

        # transform lattice to create desired base tilings from square lattice
        self.create_base_lattice()
        self.transform_base_lattice()
        self.reentrant_transformation()

        if rotate == True:
            self._rotate_base()

        # normalize lattice coordinates so the lattice is centered around 0
        self._center_base()
        # memorize the unperturbed (original) coordinates of the lattice (e.g., to calculate the outer nodes)
        self.lattice._unperturbed_coordinates = deepcopy(self.lattice.coordinates)


    @property
    def perturbable_nodes(self):
        '''
        Returns nodes that are not on the outer surface of the lattice.
        '''
        perturbable_nodes = set(range(self.lattice.num_nodes))
        outer_nodes = set(self.lattice.bottom_nodes).union(set(self.lattice.top_nodes))
        outer_nodes = outer_nodes.union(set(self.lattice.left_nodes)).union(set(self.lattice.right_nodes))
        perturbable_nodes = list(perturbable_nodes.difference(outer_nodes))
        return perturbable_nodes

    def create_base_lattice(self):
        pass

    def transform_base_lattice(self):
        pass

    def reentrant_transformation(self):
        pass

    def _rescale_base(self):
        scale_factor = np.max(self.lattice.coordinates[:,1]) - np.min(self.lattice.coordinates[:,1])
        self.lattice.coordinates /= scale_factor

    def _rescale_width(self):
        scale_factor = np.max(self.lattice.coordinates[:,0]) - np.min(self.lattice.coordinates[:,0])
        self.lattice.coordinates[:,0] /= scale_factor

    def _rescale_height(self):
        scale_factor = np.max(self.lattice.coordinates[:,1]) - np.min(self.lattice.coordinates[:,1])
        self.lattice.coordinates[:,1] /= scale_factor

    def _center_base(self):
        self.lattice.coordinates[:,0] -= (np.max(self.lattice.coordinates[:,0])+np.min(self.lattice.coordinates[:,0]))/2
        self.lattice.coordinates[:,1] -= (np.max(self.lattice.coordinates[:,1])+np.min(self.lattice.coordinates[:,1]))/2

    def _rotate_base(self):
        self.lattice.coordinates = self.lattice.coordinates[:, ::-1]
        self._rescale_base()

    def draw_lattice(self, path= None):
        self.lattice.draw(path=path)

    def delete_edges(self, fraction):
        '''
        Randomly delete edges.
        '''
        num_edges_to_delete = int(fraction*self.lattice.num_edges)
        self.lattice.edges = list(self.lattice.edges)
        for i in range(num_edges_to_delete):
            del(self.lattice.edges[np.random.randint(self.lattice.num_edges)])

    def add_random_edge(self, min_distance):
        '''
        Ramdomly add an edge with a certain minimum length.
        '''
        iterations = 0
        self.lattice.edges = list(self.lattice.edges)
        while iterations < 1e4:
            dist = 0
            while dist < min_distance:
                n1 = np.random.randint(self.lattice.num_nodes)
                n2 = np.random.randint(self.lattice.num_nodes)
                dist = np.sqrt(np.sum((self.lattice.coordinates[n1]-self.lattice.coordinates[n2])**2))
            self.lattice.edges.append([n1,n2])
            if is_graph_valid(self.lattice.edges, self.lattice.coordinates):
                break
            else:
                del(self.lattice.edges[-1])

    def randomly_delete_nodes(self, num_nodes_to_delete = 1):
        '''
        Randomly delete 'num_modes_to_delete' nodes.
        '''
        for i in range(num_nodes_to_delete):
            node_ID = random.sample(self.perturbable_nodes, 1)[0]
            self.delete_node(node_ID)

    def move_nodes(self, std, fraction=0.2, mode='uniform'):
        '''
        Randomly move a fraction of the nodes with strength 'std'.
        '''
        perturbable_nodes = self.perturbable_nodes
        num_nodes_to_perturb = int(fraction*len(perturbable_nodes))

        nodes_to_perturb = random.sample(perturbable_nodes, num_nodes_to_perturb)

        if mode == 'normal':
            for node in nodes_to_perturb:
                self.lattice.coordinates[node] += np.random.normal(0,std, np.shape(self.lattice.coordinates[node]))
        elif mode == 'uniform':
            for node in nodes_to_perturb:
                self.lattice.coordinates[node] += (np.random.random(np.shape(self.lattice.coordinates[node]))-0.5)*std

    def delete_node(self, node_id, node_list = None, edge_list = None):
        '''
        Delete node with ID 'node_id'.
        '''
        if node_list is None:
            node_list = self.lattice.coordinates
        if edge_list is None:
            edge_list = self.lattice.edges

        # filter out edges connected with node
        edge_ids = []
        for i in range(self.lattice.num_edges):
            if node_id not in self.lattice.edges[i]:
                edge_ids.append(i)
        new_edges = self.lattice.edges[edge_ids]

        # shift node IDs in edge list
        new_edges[new_edges>node_id] -= 1

        # remove coordinates of node
        new_nodes = self.lattice.coordinates[[i for i in range(len(node_list)) if i != node_id]]
        new_nodes = np.array(new_nodes)

        self.lattice.edges = new_edges
        self.lattice.coordinates = new_nodes
