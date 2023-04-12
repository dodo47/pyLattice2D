from sortedcontainers import SortedSet
import numpy as np
import torch

class Constraints:
    def __init__(self, num_nodes, constraints_to_add = None):
        '''
        Class for adding node constraints to the stiffness matrix
        and external force vector.
        '''
        self.number_DoF = num_nodes*3
        self._free_DoF = None
        self.reset_constraints()

        if constraints_to_add is not None:
            self.add_constraints_from_dict(constraints_to_add)

    @property
    def constrained_DoF(self):
        return list(SortedSet(np.arange(self.number_DoF)).difference(self._free_DoF))

    @property
    def free_DoF(self):
        return list(self._free_DoF)

    def add_constraint(self, node, x=False, y=False, phi=False):
        if x == True:
            self._free_DoF.remove(node*3)
        if y == True:
            self._free_DoF.remove(node*3+1)
        if phi == True:
            self._free_DoF.remove(node*3+2)

    def add_constraints_from_dict(self, dictionary):
        for nodes in dictionary.keys():
            self.add_constraint(nodes, **dictionary[nodes])

    def reset_constraints(self):
        self._free_DoF = SortedSet(np.arange(self.number_DoF))

    def transform_force_vector(self, force):
        return force[self.free_DoF]

    def transform_stiffness_matrix(self, stiffness):
        return stiffness[self.free_DoF][:,self.free_DoF]

    def reverse_transform_displacements(self, displacements):
        new_displacement = torch.zeros(self.number_DoF)
        new_displacement[self.free_DoF] = displacements
        return new_displacement

class ExternalForces:
    def __init__(self, num_nodes, forces_to_set = None):
        '''
        Class for creating force vector for the direct stiffness method.
        '''
        self.number_DoF = num_nodes*3
        self.reset_force()

        if forces_to_set is not None:
            self.set_forces_from_dict(forces_to_set)

    def set_force(self, node, Fx=0, Fy=0, Mphi=0):
        self.force_vector[node*3] = Fx
        self.force_vector[node*3+1] = Fy
        self.force_vector[node*3+2] = Mphi

    def set_forces_from_dict(self, dictionary):
        for nodes in dictionary.keys():
            self.set_force(nodes, **dictionary[nodes])

    def reset_force(self):
        self.force_vector = torch.zeros(self.number_DoF)

class ExternalDisplacement:
    def __init__(self, num_nodes, disp_to_set = None):
        '''
        Class for creating force vector for the direct stiffness method.
        '''
        self.number_DoF = num_nodes*3
        self.reset_displacement()

        if disp_to_set is not None:
            self.set_displacement_from_dict(disp_to_set)

    def set_displacement(self, node, x=0, y=0, phi=0):
        self.displacement[node*3] = x
        self.displacement[node*3+1] = y
        self.displacement[node*3+2] = phi

    def set_displacement_from_dict(self, dictionary):
        for nodes in dictionary.keys():
            self.set_displacement(nodes, **dictionary[nodes])

    def reset_displacement(self):
        self.displacement = torch.zeros(self.number_DoF)

    def get_internal_forces(self, stiffness):
        return -torch.matmul(stiffness, self.displacement)
