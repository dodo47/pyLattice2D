from pyLattice2D.fem_solver.utils import scalar_to_tensor, get_real_mask
from pyLattice2D.lattices.utils import get_BeamShapeFactor
from pyLattice2D.fem_solver.constraints_and_deformations import ExternalForces
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import dgl.function as fn

class StiffnessMatrix(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, graph, coordinates, E, A, threshold_function = None, edge_mask=None, edge_constraints=None):
        '''
        Given a lattice specification, return its stiffness matrix in differentiable form.

        Args:
           graph: Graph object representing the lattice.
           coordinates: Node coordinates of the graph.
           E: Young's modulus
           A: Beam area
           threshold function: Surrogate used for automatic differentiation.
           edge_mask: Masking values for edges (used to add/remove beams)
           edge_constraints: Dictionary of beams that would cross, i.e., cannot be added at the same time. 
        '''
        with graph.local_scope():
            # get node distances
            graph.ndata['coordinates'] = coordinates
            graph.apply_edges(fn.u_sub_v('coordinates', 'coordinates', 'delta'))
            E = scalar_to_tensor(E, len(graph.edges()[0]))
            A = scalar_to_tensor(A, len(graph.edges()[0]))
            I = get_BeamShapeFactor(A)

            if edge_mask is not None:
                assert(threshold_function is not None)
                emask = threshold_function(edge_mask)
            else:
                emask = 1.
            if edge_constraints is not None and edge_mask is not None:
                emask = get_real_mask(emask, edge_constraints, len(graph.edges()[0]))
            # calculate rod lengths and stiffness constants
            L = torch.norm(graph.edata['delta'], dim=1)
            graph.edata['krot'] = (E*I/L**3)*emask
            graph.edata['klin'] = (E*A/L)*emask

            # calculate angles
            cos = (graph.edata['delta'][:,0]/L).view(-1,1)
            sin = -(graph.edata['delta'][:,1]/L).view(-1,1)
            L = L.view(-1,1)

            # factors that appear many times in the stiffness matrix
            sinsq = sin**2
            cossq = cos**2
            sincos = sin*cos
            Lsin = 6*L*sin
            Lcos = 6*L*cos
            L2 = 2*L**2
            L4 = 4*L**2

            Kmat_rot = torch.cat([12*sinsq,  12*sincos, -Lsin, -12*sinsq,   -12*sincos, -Lsin,\
                                 12*sincos,  12*cossq,   -Lcos,  -12*sincos, -12*cossq,  -Lcos,\
                                 -Lsin,       -Lcos,       L4,    Lsin,       Lcos,      L2,\
                                 -12*sinsq,   -12*sincos,  Lsin,  12*sinsq,  12*sincos,  Lsin,\
                                  -12*sincos, -12*cossq,  Lcos, 12*sincos,  12*cossq,  Lcos,\
                                 -Lsin,       -Lcos,      L2,     Lsin,       Lcos,      L4],
                                 dim=1)

            # apply prefactor
            Kmat_rot = Kmat_rot*graph.edata['krot'].view(-1,1)

            # Truss rod element
            zeros = torch.zeros((len(graph.edges()[0]),1))
            Kmat_lin = torch.cat([cossq,  -sincos, zeros, -cossq,  sincos, zeros,\
                                  -sincos, sinsq,  zeros, sincos, -sinsq,  zeros,\
                                  zeros,  zeros,  zeros,  zeros,   zeros,  zeros,\
                                 -cossq, sincos, zeros,  cossq,   -sincos, zeros,\
                                 sincos,-sinsq,  zeros,  -sincos,  sinsq,  zeros,\
                                  zeros,  zeros,  zeros,  zeros,   zeros,  zeros],
                                 dim=1)
            # apply prefactor
            Kmat_lin = Kmat_lin*graph.edata['klin'].view(-1,1)

            # add both to get generalised beam element
            Kmat = Kmat_rot+Kmat_lin
            # reshape to get 6x6 matrix for each beam element
            Kmat = Kmat.view(-1,6,6)

        # combine stiffness matrices of individual beams
        # to get the full stiffness matrix
        fullmat = torch.zeros((graph.num_nodes()*3, graph.num_nodes()*3))
        for i in range(len(graph.edges()[0])):
            n0, n1 = graph.edges()[0][i], graph.edges()[1][i]
            edgeK = Kmat[i]
            for j in range(3):
                fullmat[n0*3+j, n0*3:(n0+1)*3] += edgeK[j][:3]
                fullmat[n0*3+j, n1*3:(n1+1)*3] += edgeK[j][3:]
            for j in range(3):
                fullmat[n1*3+j, n0*3:(n0+1)*3] += edgeK[j+3][:3]
                fullmat[n1*3+j, n1*3:(n1+1)*3] += edgeK[j+3][3:]

        return fullmat

class DirectStiffnessSolver(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, lattice, stiffness, experiment_setup):
        '''
        Given a lattice and its stiffness matrix, perform a single compression step.

        Returns the deformations (dr) and stress.
        '''
        lattice.set_constraints(experiment_setup)
        lattice.set_displacements(experiment_setup)
        internal_forces = lattice.Displacer.get_internal_forces(stiffness)

        forces_to_set = defaultdict(dict)
        for i in range(lattice.original.num_nodes):
            forces_to_set[i] = {'Fx': internal_forces[i*3], 'Fy': internal_forces[i*3+1], 'Mphi': internal_forces[i*3+2]}
        forces = ExternalForces(lattice.original.num_nodes, forces_to_set)

        reduced_stiffness = lattice.Constrainer.transform_stiffness_matrix(stiffness)
        reduced_force = lattice.Constrainer.transform_force_vector(forces.force_vector)

        disp = torch.linalg.solve(reduced_stiffness, reduced_force)
        assert(np.isnan(disp.detach().numpy()).any()==False)

        disp = lattice.Constrainer.reverse_transform_displacements(disp)+lattice.Displacer.displacement
        dr = torch.reshape(disp, (lattice.original.num_nodes, 3))[:,:2]

        top_force = torch.matmul(stiffness, disp)
        top_force = torch.reshape(top_force, (lattice.original.num_nodes, 3))[experiment_setup['forced_nodes']][:,1]
        stress = -torch.sum(top_force)/np.sqrt(lattice.BeamCrossArea_A)

        return dr, stress