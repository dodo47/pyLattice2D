from pyLattice2D.fem_solver.direct_stiffness import StiffnessMatrix, DirectStiffnessSolver
import torch.nn as nn

class FEModel(nn.Module):
    def __init__(self):
        '''
        Given a differentiable lattice, performs a finite element experiment.
        '''
        super().__init__()
        self.stiffness = StiffnessMatrix()
        self.update = DirectStiffnessSolver()

    def forward(self, lattice, experiment_setup, delta = 0):
        stiff = self.stiffness(lattice.graph, lattice.coordinates+delta, lattice.YoungsModulus_E, lattice.BeamCrossArea_A, lattice.threshold_function, lattice.edge_mask, lattice.edge_constraints)
        dr, stress = self.update(lattice, stiff, experiment_setup)

        return dr, stress
