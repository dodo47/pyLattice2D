from pyLattice2D.lattices.base import Base
import numpy as np

class SquareLattice(Base):
    def __init__(self, **kwargs):
        '''
        Implementation of Square tiling.
        '''
        super(SquareLattice, self).__init__(**kwargs)

    def create_base_lattice(self):
        self.num_columns = self.num_layers

        coordinates = []
        for row in range(self.num_layers):
            for col in range(self.num_columns):
                coordinates.append([col,row])
        self.lattice.coordinates = np.array(coordinates)*1.
        self._rescale_base()

        edges = []
        for row in range(self.num_layers):
            for col in range(self.num_columns):
                nodeid = row*self.num_columns+col
                if col < self.num_columns-1:
                    edges.append([nodeid, nodeid+1])
                if row < self.num_layers-1:
                    edges.append([nodeid, nodeid+self.num_columns])
        self.lattice.edges = np.array(edges, dtype=int)
