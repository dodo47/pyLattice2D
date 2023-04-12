from pyLattice2D.lattices.base import Base
import numpy as np

class TriangleLattice(Base):
    def __init__(self, row_offset = 0, check_length = True, **kwargs):
        '''
        Implementation of Equilateral Triangle tiling.
        '''
        self.row_offset = row_offset
        super(TriangleLattice, self).__init__(**kwargs)
        if check_length == True:
            self.check_edge_lengths()

    def create_base_lattice(self):
        height_correction = np.sqrt(1-0.5**2)
        self.num_columns = int(np.round(self.box_width*height_correction*(self.num_layers-1)))

        self.num_layers += self.row_offset

        coordinates = []
        for row in range(self.num_layers):
            for col in range(self.num_columns+1*(row%2==0)):
                coordinates.append([(col+0.5*(row%2)), row*height_correction])
        self.lattice.coordinates = np.array(coordinates)*1.

        edges = []
        for row in range(self.num_layers):
            for col in range(self.num_columns+1*(row%2==0)):
                if row == 0:
                    previous_nodes = 0
                elif row%2 == 0:
                    previous_nodes = int(row*(self.num_columns+0.5))
                else:
                    previous_nodes = int(int((row+1)/2)*(self.num_columns+1)+int(row/2)*self.num_columns)
                nodeid = previous_nodes+col
                if col < self.num_columns+1*(row%2==0)-1:
                    edges.append([nodeid, nodeid+1])
                if row < self.num_layers-1 and row%2!=0:
                    edges.append([nodeid, nodeid+self.num_columns])

                if row < self.num_layers-1 and row%2==0 and col != 0:
                    edges.append([nodeid, nodeid+self.num_columns])

                if row%2 == 0 and col < self.num_columns and row < self.num_layers - 1:
                    edges.append([nodeid, nodeid+self.num_columns+1])

                if row%2 == 1 and col < self.num_columns and row < self.num_layers-1:
                    edges.append([nodeid, nodeid+self.num_columns+1])

        self.lattice.edges = np.array(edges, dtype=int)

        self._remove_parts()

    def _remove_parts(self):
        for row in range(self.row_offset):
            bottom_val = np.min(self.lattice.coordinates[:,1])
            num_bottom_nodes = np.sum(self.lattice.coordinates[:,1]<=bottom_val)
            for col in range(num_bottom_nodes):
                self.delete_node(0)

        self._rescale_base()

    def check_edge_lengths(self, precision = 10):
        lengths = []
        for i,j in self.lattice.edges:
            squared_length = np.sum((self.lattice.coordinates[i]-self.lattice.coordinates[j])**2)
            squared_length = np.round(squared_length, precision)
            lengths.append(squared_length)
        assert(len(set(lengths))==1)
