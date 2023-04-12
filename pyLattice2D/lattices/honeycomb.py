from pyLattice2D.lattices.triangle import TriangleLattice
import numpy as np

class HoneycombLattice(TriangleLattice):
    def __init__(self, removing = True, **kwargs):
        '''
        Implementation of Honeycomb tiling.
        '''
        self.removing = removing
        super(HoneycombLattice, self).__init__(**kwargs)

    def transform_base_lattice(self):
        num_nodes_deleted = 0
        offset = True

        for row in range(self.num_layers):
            for col in range(self.num_columns+1*(row%2==0)):
                if row == 0:
                    previous_nodes = 0
                elif row%2 == 0:
                    previous_nodes = int(row*(self.num_columns+0.5))
                else:
                    previous_nodes = int(int((row+1)/2)*(self.num_columns+1)+int(row/2)*self.num_columns)
                nodeid = previous_nodes+col
                if row%2 == 1 and (col-1)%3 == 0:
                    self.delete_node(nodeid-num_nodes_deleted)
                    num_nodes_deleted += 1
                if row%2 == 0 and col%3 == 0:
                    self.delete_node(nodeid-num_nodes_deleted)
                    num_nodes_deleted += 1
        if self.removing == True:
            super(HoneycombLattice, self)._remove_parts()

    def _remove_parts(self):
        if self.removing == True:
            pass
        else:
            super(HoneycombLattice, self)._remove_parts()


class KagomeLattice(TriangleLattice):
    def __init__(self, **kwargs):
        '''
        Implementation of Kagome tiling.
        '''
        super(KagomeLattice, self).__init__(**kwargs)

    def transform_base_lattice(self):
        num_nodes_deleted = 0
        offset = True

        for row in range(self.num_layers):
            for col in range(self.num_columns+1*(row%2==0)):
                if row == 0:
                    previous_nodes = 0
                elif row%2 == 0:
                    previous_nodes = int(row*(self.num_columns+0.5))
                else:
                    previous_nodes = int(int((row+1)/2)*(self.num_columns+1)+int(row/2)*self.num_columns)
                nodeid = previous_nodes+col
                if row%2 == 1 and col%2 == offset:
                    self.delete_node(nodeid-num_nodes_deleted)
                    num_nodes_deleted += 1
            if row%2 == 1:
                offset = bool(1-offset)

        super(KagomeLattice, self)._remove_parts()

    def _remove_parts(self):
        pass

class ReentrantHoneycombLattice(HoneycombLattice):
    def __init__(self, **kwargs):
        '''
        Implementation of Reentrant Honeycomb tiling.
        '''
        assert(kwargs['num_layers'] not in self.forbidden_layers)
        super(ReentrantHoneycombLattice, self).__init__(check_length = False, removing = False, **kwargs)

    def reentrant_transformation(self):
        new_edges = []
        num_nodes = len(self.lattice.coordinates)
        num_bot_nodes = np.sum(self.lattice.coordinates[:,1]==0)

        for i,j in self.lattice.edges:
            if self.lattice.coordinates[i][1] == self.lattice.coordinates[j][1]:
                if j+1 < num_nodes:
                    if (j+1)%(num_bot_nodes) == 0:
                        dx1 = self.lattice.coordinates[j+2][0] - self.lattice.coordinates[j+1][0]
                        dx2 = self.lattice.coordinates[j][0] - self.lattice.coordinates[j-1][0]
                        if np.fabs(dx1 - dx2) > 1e-6:
                            new_edges.append([j+1, j+2])
                    else:
                        new_edges.append([j, j+1])
            else:
                new_edges.append([i,j])

        for i in range(num_bot_nodes-1, num_nodes, 2*num_bot_nodes):
            dx1 = self.lattice.coordinates[i][0] - self.lattice.coordinates[i-1][0]
            dx2 = self.lattice.coordinates[i-1][0] - self.lattice.coordinates[i-2][0]
            if dx1 > dx2:
                if i+2 < num_nodes:
                    new_edges.append([i+1, i+2])

        self.lattice.edges = np.array(new_edges)
        super(ReentrantHoneycombLattice, self)._remove_parts()

    @property
    def forbidden_layers(self):
        forbidden_layers = [0,1,2,3,7]
        layer = 7
        for i in range(30):
            if i%2 == 0:
                layer += 3
            else:
                layer += 4
            forbidden_layers.append(layer)
        return forbidden_layers

    def _remove_parts(self):
        pass
