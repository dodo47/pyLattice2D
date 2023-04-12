import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl
from dgl.nn.pytorch.conv import EdgeConv
from dgl.nn.pytorch.conv import NNConv
from dgl.nn import Set2Set
from pyLattice2D.fem_solver.utils import super_fn
from pyLattice2D.models.MPNN.MaskedConv import EdgeConvMasked

class LatticeGNN(nn.Module):
    '''
    GNN based on EdgeConv.
    '''
    def __init__(self, hid_nfeat):
        super().__init__()
        # neural network layers
        self.GNN1 = EdgeConv(2, hid_nfeat)
        self.GNN2 = EdgeConv(hid_nfeat, hid_nfeat)
        self.GNN3 = EdgeConv(hid_nfeat, hid_nfeat)
        
        self.l = nn.Linear(hid_nfeat, 400)
        self.l2 = nn.Linear(400, 200)
        self.out = nn.Linear(200, 1)
        
    def forward(self, g):
        # get intiial edge features
        node_features = g.ndata['coords']
        node_features = self.GNN1(g, node_features)
        node_features = self.GNN2(g, node_features)
        node_features = self.GNN3(g, node_features)
        
        node_features = F.relu(self.l(node_features))
        node_features = F.relu(self.l2(node_features))
        node_features = self.out(node_features)
        g.ndata['hoo'] = node_features
        output = dgl.mean_nodes(g, 'hoo').flatten()
        return output
    
class LatticeGNNMasked(nn.Module):
    '''
    GNN based on EdgeConv with masking of edges enabled.
    '''
    def __init__(self, hid_nfeat):
        super().__init__()
        # neural network layers
        self.GNN1 = EdgeConvMasked(2, hid_nfeat)
        self.GNN2 = EdgeConvMasked(hid_nfeat, hid_nfeat)
        self.GNN3 = EdgeConvMasked(hid_nfeat, hid_nfeat)
        
        self.l = nn.Linear(hid_nfeat, 400)
        self.l2 = nn.Linear(400, 200)
        self.out = nn.Linear(200, 1)
        
    def forward(self, g, mask = None):
        # get intiial edge features
        node_features = g.ndata['coords']
        node_features = self.GNN1(g, node_features, mask)
        node_features = self.GNN2(g, node_features, mask)
        node_features = self.GNN3(g, node_features, mask)
        
        node_features = F.relu(self.l(node_features))
        node_features = F.relu(self.l2(node_features))
        node_features = self.out(node_features)
        g.ndata['hoo'] = node_features
        output = dgl.mean_nodes(g, 'hoo').flatten()
        return output
    
class LatticeNNConv(nn.Module):
    '''
    Message passing neural network architecture.
    '''
    def __init__(self, hid_nfeat, num_message_passing):
        super().__init__()
        self.num_message_passing = num_message_passing

        self.preprocess = nn.Sequential(
            nn.Linear(2, int(hid_nfeat/2)),
            nn.ReLU(),
            nn.Linear(int(hid_nfeat/2), hid_nfeat),
        )    
        
        edge_net = nn.Sequential(
            nn.Linear(4, int(hid_nfeat/4)),
            nn.ReLU(), 
            nn.Linear(int(hid_nfeat/4), int(hid_nfeat/2)),
            nn.ReLU(),
            nn.Linear(int(hid_nfeat/2), hid_nfeat),
            nn.ReLU(), 
            nn.Linear(hid_nfeat, hid_nfeat * hid_nfeat)
        )
        
        self.GNN = NNConv(hid_nfeat, hid_nfeat, edge_net)
        self.gru  = nn.GRU(hid_nfeat, hid_nfeat)
        
        self.predict = nn.Sequential(
            nn.Linear(2*hid_nfeat, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )    
        
        self.set2set = Set2Set(hid_nfeat, 6, 1)
        
    def forward(self, g):
        # get initial edge features
        x = g.ndata['coords']
        g.apply_edges(fn.v_sub_u('coords', 'coords', 'theta'))
        L = torch.norm(g.edata['theta'], dim=1)
        cos = (g.edata['theta'][:,0]/L).view(-1,1)
        edge_features = torch.cat([g.edata['theta'], cos, L.view(-1,1)], dim=1)
        
        x = F.relu(self.preprocess(x))
        h = x.unsqueeze(0)
        for i in range(self.num_message_passing):
            m = F.relu(self.GNN(g, x, edge_features))
            x, h = self.gru(m.unsqueeze(0), h)
            x = x.squeeze(0)
            
        x = self.set2set(g, x)
        x = self.predict(x)
      
        return x.flatten()
    
class CNN(nn.Module):
    def __init__(self):
        '''
        Convolutional neural network architecture.
        '''
        super().__init__()
        # create network layers
        # convolutions
        self.conv1 = nn.Conv2d(1, 40, kernel_size=6, stride=1)
        self.conv2 = nn.Conv2d(40, 20, 6, 1)
        self.conv3 = nn.Conv2d(20, 10, 4, 2)
        # max pooling
        self.pool = nn.MaxPool2d(3, 3)
        # dropout
        self.dropout = nn.Dropout(p=0.05)
        # MLP (dense layers)
        self.fc1 = nn.Linear(350, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        '''
        Apply network to input x.
        '''
        # first convolutional layer
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        # second convolutional layer
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        # third convolutional layer
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        # first dense layer
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # output layer
        x = self.fc2(x)
        return x