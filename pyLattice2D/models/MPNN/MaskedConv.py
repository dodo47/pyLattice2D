from torch import nn
from dgl.base import DGLError
from dgl import function as fn
from dgl.utils import expand_as_pair
import torch

class EdgeConvMasked(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 batch_norm=False,
                 allow_zero_in_degree=False):
        '''
        Torch Module for EdgeConv with edge masking, adjusted from DGL library.
        (https://docs.dgl.ai/en/0.8.x/generated/dgl.nn.pytorch.conv.EdgeConv.html)
        '''
        super().__init__()
        self.batch_norm = batch_norm
        self._allow_zero_in_degree = allow_zero_in_degree

        self.theta = nn.Linear(in_feat, out_feat)
        self.phi = nn.Linear(in_feat, out_feat)

        if batch_norm:
            self.bn = nn.BatchNorm1d(out_feat)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, g, feat, mask):
        h_src, h_dst = expand_as_pair(feat, g)
        g.srcdata['x'] = h_src
        g.dstdata['x'] = h_dst
        g.apply_edges(fn.v_sub_u('x', 'x', 'theta'))
        g.edata['theta'] = self.theta(g.edata['theta'])
        g.dstdata['phi'] = self.phi(g.dstdata['x'])
        
        g.apply_edges(fn.e_add_v('theta', 'phi', 'e'))
        if mask is not None:
            mask_size = 2*mask.size()[0]
            # multiplicative masking
            g.edata['e'][:mask_size][::2] = g.edata['e'][:mask_size][::2].clone()*mask + (1-mask)*torch.min(g.edata['e'].clone())
            g.edata['e'][:mask_size][1::2] = g.edata['e'][:mask_size][1::2].clone()*mask + (1-mask)*torch.min(g.edata['e'].clone())
        g.update_all(fn.copy_e('e', 'e'), fn.max('e', 'x'))
        
        return g.dstdata['x']
    
