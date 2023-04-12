import numpy as np
import torch
import dgl
from pyLattice2D.utils.record import RecordAndSave
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from torch.utils.data import Dataset, TensorDataset
import torchvision

class CustomDataset(Dataset):
    """
    TensorDataset with support of transforms.
    """
    def __init__(self, images, labels, transform=None):
        self.images = torch.Tensor(images)
        self.labels = torch.Tensor(labels)
        self.transform = transform

    def __getitem__(self, index):
        x = self.images[index]

        if self.transform:
            x = self.transform(x)

        y = self.labels[index]

        return x, y

    def __len__(self):
        return self.images.size(0)

def load_tabular_data(original_path, tabular_path, shapes):
    '''
    Load tabular data for predicting material properties.
    '''
    stiff = []
    pois = []
    features = []
    for shape in shapes:
        loader = RecordAndSave(original_path, shape)
        loader.load()
        stiff += loader['Stiff']
        pois += loader['Poisson']
        features += np.load('{}/{}.npy'.format(tabular_path, shape)).tolist()
    return stiff, pois, features

def turn_into_graph(coords, edges, stiff, pois, dens, self_loop = True):
    '''
    Convenience function for turning loaded data into dgl graphs.
    '''
    graphs = []
    stiffness = []
    density = []
    poisson = []
    for i in range(len(coords)):
        if stiff[i] > 0 and ~np.isnan(pois[i]):
            new_edges = []
            for ed in edges[i]:
                new_edges.append(list(ed))
                new_edges.append([ed[1], ed[0]])
            new_edges = np.array(new_edges)

            lattice_graph = dgl.graph((new_edges[:,0], new_edges[:,1]))
            if self_loop == True:
                lattice_graph = dgl.add_self_loop(lattice_graph)
            lattice_graph.ndata['coords'] = torch.Tensor(coords[i])

            graphs.append(lattice_graph)
            stiffness.append(stiff[i])
            density.append(dens[i])
            poisson.append(pois[i])
    return graphs, np.array(stiffness), np.array(poisson), np.array(density)
    
def load_data(path, geometries, self_loop = True):
    '''
    Load data generated with create.
    '''
    stiffness = []
    coordinates = []
    edges = []
    density = []
    poisson = []
    for geom in geometries:
        rec = RecordAndSave(path, geom)
        rec.load()
        stiffness += rec['Stiff']
        poisson += rec['Poisson']
        coordinates += rec['Coordinates']
        edges += rec['Edges']
        density += rec['Density']
    graphs, stiffness, poisson, density = turn_into_graph(coordinates, edges, stiffness, poisson, density, self_loop)
    return graphs, stiffness, poisson, density

    
def train(model, g, tgts, optimizer, lossf):
    '''
    Train a GNN.
    '''
    model.train()
    optimizer.zero_grad()
    y = model(g)
    loss = lossf(y, tgts)
    loss.backward()
    optimizer.step()

# function to test the model on testing graph g with ground truths tgts
def test(model, g, tgts, lossf, epoch):
    '''
    Test a GNN.
    '''
    model.eval()
    y = model(g)
    loss = lossf(y, tgts).cpu().detach().sqrt().numpy()
    lossmax = float(((y-tgts)**2).cpu().detach().sqrt().max().numpy())
    pearson = pearsonr(y.cpu().detach().numpy(), tgts.cpu().detach().numpy())
    
    mfit = LinearRegression(fit_intercept=True)
    mfit.fit(tgts.cpu().detach().numpy().reshape(-1, 1), y.cpu().detach().numpy())
    
    print('EPOCH {}: test root MSE = {} -- test max SE = {} --- test PearsonR: {} -- test Slope: {} -- test Intercept: {}'.format(epoch, loss, lossmax, pearson, mfit.coef_, mfit.intercept_))
    return loss, lossmax, mfit.coef_